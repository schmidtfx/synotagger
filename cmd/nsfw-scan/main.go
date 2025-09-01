package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/schmidtfx/synotagger/internal/imagetools"
	"github.com/schmidtfx/synotagger/internal/state"
	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/image/draw"
)

type Result struct {
	Path       string             `json:"path"`
	Scores     map[string]float32 `json:"scores,omitempty"`
	NSFWScore  float32            `json:"nsfw_score"`
	Threshold  float32            `json:"threshold"`
	NSFW       bool               `json:"nsfw"`
	Err        string             `json:"err,omitempty"`
	DurationMs int64              `json:"duration_ms"`
	TopClass   string             `json:"top_class,omitempty"`
	NumDet     int                `json:"num_det,omitempty"`
}

func main() {
	modelType := flag.String("model-type", "classifier", "classifier|detector")
	modelPath := flag.String("model", "/models/nsfwjs.onnx", "Path to ONNX model")
	inputName := flag.String("input-name", "input_1", "Model input tensor name")
	outputName := flag.String("output-name", "predictions", "Model output tensor name")
	nchw := flag.Bool("nchw", false, "Model expects NCHW (true) or NHWC (false)")

	inputSize := flag.Int("size", 224, "Square resize to SIZE x SIZE")
	meanStr := flag.String("mean", "0.0,0.0,0.0", "RGB mean, comma-separated")
	stdStr := flag.String("std", "1.0,1.0,1.0", "RGB std, comma-separated")

	classList := flag.String("classes", "drawing,hentai,neutral,porn,sexy", "Comma-separated class names")
	nsfwExpr := flag.String("nsfw-expr", "porn+sexy+hentai", "Classifier sum; ignored for detector")

	detConf := flag.Float64("det-conf", 0.25, "Detector conf threshold (obj * class prob)")
	detIou := flag.Float64("det-iou", 0.45, "Detector NMS IoU threshold")
	detMax := flag.Int("det-max", 300, "Max detections after NMS")

	threshold := flag.Float64("th", 0.85, "Photo-level threshold (0..1)")

	photosDir := flag.String("dir", "/photos", "Directory to scan (recursive)")
	outPath := flag.String("out", "/output/results.jsonl", "Output JSONL file")
	csvPath := flag.String("csv", "", "Optional CSV output path")
	workers := flag.Int("workers", runtime.NumCPU(), "Parallel workers")

	writeTags := flag.Bool("write-tags", false, "Write Synology-readable tags (XMP Subject & IPTC Keywords)")
	tagPrefix := flag.String("tag-prefix", "nsfw", "Base tag keyword to add")
	tagSubs := flag.String("tag-subs", "auto", "Subtags behavior: 'none'|'auto'|'all'")
	sidecar := flag.Bool("sidecar", false, "Write .xmp sidecar instead of modifying original files")

	statePath := flag.String("state", "/output/state.db", "SQLite state file")
	skipUnchanged := flag.Bool("skip-unchanged", true, "Skip files whose size+mtime haven't changed")
	rehashDays := flag.Int("rehash-days", 30, "Periodic rescan window in days")

	ortLib := flag.String("ort", "/usr/lib/libonnxruntime.so.1.22.0", "Path to onnxruntime shared library")
	flag.Parse()

	ort.SetSharedLibraryPath(*ortLib)
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("onnxruntime init: %v", err)
	}
	defer ort.DestroyEnvironment()

	db, err := state.Open(*statePath)
	if err != nil {
		log.Fatalf("open state: %v", err)
	}
	defer db.Close()

	session, err := ort.NewDynamicAdvancedSession(*modelPath, []string{*inputName}, []string{*outputName}, nil)
	if err != nil {
		log.Fatalf("session: %v", err)
	}
	defer session.Destroy()

	mean := imagetools.ParseTriplet(*meanStr, 0.0)
	std := imagetools.ParseTriplet(*stdStr, 1.0)
	classes := imagetools.SplitTrimComma(*classList)
	nsfwTerms := imagetools.SplitTrimPlus(*nsfwExpr)

	outF, err := os.Create(*outPath)
	if err != nil {
		log.Fatalf("open out: %v", err)
	}
	defer outF.Close()

	var csvW *csv.Writer
	if *csvPath != "" {
		cf, e := os.Create(*csvPath)
		if e != nil {
			log.Fatalf("open csv: %v", e)
		}
		defer cf.Close()
		csvW = csv.NewWriter(cf)
		defer csvW.Flush()
		h := []string{"path", "nsfw", "nsfw_score", "threshold", "duration_ms"}
		if *modelType == "detector" {
			h = append(h, "top_class", "num_det")
		}
		for _, c := range classes {
			h = append(h, "score_"+c)
		}
		_ = csvW.Write(h)
	}

	type job struct {
		path string
		info os.FileInfo
	}
	paths := make(chan job, 2048)
	var mu sync.Mutex
	var wg sync.WaitGroup

	worker := func() {
		defer wg.Done()
		for j := range paths {
			p := j.path
			fi := j.info
			rec, _ := db.Get(p)
			unchanged := rec != nil && rec.Size == fi.Size() && rec.MtimeNs == fi.ModTime().UnixNano()
			lastScanOld := false
			if rec != nil && *rehashDays > 0 {
				lastScanOld = time.Unix(rec.LastScanned, 0).Before(time.Now().AddDate(0, 0, -(*rehashDays)))
			}
			if *skipUnchanged && unchanged && !lastScanOld {
				res := Result{Path: p, NSFW: rec.NSFWScore >= float32(*threshold), NSFWScore: rec.NSFWScore, Threshold: float32(*threshold)}
				writeRow(&mu, outF, csvW, classes, res)
				continue
			}

			var res Result
			if *modelType == "detector" {
				res = runDetector(session, p, *inputName, *outputName, *nchw, *inputSize, mean, std, classes, float32(*detConf), float32(*detIou), *detMax, float32(*threshold))
			} else {
				res = runClassifier(session, p, *inputName, *outputName, *nchw, *inputSize, mean, std, classes, nsfwTerms, float32(*threshold))
			}

			if *writeTags && res.NSFW && res.Err == "" {
				subs := decideSubs(res, *modelType, *tagSubs)
				if err := tagForSynology(p, *tagPrefix, subs, *sidecar); err != nil {
					log.Printf("exiftool tag failed for %s: %v", p, err)
				}
			}

			r := &state.Record{
				Path:        p,
				Size:        fi.Size(),
				MtimeNs:     fi.ModTime().UnixNano(),
				LastScanned: time.Now().Unix(),
				NSFWScore:   res.NSFWScore,
				TagsWritten: *writeTags && res.NSFW && res.Err == "",
				Err:         res.Err,
			}
			if err := db.Upsert(r); err != nil {
				log.Printf("state upsert failed for %s: %v", p, err)
			}

			writeRow(&mu, outF, csvW, classes, res)
		}
	}

	for i := 0; i < *workers; i++ {
		wg.Add(1)
		go worker()
	}

	filepath.Walk(*photosDir, func(p string, info os.FileInfo, e error) error {
		if e != nil || info.IsDir() {
			return nil
		}
		lp := strings.ToLower(p)
		if strings.HasSuffix(lp, ".jpg") || strings.HasSuffix(lp, ".jpeg") || strings.HasSuffix(lp, ".png") {
			paths <- job{path: p, info: info}
		}
		return nil
	})
	close(paths)
	wg.Wait()
}

func writeRow(mu *sync.Mutex, outF *os.File, csvW *csv.Writer, classes []string, res Result) {
	b, _ := json.Marshal(res)
	mu.Lock()
	outF.Write(b)
	outF.Write([]byte("\n"))
	if csvW != nil {
		row := []string{
			res.Path,
			fmt.Sprintf("%t", res.NSFW),
			fmt.Sprintf("%.6f", res.NSFWScore),
			fmt.Sprintf("%.6f", res.Threshold),
			fmt.Sprintf("%d", res.DurationMs),
		}
		if res.TopClass != "" || res.NumDet > 0 {
			row = append(row, res.TopClass, fmt.Sprintf("%d", res.NumDet))
		}
		for _, c := range classes {
			row = append(row, fmt.Sprintf("%.6f", res.Scores[c]))
		}
		_ = csvW.Write(row)
	}
	mu.Unlock()
}

// ---- Classifier ----
func runClassifier(session *ort.DynamicAdvancedSession, path, inName, outName string, nchw bool, size int, mean, std [3]float32, classes, nsfwTerms []string, th float32) Result {
	st := time.Now()
	buf, err := os.ReadFile(path)
	if err != nil {
		return Result{Path: path, Err: err.Error(), Threshold: th}
	}
	img, _, err := image.Decode(bytes.NewReader(buf))
	if err != nil {
		return Result{Path: path, Err: err.Error(), Threshold: th}
	}
	dst := image.NewRGBA(image.Rect(0, 0, size, size))
	draw.CatmullRom.Scale(dst, dst.Bounds(), img, img.Bounds(), draw.Over, nil)
	var data []float32
	if nchw {
		data = imagetools.ToCHW(dst, size, mean, std)
	} else {
		data = imagetools.ToHWC(dst, size, mean, std)
	}
	var inShape ort.Shape
	if nchw {
		inShape = ort.NewShape(1, 3, int64(size), int64(size))
	} else {
		inShape = ort.NewShape(1, int64(size), int64(size), 3)
	}
	inTensor, err := ort.NewTensor(inShape, data)
	if err != nil {
		return Result{Path: path, Err: err.Error(), Threshold: th}
	}
	defer inTensor.Destroy()
	outTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(len(classes))))
	if err != nil {
		return Result{Path: path, Err: err.Error(), Threshold: th}
	}
	defer outTensor.Destroy()
	if err := session.Run([]ort.Value{inTensor}, []ort.Value{outTensor}); err != nil {
		return Result{Path: path, Err: err.Error(), Threshold: th}
	}
	probs := softmax(outTensor.GetData())
	scores := map[string]float32{}
	L := min(len(classes), len(probs))
	for i := 0; i < L; i++ {
		scores[classes[i]] = probs[i]
	}
	var nsfw float32
	for _, t := range nsfwTerms {
		if v, ok := scores[t]; ok {
			nsfw += v
		}
	}
	if len(nsfwTerms) == 1 && nsfwTerms[0] == "nsfw" && len(probs) == 2 {
		nsfw = probs[1]
	}
	return Result{Path: path, Scores: scores, NSFWScore: nsfw, Threshold: th, NSFW: nsfw >= th, DurationMs: time.Since(st).Milliseconds()}
}

// ---- Detector (YOLO-style [x,y,w,h,obj,cls...]) ----
type det struct {
	x1, y1, x2, y2, score float32
	cls                   int
}

func runDetector(session *ort.DynamicAdvancedSession, path, inName, outName string, nchw bool, size int, mean, std [3]float32, labels []string, confTh, iouTh float32, maxDet int, th float32) Result {
	st := time.Now()
	buf, err := os.ReadFile(path)
	if err != nil {
		return Result{Path: path, Err: err.Error(), Threshold: th}
	}
	img, _, err := image.Decode(bytes.NewReader(buf))
	if err != nil {
		return Result{Path: path, Err: err.Error(), Threshold: th}
	}

	dst := image.NewRGBA(image.Rect(0, 0, size, size))
	draw.CatmullRom.Scale(dst, dst.Bounds(), img, img.Bounds(), draw.Over, nil)

	var data []float32
	if nchw {
		data = imagetools.ToCHW(dst, size, mean, std)
	} else {
		data = imagetools.ToHWC(dst, size, mean, std)
	}

	var inShape ort.Shape
	if nchw {
		inShape = ort.NewShape(1, 3, int64(size), int64(size))
	} else {
		inShape = ort.NewShape(1, int64(size), int64(size), 3)
	}
	inTensor, err := ort.NewTensor(inShape, data)
	if err != nil {
		return Result{Path: path, Err: err.Error(), Threshold: th}
	}
	defer inTensor.Destroy()

	// Allocate generic output tensor; model will fill what it needs
	outTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(8400), int64(5+len(labels))))
	if err != nil {
		return Result{Path: path, Err: err.Error(), Threshold: th}
	}
	defer outTensor.Destroy()

	if err := session.Run([]ort.Value{inTensor}, []ort.Value{outTensor}); err != nil {
		return Result{Path: path, Err: err.Error(), Threshold: th}
	}
	raw := outTensor.GetData()
	numAttrs := 5 + len(labels)
	N := len(raw) / numAttrs
	dets := make([]det, 0, N)
	for i := 0; i < N; i++ {
		base := i * numAttrs
		obj := raw[base+4]
		if obj < 1e-6 {
			continue
		}
		bestC, bestP := -1, float32(0)
		for c := 0; c < len(labels); c++ {
			p := raw[base+5+c]
			if p > bestP {
				bestP, bestC = p, c
			}
		}
		score := obj * bestP
		if score < confTh {
			continue
		}
		x := raw[base+0]
		y := raw[base+1]
		w := raw[base+2]
		h := raw[base+3]
		x1 := x - w/2
		y1 := y - h/2
		x2 := x + w/2
		y2 := y + h/2
		dets = append(dets, det{x1, y1, x2, y2, score, bestC})
	}
	dets = nmsPerClass(dets, iouTh, maxDet)
	var maxScore float32
	var top string
	for _, d := range dets {
		if d.score > maxScore {
			maxScore = d.score
			if d.cls >= 0 && d.cls < len(labels) {
				top = labels[d.cls]
			}
		}
	}
	return Result{Path: path, NSFWScore: maxScore, Threshold: th, NSFW: maxScore >= th, TopClass: top, NumDet: len(dets), DurationMs: time.Since(st).Milliseconds()}
}

func nmsPerClass(dets []det, iouTh float32, maxDet int) []det {
	m := map[int][]det{}
	for _, d := range dets {
		m[d.cls] = append(m[d.cls], d)
	}
	out := make([]det, 0, len(dets))
	for _, arr := range m {
		// sort by score desc
		for i := 0; i < len(arr); i++ {
			k := i
			for j := i + 1; j < len(arr); j++ {
				if arr[j].score > arr[k].score {
					k = j
				}
			}
			arr[i], arr[k] = arr[k], arr[i]
		}
		keep := []det{}
		for _, d := range arr {
			ok := true
			for _, k := range keep {
				if iou(d, k) > iouTh {
					ok = false
					break
				}
			}
			if ok {
				keep = append(keep, d)
				if len(keep) >= maxDet {
					break
				}
			}
		}
		out = append(out, keep...)
	}
	return out
}
func iou(a, b det) float32 {
	ix1 := maxf(a.x1, b.x1)
	iy1 := maxf(a.y1, b.y1)
	ix2 := minf(a.x2, b.x2)
	iy2 := minf(a.y2, b.y2)
	iw := maxf(0, ix2-ix1)
	ih := maxf(0, iy2-iy1)
	inter := iw * ih
	areaA := (a.x2 - a.x1) * (a.y2 - a.y1)
	areaB := (b.x2 - b.x1) * (b.y2 - b.y1)
	union := areaA + areaB - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}
func maxf(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
func minf(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func decideSubs(res Result, modelType, mode string) []string {
	switch mode {
	case "all", "auto":
		if modelType == "detector" {
			if res.TopClass != "" {
				return []string{res.TopClass}
			}
			return nil
		}
		// classifier path omitted for brevity
		return nil
	default:
		return nil
	}
}

func tagForSynology(path string, base string, subs []string, sidecar bool) error {
	args := []string{
		"-XMP-dc:Subject+=" + base,
		"-IPTC:Keywords+=" + base,
		"-P",
		"-overwrite_original",
	}
	for _, s := range subs {
		if s == "" {
			continue
		}
		args = append(args, "-XMP-dc:Subject+="+base+":"+s, "-IPTC:Keywords+="+base+":"+s)
	}
	if sidecar {
		args = append(args, "-o", "%d%f.xmp")
	}
	args = append(args, path)
	cmd := exec.Command("exiftool", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func softmax(x []float32) []float32 {
	if len(x) == 0 {
		return x
	}
	max := x[0]
	for _, v := range x {
		if v > max {
			max = v
		}
	}
	sum := 0.0
	out := make([]float32, len(x))
	for i := range x {
		e := math.Exp(float64(x[i] - max))
		sum += e
		out[i] = float32(e)
	}
	for i := range out {
		out[i] = float32(float64(out[i]) / sum)
	}
	return out
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
