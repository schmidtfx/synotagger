// cmd/nsfw-scan/main.go
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
	Scores     map[string]float32 `json:"scores,omitempty"` // classifier per-class
	NSFWScore  float32            `json:"nsfw_score"`
	Threshold  float32            `json:"threshold"`
	NSFW       bool               `json:"nsfw"`
	Err        string             `json:"err,omitempty"`
	DurationMs int64              `json:"duration_ms"`

	// detector-only fields
	TopClass string `json:"top_class,omitempty"`
	NumDet   int    `json:"num_det,omitempty"`
}

func main() {
	// --- Model / preprocessing ---
	modelType := flag.String("model-type", "classifier", "classifier|detector (YOLO-style)")
	modelPath := flag.String("model", "/models/nsfwjs.onnx", "Path to ONNX model")
	inputName := flag.String("input-name", "input_1", "Model input tensor name")
	outputName := flag.String("output-name", "predictions", "Model output tensor name")
	nchw := flag.Bool("nchw", false, "Model expects NCHW (true) or NHWC (false)")
	inputSize := flag.Int("size", 224, "Square resize (e.g., 224, 320, 640)")
	meanStr := flag.String("mean", "0.0,0.0,0.0", "RGB mean (e.g., 0.485,0.456,0.406)")
	stdStr := flag.String("std", "1.0,1.0,1.0", "RGB std (e.g., 0.229,0.224,0.225)")

	// --- Scoring / labels ---
	classList := flag.String("classes", "drawing,hentai,neutral,porn,sexy", "Comma-separated labels. Detector uses these as class names.")
	nsfwExpr := flag.String("nsfw-expr", "porn+sexy+hentai", "Classifier: sum of these class names (use +). Ignored for detector.")

	// --- Detector knobs ---
	detConf := flag.Float64("det-conf", 0.25, "Detector: min (obj * classProb)")
	detIou := flag.Float64("det-iou", 0.45, "Detector: NMS IoU threshold")
	detMax := flag.Int("det-max", 300, "Detector: max kept detections after NMS")

	// --- Photo-level decision threshold ---
	threshold := flag.Float64("th", 0.85, "Photo-level threshold for NSFW decision")

	// --- Scan / output ---
	photosDir := flag.String("dir", "/photos", "Directory to scan (recursive)")
	outPath := flag.String("out", "/output/results.jsonl", "Output JSONL path")
	csvPath := flag.String("csv", "", "Optional CSV output path")
	workers := flag.Int("workers", runtime.NumCPU(), "Parallel workers")

	// --- Tagging for Synology Photos ---
	writeTags := flag.Bool("write-tags", false, "Write XMP dc:subject & IPTC Keywords")
	tagPrefix := flag.String("tag-prefix", "nsfw", "Base tag (e.g., nsfw)")
	tagSubs := flag.String("tag-subs", "auto", "Subtags: none|auto|all")
	sidecar := flag.Bool("sidecar", false, "Write .xmp sidecars instead of touching originals")

	// --- State / incremental ---
	statePath := flag.String("state", "/output/state.db", "SQLite state file")
	skipUnchanged := flag.Bool("skip-unchanged", true, "Skip files when size+mtime unchanged")
	rehashDays := flag.Int("rehash-days", 30, "Periodic rescan window in days (0=disable)")

	// --- ONNX Runtime ---
	ortLib := flag.String("ort", "/usr/lib/libonnxruntime.so.1.22.0", "Path to onnxruntime shared library")

	flag.Parse()

	// Initialize ORT
	ort.SetSharedLibraryPath(*ortLib)
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("onnxruntime init: %v", err)
	}
	defer ort.DestroyEnvironment()

	// State DB
	db, err := state.Open(*statePath)
	if err != nil {
		log.Fatalf("state open: %v", err)
	}
	defer db.Close()

	// Session
	session, err := ort.NewDynamicAdvancedSession(*modelPath, []string{*inputName}, []string{*outputName}, nil)
	if err != nil {
		log.Fatalf("session: %v", err)
	}
	defer session.Destroy()

	// Precompute normalization & labels
	mean := imagetools.ParseTriplet(*meanStr, 0.0)
	std := imagetools.ParseTriplet(*stdStr, 1.0)
	classes := imagetools.SplitTrimComma(*classList)
	nsfwTerms := imagetools.SplitTrimPlus(*nsfwExpr) // "a+b+c" -> ["a","b","c"]

	// Outputs
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
		header := []string{"path", "nsfw", "nsfw_score", "threshold", "duration_ms"}
		if *modelType == "detector" {
			header = append(header, "top_class", "num_det")
		}
		for _, c := range classes {
			header = append(header, "score_"+c)
		}
		_ = csvW.Write(header)
	}

	// Worker pool
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

			// Incremental skip (size+mtime)
			rec, _ := db.Get(p)
			unchanged := rec != nil && rec.Size == fi.Size() && rec.MtimeNs == fi.ModTime().UnixNano()
			lastScanOld := false
			if rec != nil && *rehashDays > 0 {
				lastScanOld = time.Unix(rec.LastScanned, 0).Before(time.Now().AddDate(0, 0, -(*rehashDays)))
			}
			if *skipUnchanged && unchanged && !lastScanOld {
				// Reuse previous score for visibility in outputs
				res := Result{
					Path:      p,
					NSFWScore: rec.NSFWScore,
					Threshold: float32(*threshold),
					NSFW:      rec.NSFWScore >= float32(*threshold),
				}
				writeRow(&mu, outF, csvW, classes, res)
				continue
			}

			// Run inference depending on model type
			var res Result
			if *modelType == "detector" {
				res = runDetectorDualLayout(session, p, *inputName, *outputName, *nchw, *inputSize, mean, std, classes, float32(*detConf), float32(*detIou), *detMax, float32(*threshold))
			} else {
				res = runClassifier(session, p, *inputName, *outputName, *nchw, *inputSize, mean, std, classes, nsfwTerms, float32(*threshold))
			}

			// Tagging
			if *writeTags && res.NSFW && res.Err == "" {
				subs := decideSubs(res, *modelType, *tagSubs)
				if err := tagForSynology(p, *tagPrefix, subs, *sidecar); err != nil {
					log.Printf("exiftool tag failed for %s: %v", p, err)
				}
			}

			// Upsert state
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

	// Walk directory
	filepath.Walk(*photosDir, func(p string, info os.FileInfo, e error) error {
		if e != nil || info.IsDir() {
			return nil
		}

		// Skip Synology thumbs/caches
		lp := strings.ToLower(p)
		if strings.Contains(lp, "/@eadir/") || strings.Contains(lp, "\\@eadir\\") {
			return nil
		}
		if strings.Contains(lp, "synophoto_thumb") {
			return nil
		}

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

// -------------------- Classifier path --------------------

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

	// Resize
	dst := image.NewRGBA(image.Rect(0, 0, size, size))
	draw.CatmullRom.Scale(dst, dst.Bounds(), img, img.Bounds(), draw.Over, nil)

	// Make input
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

	// Output tensor
	outTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(len(classes))))
	if err != nil {
		return Result{Path: path, Err: err.Error(), Threshold: th}
	}
	defer outTensor.Destroy()

	// Run
	if err := session.Run([]ort.Value{inTensor}, []ort.Value{outTensor}); err != nil {
		return Result{Path: path, Err: err.Error(), Threshold: th}
	}

	// Post-process
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

	return Result{
		Path:       path,
		Scores:     scores,
		NSFWScore:  nsfw,
		Threshold:  th,
		NSFW:       nsfw >= th,
		DurationMs: time.Since(st).Milliseconds(),
	}
}

// -------------------- Detector path (YOLO-style) --------------------

type det struct {
	x1, y1, x2, y2 float32
	score          float32
	cls            int
}

// runDetectorDualLayout supports both [1, N, 5+classes] and [1, 5+classes, N]
func runDetectorDualLayout(session *ort.DynamicAdvancedSession, path, inName, outName string, nchw bool, size int, mean, std [3]float32, labels []string, confTh, iouTh float32, maxDet int, th float32) Result {
	st := time.Now()

	buf, err := os.ReadFile(path)
	if err != nil {
		return Result{Path: path, Err: err.Error(), Threshold: th}
	}
	img, _, err := image.Decode(bytes.NewReader(buf))
	if err != nil {
		return Result{Path: path, Err: err.Error(), Threshold: th}
	}

	// Resize
	dst := image.NewRGBA(image.Rect(0, 0, size, size))
	draw.CatmullRom.Scale(dst, dst.Bounds(), img, img.Bounds(), draw.Over, nil)

	// Input
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

	attrs := 5 + len(labels)

	// First try standard layout [1, N, 5+classes] (e.g., 8400 x 22)
	outTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(8400), int64(attrs)))
	if err != nil {
		return Result{Path: path, Err: err.Error(), Threshold: th}
	}
	defer outTensor.Destroy()

	runErr := session.Run([]ort.Value{inTensor}, []ort.Value{outTensor})
	if runErr == nil {
		return parseDetectionsStandard(outTensor.GetData(), labels, confTh, iouTh, maxDet, th, path, st)
	}

	// Fallback: transposed layout [1, 5+classes, N] (e.g., 22 x 8400)
	outTensor2, err2 := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(attrs), int64(8400)))
	if err2 != nil {
		return Result{Path: path, Err: err2.Error(), Threshold: th}
	}
	defer outTensor2.Destroy()

	if err = session.Run([]ort.Value{inTensor}, []ort.Value{outTensor2}); err != nil {
		// still failing? return the original error
		return Result{Path: path, Err: runErr.Error(), Threshold: th}
	}
	return parseDetectionsTransposed(outTensor2.GetData(), labels, confTh, iouTh, maxDet, th, path, st)
}

// raw layout: [N, 5+classes]
func parseDetectionsStandard(raw []float32, labels []string, confTh, iouTh float32, maxDet int, th float32, path string, st time.Time) Result {
	numAttrs := 5 + len(labels)
	N := len(raw) / numAttrs
	dets := make([]det, 0, N)
	for i := 0; i < N; i++ {
		base := i * numAttrs
		obj := raw[base+4]
		if obj <= 0 {
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
	return finalizeDetections(dets, labels, iouTh, maxDet, th, path, st)
}

// raw layout: [5+classes, N] (transposed)
func parseDetectionsTransposed(raw []float32, labels []string, confTh, iouTh float32, maxDet int, th float32, path string, st time.Time) Result {
	numAttrs := 5 + len(labels)
	N := len(raw) / numAttrs
	dets := make([]det, 0, N)
	for j := 0; j < N; j++ {
		obj := raw[4*N+j]
		if obj <= 0 {
			continue
		}
		bestC, bestP := -1, float32(0)
		for c := 0; c < len(labels); c++ {
			p := raw[(5+c)*N+j]
			if p > bestP {
				bestP, bestC = p, c
			}
		}
		score := obj * bestP
		if score < confTh {
			continue
		}
		x := raw[0*N+j]
		y := raw[1*N+j]
		w := raw[2*N+j]
		h := raw[3*N+j]
		x1 := x - w/2
		y1 := y - h/2
		x2 := x + w/2
		y2 := y + h/2
		dets = append(dets, det{x1, y1, x2, y2, score, bestC})
	}
	return finalizeDetections(dets, labels, iouTh, maxDet, th, path, st)
}

func finalizeDetections(dets []det, labels []string, iouTh float32, maxDet int, th float32, path string, st time.Time) Result {
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
	return Result{
		Path:       path,
		NSFWScore:  maxScore,
		Threshold:  th,
		NSFW:       maxScore >= th,
		TopClass:   top,
		NumDet:     len(dets),
		DurationMs: time.Since(st).Milliseconds(),
	}
}

func nmsPerClass(dets []det, iouTh float32, maxDet int) []det {
	byClass := map[int][]det{}
	for _, d := range dets {
		byClass[d.cls] = append(byClass[d.cls], d)
	}
	out := make([]det, 0, len(dets))
	for _, arr := range byClass {
		// sort by score desc (selection sort—small arrays)
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

// -------------------- Tagging helpers --------------------

func decideSubs(res Result, modelType, mode string) []string {
	switch mode {
	case "none":
		return nil
	case "auto":
		if modelType == "detector" {
			if res.TopClass != "" {
				return []string{res.TopClass}
			}
			return nil
		}
		// classifier: pick top non-neutral class if available
		var best string
		var bestVal float32
		for k, v := range res.Scores {
			if k == "sfw" || k == "neutral" || k == "drawing" {
				continue
			}
			if v > bestVal {
				bestVal, best = v, k
			}
		}
		if best != "" {
			return []string{best}
		}
		return nil
	case "all":
		if modelType == "detector" {
			if res.TopClass != "" {
				return []string{res.TopClass}
			}
			return nil
		}
		out := []string{}
		for k, v := range res.Scores {
			if k == "sfw" || k == "neutral" || k == "drawing" {
				continue
			}
			if v > 0.2 {
				out = append(out, k)
			}
		}
		return out
	default:
		return nil
	}
}

func tagForSynology(path string, base string, subs []string, sidecar bool) error {
	args := []string{
		"-XMP-dc:Subject+=" + base,
		"-IPTC:Keywords+=" + base,
		"-P", // preserve file times
		"-overwrite_original",
	}
	for _, s := range subs {
		if s == "" {
			continue
		}
		args = append(args,
			"-XMP-dc:Subject+"+base+":"+s,
			"-IPTC:Keywords+"+base+":"+s,
		)
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

// -------------------- utilities --------------------

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
	} else {
		return b
	}
}
func maxf(a, b float32) float32 {
	if a > b {
		return a
	} else {
		return b
	}
}
func minf(a, b float32) float32 {
	if a < b {
		return a
	} else {
		return b
	}
}
