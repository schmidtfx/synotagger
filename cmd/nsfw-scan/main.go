package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
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
	"io"
	"strings"
	"sync"
	"time"

	"golang.org/x/image/draw"
	ort "github.com/yalue/onnxruntime_go"
	"github.com/yourname/nsfw-go/internal/imagetools"
	"github.com/yourname/nsfw-go/internal/state"
)

type Result struct {
	Path       string             `json:"path"`
	Scores     map[string]float32 `json:"scores,omitempty"`
	NSFWScore  float32            `json:"nsfw_score"`
	Threshold  float32            `json:"threshold"`
	NSFW       bool               `json:"nsfw"`
	Err        string             `json:"err,omitempty"`
	DurationMs int64              `json:"duration_ms"`
}

func main() {
	// Required model config (names vary by model; pass via flags or .env/docker-compose)
	modelPath := flag.String("model", "/models/nsfwjs.onnx", "Path to ONNX model")
	inputName := flag.String("input-name", "input_1", "Model input tensor name")
	outputName := flag.String("output-name", "predictions", "Model output tensor name")
	nchw := flag.Bool("nchw", false, "Model expects NCHW (true) or NHWC (false)")

	// Preprocessing
	inputSize := flag.Int("size", 224, "Square resize to SIZE x SIZE")
	meanStr := flag.String("mean", "0.0,0.0,0.0", "RGB mean, comma-separated")
	stdStr := flag.String("std", "1.0,1.0,1.0", "RGB std, comma-separated")

	// Classes & scoring
	classList := flag.String("classes", "drawing,neutral,sexy,hentai,porn", "Comma-separated class names; for 2-class use 'sfw,nsfw'")
	nsfwExpr := flag.String("nsfw-expr", "sexy+hentai+porn", "How to compute NSFW score: e.g. 'nsfw' or 'sexy+hentai+porn' (use + to sum)")

	// Scan control
	photosDir := flag.String("dir", "/photos", "Directory to scan (recursive)")
	outPath := flag.String("out", "/output/results.jsonl", "Output JSONL file")
	csvPath := flag.String("csv", "", "Optional CSV output path")
	threshold := flag.Float64("th", 0.85, "NSFW threshold (0..1)")
	workers := flag.Int("workers", runtime.NumCPU(), "Parallel workers")

	// Tagging (Synology Photos picks up XMP dc:subject / IPTC Keywords)
	writeTags := flag.Bool("write-tags", false, "Write Synology-readable tags (XMP Subject & IPTC Keywords)")
	tagPrefix := flag.String("tag-prefix", "nsfw", "Base tag keyword to add")
	tagSubs := flag.String("tag-subs", "auto", "Subtags behavior: 'none'|'auto'|'all'")
	sidecar := flag.Bool("sidecar", false, "Write .xmp sidecar instead of modifying original files")

	// Persistent state / incremental scanning
	statePath := flag.String("state", "/output/state.db", "SQLite state file to remember processed files")
	skipUnchanged := flag.Bool("skip-unchanged", true, "Skip files whose size+mtime haven't changed")
	hashMode := flag.String("hash-mode", "changed", "none|changed|always: compute sha256 for rename detection & integrity")
	rehashDays := flag.Int("rehash-days", 30, "If >0 and last scan older than N days, recompute hash even if unchanged")

	// ONNX Runtime shared library
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
	if err != nil { log.Fatalf("open state: %v", err) }
	defer db.Close()

	session, err := ort.NewDynamicAdvancedSession(*modelPath, nil)
	if err != nil { log.Fatalf("session: %v", err) }
	defer session.Destroy()

	// Precompute normalization
	mean := imagetools.ParseTriplet(*meanStr, 0.0)
	std := imagetools.ParseTriplet(*stdStr, 1.0)
	classes := imagetools.SplitTrimComma(*classList)
	nsfwTerms := imagetools.SplitTrimPlus(*nsfwExpr) // split by +

	// Outputs
	outF, err := os.Create(*outPath)
	if err != nil { log.Fatalf("open out: %v", err) }
	defer outF.Close()

	var csvW *csv.Writer
	if *csvPath != "" {
		cf, e := os.Create(*csvPath); if e != nil { log.Fatalf("open csv: %v", e) }
		defer cf.Close()
		csvW = csv.NewWriter(cf)
		defer csvW.Flush()
		// CSV header
		h := []string{"path","nsfw","nsfw_score","threshold","duration_ms"}
		for _, c := range classes { h = append(h, "score_"+c) }
		_ = csvW.Write(h)
	}

	// Worker pool
	type job struct{ path string; info os.FileInfo }
	paths := make(chan job, 2048)
	var mu sync.Mutex // serialize writes
	var wg sync.WaitGroup

	worker := func() {
		defer wg.Done()
		for j := range paths {
			p := j.path
			fi := j.info
			rec, _ := db.Get(p)

			// quick skip based on size+mtime
			unchanged := rec != nil && rec.Size == fi.Size() && rec.MtimeNs == fi.ModTime().UnixNano()
			needHash := *hashMode == "always" || (*hashMode == "changed" && !unchanged)
			lastScanOld := false
			if rec != nil && *rehashDays > 0 {
				lastScanOld = time.Unix(rec.LastScanned,0).Before(time.Now().AddDate(0,0-*rehashDays))
			}
			if *skipUnchanged && unchanged && !lastScanOld {
				// nothing to do; still write a line for visibility
				res := Result{Path: p, NSFW: rec.NSFWScore >= float32(*threshold), NSFWScore: rec.NSFWScore, Threshold: float32(*threshold), DurationMs: 0}
				b, _ := json.Marshal(res)
				mu.Lock(); outF.Write(b); outF.Write([]byte("\n")); if csvW != nil { _ = csvW.Write([]string{res.Path, fmt.Sprintf("%t", res.NSFW), fmt.Sprintf("%.6f", res.NSFWScore), fmt.Sprintf("%.6f", res.Threshold), "0"}) }; mu.Unlock()
				continue
			}

			// Optional content hash (detect renames)
			var sumHex string
			if needHash || lastScanOld {
				if s, e := sha256OfFile(p); e == nil { sumHex = s } else { log.Printf("hash fail %s: %v", p, e) }
			}
			if sumHex != "" && rec == nil {
				// try find by hash (rename detection)
				if prev, _ := db.FindByHash(sumHex); prev != nil {
					rec = prev
				}
			}

			// Run classification
			res := classifyOne(session, p, *inputName, *outputName, *nchw, *inputSize, mean, std, classes, nsfwTerms, float32(*threshold))

			// Tagging
			if *writeTags && res.NSFW && res.Err == "" {
				subs := decideSubs(res, *tagSubs)
				if err := tagForSynology(p, *tagPrefix, subs, *sidecar); err != nil {
					log.Printf("exiftool tag failed for %s: %v", p, err)
				}
			}

			// Upsert state
			r := &state.Record{
				Path:        p,
				Size:        fi.Size(),
				MtimeNs:     fi.ModTime().UnixNano(),
				SHA256:      sumHex,
				LastScanned: state.NowUnix(),
				NSFWScore:   res.NSFWScore,
				TagsWritten: *writeTags && res.NSFW && res.Err == "",
				Err:         res.Err,
			}
			if err := db.Upsert(r); err != nil {
				log.Printf("state upsert failed for %s: %v", p, err)
			}

			// Persist result rows
			b, _ := json.Marshal(res)
			mu.Lock()
			outF.Write(b); outF.Write([]byte("\n"))
			if csvW != nil {
				row := []string{
					res.Path,
					fmt.Sprintf("%t", res.NSFW),
					fmt.Sprintf("%.6f", res.NSFWScore),
					fmt.Sprintf("%.6f", res.Threshold),
					fmt.Sprintf("%d", res.DurationMs),
				}
				for _, c := range classes {
					row = append(row, fmt.Sprintf("%.6f", res.Scores[c]))
				}
				_ = csvW.Write(row)
			}
			mu.Unlock()
		}
	}

	for i := 0; i < *workers; i++ { wg.Add(1); go worker() }

	// Walk the directory
	filepath.Walk(*photosDir, func(p string, info os.FileInfo, e error) error {
		if e != nil || info.IsDir() { return nil }
		lp := strings.ToLower(p)
		if strings.HasSuffix(lp, ".jpg") || strings.HasSuffix(lp, ".jpeg") || strings.HasSuffix(lp, ".png") {
			paths <- job{path:p, info:info}
		}
		return nil
	})
	close(paths)
	wg.Wait()
}

func classifyOne(session *ort.DynamicAdvancedSession, path, inName, outName string, nchw bool, size int, mean, std [3]float32, classes, nsfwTerms []string, th float32) Result {
	st := time.Now()
	buf, err := os.ReadFile(path)
	if err != nil { return Result{Path: path, Err: err.Error(), Threshold: th} }

	img, _, err := image.Decode(bytes.NewReader(buf))
	if err != nil { return Result{Path: path, Err: err.Error(), Threshold: th} }

	// Resize
	dst := image.NewRGBA(image.Rect(0, 0, size, size))
	draw.CatmullRom.Scale(dst, dst.Bounds(), img, img.Bounds(), draw.Over, nil)

	// To float32 tensor
	var data []float32
	if nchw {
		data = imagetools.ToCHW(dst, size, mean, std)
	} else {
		data = imagetools.ToHWC(dst, size, mean, std)
	}

	// Build input tensor
	var inShape ort.Shape
	if nchw { inShape = ort.NewShape(1, 3, int64(size), int64(size)) } else { inShape = ort.NewShape(1, int64(size), int64(size), 3) }
	inTensor, err := ort.NewTensor(inShape, data)
	if err != nil { return Result{Path: path, Err: err.Error(), Threshold: th} }
	defer inTensor.Destroy()

	// Prepare output tensor (length = len(classes) is typical; if different, we'll adjust from returned data)
	outTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(len(classes))))
	if err != nil { return Result{Path: path, Err: err.Error(), Threshold: th} }
	defer outTensor.Destroy()

	if err := session.Run(
		map[string]ort.Value{inName: inTensor},
		map[string]ort.Value{outName: outTensor},
	); err != nil {
		return Result{Path: path, Err: err.Error(), Threshold: th}
	}

	probs := softmax(outTensor.GetData())
	scores := map[string]float32{}
	L := min(len(classes), len(probs))
	for i := 0; i < L; i++ { scores[classes[i]] = probs[i] }

	// Compute NSFW via expression (sum of named classes or single 'nsfw')
	var nsfw float32
	for _, t := range nsfwTerms {
		if v, ok := scores[t]; ok { nsfw += v }
	}
	if len(nsfwTerms) == 1 && nsfwTerms[0] == "nsfw" && scores["nsfw"] == 0 && len(probs) == 2 {
		// Fallback for 2-class idx 1 if classes not named
		nsfw = probs[1]
	}

	return Result{
		Path: path,
		Scores: scores,
		NSFWScore: nsfw,
		Threshold: th,
		NSFW: nsfw >= th,
		DurationMs: time.Since(st).Milliseconds(),
	}
}

func softmax(x []float32) []float32 {
	if len(x) == 0 { return x }
	max := x[0]; for _, v := range x { if v > max { max = v } }
	sum := 0.0
	out := make([]float32, len(x))
	for i := range x {
		e := math.Exp(float64(x[i]-max))
		sum += e
		out[i] = float32(e)
	}
	for i := range out { out[i] = float32(float64(out[i]) / sum) }
	return out
}

func min(a, b int) int { if a < b { return a }; return b }

// decideSubs chooses subtags based on result scores and mode.
func decideSubs(res Result, mode string) []string {
	switch mode {
	case "all":
		out := []string{}
		for k, v := range res.Scores {
			if k == "sfw" || k == "neutral" || k == "drawing" { continue }
			if v > 0.2 { out = append(out, k) }
		}
		return out
	case "auto":
		var best string
		var bestVal float32
		for k, v := range res.Scores {
			if k == "sfw" || k == "neutral" || k == "drawing" { continue }
			if v > bestVal { bestVal, best = v, k }
		}
		if best != "" { return []string{best} }
		return nil
	default:
		return nil
	}
}

// tagForSynology writes XMP dc:subject & IPTC Keywords via exiftool; optional sidecar output.
func tagForSynology(path string, base string, subs []string, sidecar bool) error {
	args := []string{
		"-XMP-dc:Subject+="+base,
		"-IPTC:Keywords+="+base,
		"-P",                 // preserve file times
		"-overwrite_original",
	}
	for _, s := range subs {
		if s == "" { continue }
		args = append(args, "-XMP-dc:Subject+="+base+":"+s, "-IPTC:Keywords+="+base+":"+s)
	}
	if sidecar {
		args = append(args, "-o", "%d%f.xmp")
	}
	args = append(args, path)
	cmd := exec.Command("exiftool", args...)
	// Let output stream to container logs for visibility
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func sha256OfFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil { return "", err }
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil { return "", err }
	return hex.EncodeToString(h.Sum(nil)), nil
}
