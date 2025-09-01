# nsfw-go

Fast NSFW folder scanner in Go using ONNX Runtime. Designed for Synology via Docker.

## Features
- Recursive scan of a photo directory (JPEG/PNG)
- ONNX classifier inference (e.g. NSFWJS 5-class or 2-class OpenNSFW)
- JSONL + optional CSV output
- Optional quarantine: move files over a threshold
- **Tagging for Synology Photos**: writes XMP dc:subject & IPTC Keywords (or XMP sidecars) so Photos picks them up as tags

## Quick start (Synology)
1. Put your model at `/volume1/models/nsfwjs.onnx`.
2. Update `docker-compose.yml` if your model’s input/output names, layout, or classes differ.
3. In Container Manager or SSH:
   ```bash
   cd /volume1/docker/nsfw-go
   docker compose build
   docker compose up -d
   ```
4. Results:
   - JSONL: `/volume1/nsfw-output/results.jsonl`
   - CSV: `/volume1/nsfw-output/results.csv`
   - Quarantine (if enabled): `/volume1/nsfw-output/quarantine`
5. In **Synology Photos**, run a rescan so the new tags are indexed. Then filter/search by `nsfw` (and subtags like `nsfw:sexy`).

## Tagging behavior
- Enable with `-write-tags`. Base tag is `nsfw` (`-tag-prefix`), plus optional subtags:
  - `-tag-subs auto` (default): add top-1 indicative class (e.g., `nsfw:porn`).
  - `-tag-subs all`: add all indicative classes above 0.2.
  - `-tag-subs none`: only the base tag.
- `-sidecar` writes XMP sidecars instead of touching originals (recommended for HEIC/CR3 and for non-destructive workflows).
- Requires ExifTool (already included in the image).

## Models
- **NSFWJS (5-class)**: input `input_1`, output `predictions`, NHWC, size 224, classes `drawing,neutral,sexy,hentai,porn`, nsfw score = `sexy+hentai+porn`.
- **OpenNSFW (2-class)**: set `-classes "sfw,nsfw"` and `-nsfw-expr "nsfw"`. Some variants use NCHW; set `-nchw=true`.
- **ImageNet normalization**: use `-mean 0.485,0.456,0.406 -std 0.229,0.224,0.225` if required by your model.

## CLI
```bash
nsfw-scan -model /models/model.onnx   -input-name INPUT -output-name OUTPUT   -nchw=false -size 224 -mean 0,0,0 -std 1,1,1   -classes drawing,neutral,sexy,hentai,porn   -nsfw-expr sexy+hentai+porn   -dir /photos -out /output/results.jsonl -csv /output/results.csv   -th 0.85 -workers 4 -move /output/quarantine   -write-tags -tag-prefix nsfw -tag-subs auto -sidecar
```

## Notes
- ONNX Runtime 1.22.0 is bundled to avoid ABI mismatches with the Go wrapper.
- If you’re unsure of your model’s input/output names, inspect the ONNX with Netron.
- Performance tips: increase `-workers` up to your CPU core count; keep the photos volume on local storage.

## License
MIT


## Tracking processed files
The tool keeps a **state DB** (bbolt) at `/output/state.db` by default. It records file size, mtime, optional SHA-1, and the last NSFW result.

Flags:
- `-state /output/state.db` — path to the DB
- `-skip-known=true` — skip files already processed and unchanged
- `-hash sha1|none` — enable content hashing for stronger change detection (slower I/O)

Change detection logic:
1) If path not in DB → process.
2) If size or mtime changed → process.
3) If `-hash sha1` is enabled, compare file SHA-1; if different → process.
4) Otherwise, skip and reuse previous result.



## Incremental scanning & state
The tool keeps a SQLite state DB (default `/output/state.db`) to skip already-processed files.
- **Skip unchanged**: compares size+mtime (fast). `-skip-unchanged=true`
- **Hash modes**: `-hash-mode none|changed|always`
  - `changed`: hash only when size/mtime changed (detects renames after edits)
  - `always`: hash every file (strongest rename detection; slower)
- **Rehash window**: `-rehash-days 30` forces a periodic rehash to guard against silent bitrot.

You can point `-state` to any writable path (e.g., `/volume1/nsfw-output/state.db`).
