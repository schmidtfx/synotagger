package imagetools

import (
	"image"
	"image/color"
	"strconv"
	"strings"
)

func ParseTriplet(s string, def float64) [3]float32 {
	var out [3]float32
	parts := strings.Split(s, ",")
	for i := 0; i < 3; i++ {
		v := def
		if i < len(parts) {
			if f, err := strconv.ParseFloat(strings.TrimSpace(parts[i]), 32); err == nil {
				v = f
			}
		}
		out[i] = float32(v)
	}
	return out
}

func SplitTrimComma(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		t := strings.TrimSpace(p)
		if t != "" { out = append(out, t) }
	}
	return out
}

func SplitTrimPlus(s string) []string {
	parts := strings.Split(s, "+")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		t := strings.TrimSpace(p)
		if t != "" { out = append(out, t) }
	}
	return out
}

func ToHWC(img *image.RGBA, size int, mean, std [3]float32) []float32 {
	data := make([]float32, size*size*3)
	idx := 0
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			c := img.RGBAAt(x, y)
			r, g, b := normalize(c, mean, std)
			data[idx+0] = r
			data[idx+1] = g
			data[idx+2] = b
			idx += 3
		}
	}
	return data
}

func ToCHW(img *image.RGBA, size int, mean, std [3]float32) []float32 {
	data := make([]float32, 3*size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			c := img.RGBAAt(x, y)
			r, g, b := normalize(c, mean, std)
			i := y*size + x
			data[0*size*size+i] = r
			data[1*size*size+i] = g
			data[2*size*size+i] = b
		}
	}
	return data
}

func normalize(px color.RGBA, mean, std [3]float32) (r, g, b float32) {
	r = (float32(px.R)/255 - mean[0]) / std[0]
	g = (float32(px.G)/255 - mean[1]) / std[1]
	b = (float32(px.B)/255 - mean[2]) / std[2]
	return
}
