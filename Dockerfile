# --- Build stage ---
FROM golang:1.24.4-bookworm AS build
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=1 GOOS=linux GOARCH=$(go env GOARCH) go build -o /out/nsfw-scan ./cmd/nsfw-scan

# --- Runtime stage ---
FROM debian:bookworm-slim
ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libgomp1 libimage-exiftool-perl \
 && rm -rf /var/lib/apt/lists/*

# Map TARGETARCH -> ORT archive label (amd64 -> x64, arm64 -> aarch64)
RUN archLabel=$([ "$TARGETARCH" = "arm64" ] && echo "aarch64" || echo "x64") \
 && ver=1.22.0 \
 && curl -L -o /tmp/ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v${ver}/onnxruntime-linux-${archLabel}-${ver}.tgz \
 && mkdir -p /usr/lib/onnx && tar -xzf /tmp/ort.tgz -C /usr/lib/onnx --strip-components=1 \
 && ln -s /usr/lib/onnx/lib/libonnxruntime.so.${ver} /usr/lib/libonnxruntime.so.${ver}

COPY --from=build /out/nsfw-scan /usr/local/bin/nsfw-scan

VOLUME ["/photos", "/output", "/models"]
ENTRYPOINT ["nsfw-scan", "-ort", "/usr/lib/libonnxruntime.so.1.22.0"]
