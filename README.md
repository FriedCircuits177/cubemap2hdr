# cube2hdr

Convert six square cubemap faces (JPG/PNG) into one equirectangular panorama in
Radiance `.hdr` format.

Three dependencies: `image` (decoders only), `rayon` (parallelism), `clap` (CLI).
The RGBE writer is implemented in-tree, so no HDR library is needed.

## Build

```sh
cargo build --release
cargo test            # projection + RGBE round-trip + RLE round-trip
```

## Usage

```sh
./target/release/cube2hdr \
  --px right.jpg --nx left.jpg \
  --py top.jpg   --ny bottom.jpg \
  --pz back.jpg  --nz front.jpg \
  --output pano.hdr
```

Defaults: output width `4 x face_size` (height = width/2), sRGB inputs
linearised on load, bilinear sampling with 2x2 supersampling, RLE-compressed
output.

| Flag | Default | Notes |
|---|---|---|
| `--width` | `4 x face size` | Height is always half |
| `--filter` | `bilinear` | `bicubic` (Catmull-Rom) is sharper when upscaling past 4x |
| `--samples` | `2` | N x N supersample grid per output pixel, 1-8 |
| `--input-transfer` | `srgb` | Use `linear` if faces already hold linear light |
| `--exposure` | `1.0` | Linear scale on the output |
| `--yaw` | `0` | Degrees; rotates the panorama horizontally |
| `--no-rle` | off | Write flat scanlines instead of adaptive RLE |
| `-j`, `--threads` | all cores | |

## Face convention

OpenGL-style, right-handed, Y-up:

```
--px  +X  right
--nx  -X  left
--py  +Y  up
--ny  -Y  down
--pz  +Z  back
--nz  -Z  front   <- horizontal centre of the panorama
```

Face orientation mismatches are the most common source of bad output. If the
panorama is rotated but otherwise correct, use `--yaw` rather than renaming
files. If individual faces look flipped, the source tool uses a different
handedness and the faces need transposing before conversion.

## Quality notes

- **Inverse mapping.** Every output pixel is traced back to a face, so there are
  no gaps or seams. Forward-projecting faces would leave both.
- **Linear light.** `.hdr` is conventionally linear, so sRGB inputs are
  linearised on load via an exact 256-entry LUT (8-bit sources) or the full
  transfer function (16-bit/float sources).
- **Precision.** All sampling and accumulation happen in `f32`; quantisation to
  RGBE occurs once, at write time. RGBE carries roughly 1% relative precision
  per channel with a shared exponent — that is the format's floor, not the
  pipeline's.
- **Dynamic range.** `.hdr` is a float container. Feeding it 8-bit sRGB faces
  yields a valid linear `.hdr`, but it cannot recover range that was never
  captured. For genuine HDR output the source faces need to carry it.
- **Edges.** Face sampling clamps at the border rather than reaching into the
  neighbouring face. With correctly rendered faces the error is sub-pixel. If
  you ever see edge artefacts at very low face resolutions, seam-aware sampling
  in `CubeMap::sample` is the place to add it.
