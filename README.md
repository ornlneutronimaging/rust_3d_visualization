# 3-D Volume Viewer

Interactive viewer for reconstructed CT volumes: point it at a folder of TIFF
slices (e.g. the output folder written by `rust_ct_reconstruction`,
`image_0000.tiff`, `image_0001.tiff`, …) or at a single multi-page TIFF file
(one Z-slice per page) and explore the volume as orthogonal slices or as a
GPU-rendered 3-D volume.

## Usage

```bash
# With the reconstruction output folder as argument…
./launch_3d_visualization.sh /SNS/VENUS/IPTS-XXXXX/shared/.../reconstructed

# …or a single multi-page TIFF file…
./launch_3d_visualization.sh /SNS/VENUS/IPTS-XXXXX/shared/.../stack.tiff

# …or without: browse — or drag & drop a folder / TIFF file onto the window.
./launch_3d_visualization.sh
```

The launch script rebuilds the binary automatically when the sources changed
(requires `cargo` on first build). A graphical session (e.g. ThinLinc) is
needed.

## What it does

- Loads every `.tif`/`.tiff` directly inside a folder, in sorted filename
  order along Z (files are decoded in parallel; any grayscale bit depth is
  accepted and converted to f32) — or all pages of a single multi-page TIFF
  file, in page order. NaN/Inf voxels from the reconstruction are handled
  gracefully. Inputs can also be drag & dropped onto the window.
- **3-D volume tab** — orthographic raycasting of the volume on the GPU:
  - modes: maximum-intensity projection (MIP), alpha compositing, X-ray (mean)
  - drag to rotate, right-drag / shift-drag to pan, scroll to zoom,
    double-click to reset the view
  - clip box (min/max per axis) to cut into the volume
  - the volume is mean-pooled down to a configurable 3-D texture size
    (default ≤ 256³) so arbitrarily large volumes render smoothly
- **Slices tab** — full-resolution axial (Z), coronal (Y) and sagittal (X)
  slices with a slider each and a cursor value readout.
- Shared display settings: colormap (Gray, Viridis, Inferno, …, Jet) and
  min/max display range with auto-contrast (0.5–99.5 percentiles).

## Memory note

The full-resolution volume is held in RAM as f32 (e.g. a 2048×2048×1500
reconstruction is ~25 GB); peak usage during loading is about one volume plus
a small chunk of decoded slices. Run it on an analysis node for large volumes.

## Development

```bash
cargo test           # unit tests (loader, downsampling, statistics)
cargo build --release
```
