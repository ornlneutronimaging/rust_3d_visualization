//! Loading reconstructed TIFF slices into a 3-D volume, either from a folder
//! of single-slice files or from one multi-page TIFF stack.
//!
//! The expected folder input is the output of `rust_ct_reconstruction`
//! (`image_0000.tiff`, `image_0001.tiff`, … — 32-bit float grayscale), but any
//! folder of equally-sized TIFF images works; files are stacked in sorted
//! filename order along z. Multi-page TIFFs contribute one z-slice per page,
//! so a single file holding the whole stack also works (see [`load_file`]).
//!
//! Folder files are decoded in parallel (they usually sit on NFS/GPFS where
//! per-file latency dominates), in chunks so peak memory stays close to one
//! copy of the volume.

use crate::volume::Volume;
use anyhow::{Context, Result, anyhow, bail};
use ndarray::{Array2, Array3};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

pub const SUPPORTED_EXTENSIONS: &[&str] = &["tif", "tiff"];

/// How many files are decoded in parallel before being appended to the
/// volume buffer; bounds peak memory at ~one volume + one chunk of frames.
const CHUNK_FILES: usize = 64;

fn ext_of(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase()
}

/// Every TIFF file directly inside `dir`, sorted by filename.
pub fn list_tiffs_in_dir(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("read dir {}", dir.display()))? {
        let path = entry?.path();
        if path.is_file() && SUPPORTED_EXTENSIONS.contains(&ext_of(&path).as_str()) {
            out.push(path);
        }
    }
    if out.is_empty() {
        bail!("No TIFF files found in {}", dir.display());
    }
    out.sort();
    Ok(out)
}

/// Load `path` into a `(nz, ny, nx)` volume: a directory is treated as a
/// folder of TIFF slices, anything else as a single multi-page TIFF file.
pub fn load_path(
    path: &Path,
    progress: impl Fn(usize, usize) + Send + Sync,
) -> Result<Volume> {
    if path.is_dir() {
        load_folder(path, progress)
    } else {
        load_file(path, progress)
    }
}

/// Load every TIFF in `dir` into a `(nz, ny, nx)` volume.
/// `progress(files_done, files_total)` is called from worker threads.
pub fn load_folder(
    dir: &Path,
    progress: impl Fn(usize, usize) + Send + Sync,
) -> Result<Volume> {
    let paths = list_tiffs_in_dir(dir)?;
    let total = paths.len();
    let counter = AtomicUsize::new(0);

    let mut buf: Vec<f32> = Vec::new();
    let mut dims: Option<(usize, usize)> = None; // (height, width)
    let mut n_frames = 0usize;

    for chunk in paths.chunks(CHUNK_FILES) {
        let decoded: Vec<(usize, Result<Vec<Array2<f32>>>)> = chunk
            .par_iter()
            .enumerate()
            .map(|(i, path)| {
                let r = load_tiff(path);
                let done = counter.fetch_add(1, Ordering::Relaxed) + 1;
                progress(done, total);
                (i, r)
            })
            .collect();

        // `par_iter().collect()` preserves order, but keep the index around
        // so a change of iterator can never silently shuffle slices.
        let mut decoded = decoded;
        decoded.sort_by_key(|(i, _)| *i);

        for (i, result) in decoded {
            let frames =
                result.with_context(|| format!("loading {}", chunk[i].display()))?;
            for frame in frames {
                let (h, w) = (frame.shape()[0], frame.shape()[1]);
                match dims {
                    None => {
                        dims = Some((h, w));
                        // All slices should be the same size, so reserve the
                        // whole volume up front — and fail with a clear
                        // message rather than aborting if it does not fit.
                        let estimate = total * h * w;
                        buf.try_reserve_exact(estimate).map_err(|_| {
                            anyhow!(
                                "volume does not fit in memory: {} slices of {}x{} f32 = {:.1} GB",
                                total,
                                w,
                                h,
                                (estimate * 4) as f64 / 1e9
                            )
                        })?;
                    }
                    Some((dh, dw)) if (dh, dw) != (h, w) => {
                        bail!(
                            "Slice size mismatch: {}x{} in {} does not match {}x{}",
                            w,
                            h,
                            chunk[i].display(),
                            dw,
                            dh
                        );
                    }
                    _ => {}
                }
                buf.extend_from_slice(frame.as_slice().expect("frame is standard layout"));
                n_frames += 1;
            }
        }
    }

    let (h, w) = dims.ok_or_else(|| anyhow!("No frames were loaded"))?;
    let data = Array3::from_shape_vec((n_frames, h, w), buf)?;
    Ok(Volume::new(data, dir.to_path_buf(), total))
}

/// Load a single (possibly multi-page) TIFF file into a `(nz, ny, nx)`
/// volume, one z-slice per page in file order.
/// `progress(pages_done, pages_total)` is called as pages are decoded.
pub fn load_file(path: &Path, progress: impl Fn(usize, usize) + Send + Sync) -> Result<Volume> {
    if !SUPPORTED_EXTENSIONS.contains(&ext_of(path).as_str()) {
        bail!("{} is not a TIFF file", path.display());
    }

    // A cheap IFD walk first, so the progress bar and the up-front
    // allocation both know the page count.
    let total = count_pages(path)?;

    let mut decoder = open_decoder(path)?;
    let mut buf: Vec<f32> = Vec::new();
    let mut dims: Option<(usize, usize)> = None; // (height, width)
    let mut n_frames = 0usize;

    loop {
        let frame = read_page(&mut decoder)
            .with_context(|| format!("loading page {} of {}", n_frames + 1, path.display()))?;
        let (h, w) = (frame.shape()[0], frame.shape()[1]);
        match dims {
            None => {
                dims = Some((h, w));
                let estimate = total * h * w;
                buf.try_reserve_exact(estimate).map_err(|_| {
                    anyhow!(
                        "volume does not fit in memory: {} pages of {}x{} f32 = {:.1} GB",
                        total,
                        w,
                        h,
                        (estimate * 4) as f64 / 1e9
                    )
                })?;
            }
            Some((dh, dw)) if (dh, dw) != (h, w) => {
                bail!(
                    "Page size mismatch: {}x{} on page {} of {} does not match {}x{}",
                    w,
                    h,
                    n_frames + 1,
                    path.display(),
                    dw,
                    dh
                );
            }
            _ => {}
        }
        buf.extend_from_slice(frame.as_slice().expect("frame is standard layout"));
        n_frames += 1;
        progress(n_frames, total);

        if !decoder.more_images() {
            break;
        }
        decoder.next_image()?;
    }

    let (h, w) = dims.ok_or_else(|| anyhow!("No frames were loaded"))?;
    let data = Array3::from_shape_vec((n_frames, h, w), buf)?;
    Ok(Volume::new(data, path.to_path_buf(), 1))
}

fn open_decoder(path: &Path) -> Result<tiff::decoder::Decoder<std::io::BufReader<std::fs::File>>> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    tiff::decoder::Decoder::new(std::io::BufReader::new(file))
        .with_context(|| format!("decode TIFF {}", path.display()))
}

/// Number of pages (IFDs) in a TIFF file, without decoding any pixel data.
fn count_pages(path: &Path) -> Result<usize> {
    let mut decoder = open_decoder(path)?;
    let mut n = 1usize;
    while decoder.more_images() {
        decoder.next_image()?;
        n += 1;
    }
    Ok(n)
}

/// Read every page of a (possibly multi-page) TIFF file.
fn load_tiff(path: &Path) -> Result<Vec<Array2<f32>>> {
    let mut decoder = open_decoder(path)?;
    let mut out = Vec::new();
    loop {
        out.push(read_page(&mut decoder)?);
        if !decoder.more_images() {
            break;
        }
        decoder.next_image()?;
    }
    Ok(out)
}

/// Decode the current page of `decoder` into an `(h, w)` f32 frame.
fn read_page<R: std::io::Read + std::io::Seek>(
    decoder: &mut tiff::decoder::Decoder<R>,
) -> Result<Array2<f32>> {
    use tiff::decoder::DecodingResult;

    let (w, h) = decoder.dimensions()?;
    let (w, h) = (w as usize, h as usize);

    let data = decoder.read_image()?;
    let values: Vec<f32> = match data {
        DecodingResult::U8(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::U16(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::U32(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::U64(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::I8(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::I16(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::I32(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::I64(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::F16(v) => v.into_iter().map(|x| x.to_f32()).collect(),
        DecodingResult::F32(v) => v,
        DecodingResult::F64(v) => v.into_iter().map(|x| x as f32).collect(),
    };

    to_frame(values, w, h)
}

/// Turn a flat, row-major buffer into an `(h, w)` array. If the buffer carries
/// several samples per pixel (e.g. RGB TIFF) only the first sample is kept.
fn to_frame(values: Vec<f32>, w: usize, h: usize) -> Result<Array2<f32>> {
    let expected = w * h;
    if values.len() == expected {
        return Ok(Array2::from_shape_vec((h, w), values)?);
    }
    if expected > 0 && values.len() % expected == 0 {
        let spp = values.len() / expected;
        let first: Vec<f32> = (0..expected).map(|i| values[i * spp]).collect();
        return Ok(Array2::from_shape_vec((h, w), first)?);
    }
    bail!(
        "Pixel count {} is not compatible with {}x{}",
        values.len(),
        w,
        h
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_f32_tiff(path: &Path, w: usize, h: usize, data: &[f32]) {
        let file = std::fs::File::create(path).unwrap();
        let mut enc = tiff::encoder::TiffEncoder::new(std::io::BufWriter::new(file)).unwrap();
        enc.write_image::<tiff::encoder::colortype::Gray32Float>(w as u32, h as u32, data)
            .unwrap();
    }

    #[test]
    fn folder_of_f32_tiffs_stacks_in_name_order() {
        let dir = std::env::temp_dir().join("volume_3d_viewer_loader_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        for z in 0..3usize {
            let data: Vec<f32> = (0..6).map(|i| (z * 100 + i) as f32).collect();
            write_f32_tiff(&dir.join(format!("image_{z:04}.tiff")), 3, 2, &data);
        }

        let vol = load_folder(&dir, |_, _| {}).unwrap();
        assert_eq!(vol.data.dim(), (3, 2, 3));
        assert_eq!(vol.data[[0, 0, 0]], 0.0);
        assert_eq!(vol.data[[2, 0, 0]], 200.0);
        assert_eq!(vol.data[[1, 1, 2]], 105.0);
    }

    #[test]
    fn multipage_tiff_loads_as_volume() {
        let dir = std::env::temp_dir().join("volume_3d_viewer_multipage_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stack.tiff");

        {
            let file = std::fs::File::create(&path).unwrap();
            let mut enc =
                tiff::encoder::TiffEncoder::new(std::io::BufWriter::new(file)).unwrap();
            for z in 0..4usize {
                let data: Vec<f32> = (0..6).map(|i| (z * 100 + i) as f32).collect();
                enc.write_image::<tiff::encoder::colortype::Gray32Float>(3, 2, &data)
                    .unwrap();
            }
        }

        let last = std::sync::Mutex::new((0usize, 0usize));
        let vol = load_path(&path, |done, total| {
            *last.lock().unwrap() = (done, total);
        })
        .unwrap();
        assert_eq!(vol.data.dim(), (4, 2, 3));
        assert_eq!(vol.data[[0, 0, 0]], 0.0);
        assert_eq!(vol.data[[3, 1, 2]], 305.0);
        assert_eq!(*last.lock().unwrap(), (4, 4));
    }
}
