//! The in-memory reconstructed volume and the operations the viewer needs
//! from it: intensity statistics for auto-contrast and mean-pooled
//! downsampling to a size that fits in a GPU 3-D texture.

use ndarray::Array3;
use rayon::prelude::*;
use std::path::PathBuf;

/// Full-resolution volume, `(nz, ny, nx)` with `z` the slice index (one
/// reconstructed TIFF per z), `y` the image row and `x` the image column.
pub struct Volume {
    pub data: Array3<f32>,
    /// Smallest / largest finite value in the volume (sampled).
    pub vmin: f32,
    pub vmax: f32,
    /// Robust display range (0.5 % / 99.5 % percentiles, sampled).
    pub p_low: f32,
    pub p_high: f32,
    pub folder: PathBuf,
    pub n_files: usize,
}

/// Mean-pooled volume ready to be uploaded as a GL 3-D texture:
/// x fastest, then y, then z — the layout `glTexImage3D` expects.
pub struct TextureData {
    pub data: Vec<f32>,
    /// `[nx, ny, nz]` of the downsampled volume.
    pub dims: [usize; 3],
}

impl Volume {
    pub fn new(data: Array3<f32>, folder: PathBuf, n_files: usize) -> Self {
        let (vmin, vmax, p_low, p_high) = compute_stats(&data);
        Self {
            data,
            vmin,
            vmax,
            p_low,
            p_high,
            folder,
            n_files,
        }
    }

    pub fn nz(&self) -> usize {
        self.data.dim().0
    }
    pub fn ny(&self) -> usize {
        self.data.dim().1
    }
    pub fn nx(&self) -> usize {
        self.data.dim().2
    }

    /// Downsample by integer-step mean pooling so no axis exceeds `max_dim`.
    /// Non-finite voxels (NaN/Inf from the reconstruction) are excluded from
    /// the mean; an all-non-finite block becomes 0.
    pub fn downsample_for_texture(&self, max_dim: usize) -> TextureData {
        let (nz, ny, nx) = self.data.dim();
        let step = nx.max(ny).max(nz).div_ceil(max_dim.max(1)).max(1);
        let (dx, dy, dz) = (nx.div_ceil(step), ny.div_ceil(step), nz.div_ceil(step));
        let src = self.data.as_slice().expect("volume is standard layout");

        let mut out = vec![0.0f32; dx * dy * dz];
        out.par_chunks_mut(dx * dy)
            .enumerate()
            .for_each(|(oz, slab)| {
                let z0 = oz * step;
                let z1 = (z0 + step).min(nz);
                for oy in 0..dy {
                    let y0 = oy * step;
                    let y1 = (y0 + step).min(ny);
                    for ox in 0..dx {
                        let x0 = ox * step;
                        let x1 = (x0 + step).min(nx);
                        let mut sum = 0.0f64;
                        let mut cnt = 0u32;
                        for z in z0..z1 {
                            for y in y0..y1 {
                                let base = (z * ny + y) * nx;
                                for v in &src[base + x0..base + x1] {
                                    if v.is_finite() {
                                        sum += *v as f64;
                                        cnt += 1;
                                    }
                                }
                            }
                        }
                        slab[oy * dx + ox] = if cnt > 0 { (sum / cnt as f64) as f32 } else { 0.0 };
                    }
                }
            });

        TextureData {
            data: out,
            dims: [dx, dy, dz],
        }
    }
}

/// Sampled min/max and robust (0.5 %, 99.5 %) percentiles, ignoring
/// non-finite values. Falls back to (0, 1) if nothing finite is found.
fn compute_stats(data: &Array3<f32>) -> (f32, f32, f32, f32) {
    let flat = data.as_slice().expect("volume is standard layout");
    let stride = (flat.len() / 16_000_000).max(1);

    let mut vmin = f32::INFINITY;
    let mut vmax = f32::NEG_INFINITY;
    for v in flat.iter().step_by(stride) {
        if v.is_finite() {
            vmin = vmin.min(*v);
            vmax = vmax.max(*v);
        }
    }
    if !vmin.is_finite() || !vmax.is_finite() {
        return (0.0, 1.0, 0.0, 1.0);
    }
    if vmin == vmax {
        return (vmin, vmax, vmin, vmax);
    }

    const BINS: usize = 4096;
    let mut counts = [0u64; BINS];
    let scale = (BINS - 1) as f32 / (vmax - vmin);
    let mut total = 0u64;
    for v in flat.iter().step_by(stride) {
        if v.is_finite() {
            counts[((v - vmin) * scale) as usize] += 1;
            total += 1;
        }
    }

    let percentile = |frac: f64| -> f32 {
        let target = (total as f64 * frac) as u64;
        let mut acc = 0u64;
        for (i, c) in counts.iter().enumerate() {
            acc += c;
            if acc >= target {
                return vmin + i as f32 / scale;
            }
        }
        vmax
    };

    let p_low = percentile(0.005);
    let mut p_high = percentile(0.995);
    if p_high <= p_low {
        p_high = vmax;
    }
    (vmin, vmax, p_low, p_high)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downsample_dims_and_mean() {
        // 4x4x4 volume of constant 2.0, downsampled to <= 2 per axis.
        let data = Array3::<f32>::from_elem((4, 4, 4), 2.0);
        let vol = Volume::new(data, PathBuf::new(), 4);
        let td = vol.downsample_for_texture(2);
        assert_eq!(td.dims, [2, 2, 2]);
        assert!(td.data.iter().all(|v| (*v - 2.0).abs() < 1e-6));
    }

    #[test]
    fn downsample_ignores_nan() {
        let mut data = Array3::<f32>::from_elem((2, 2, 2), 3.0);
        data[[0, 0, 0]] = f32::NAN;
        let vol = Volume::new(data, PathBuf::new(), 2);
        let td = vol.downsample_for_texture(1);
        assert_eq!(td.dims, [1, 1, 1]);
        assert!((td.data[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn stats_span_the_data() {
        let data = Array3::<f32>::from_shape_fn((8, 8, 8), |(z, y, x)| (z + y + x) as f32);
        let vol = Volume::new(data, PathBuf::new(), 8);
        assert_eq!(vol.vmin, 0.0);
        assert_eq!(vol.vmax, 21.0);
        assert!(vol.p_low <= vol.p_high);
    }
}
