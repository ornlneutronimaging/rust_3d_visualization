//! Rehydrated hyperspectral CT files (NeCTAR's rehydration step): a 4-D
//! data set — one attenuation volume per TOF bin — kept in its compact form.
//!
//! The file holds the reconstructed volume `V_k` of every subspace component
//! of the dehydration (`component_volumes`, (components, z, y, x) float32)
//! and the basis spectra of the components (`subspace_basis`, (components,
//! TOF bins)); the attenuation volume of TOF bin `b` is
//!
//! ```text
//! μ(x, y, z, b) = Σ_k V_k(x, y, z) · basis[k, b]
//! ```
//!
//! so any TOF volume — or the mean over a range of bins — is one weighted sum
//! of a handful of volumes, computed here on the fly when the TOF slider
//! moves. The optional full array `hyper_volume` (TOF, z, y, x) the file may
//! also carry is not needed and not read.

use crate::volume::Volume;
use anyhow::{Context, Result, bail};
use detector_orientation::Selection;
use hdf5_metno::types::VarLenUnicode;
use ndarray::Array3;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

pub const EXTENSIONS: &[&str] = &["h5", "hdf5"];

/// Whether `path` is a rehydrated file by its extension.
pub fn is_hyper_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| EXTENSIONS.contains(&e.to_lowercase().as_str()))
}

/// A rehydrated data set in memory: the component volumes and the basis.
pub struct HyperSet {
    pub path: PathBuf,
    pub dataset_name: String,
    /// (z, y, x) of one volume.
    pub shape: [usize; 3],
    /// The component volumes, one after the other (`n_components` ×
    /// `voxels()` values).
    components: Vec<f32>,
    pub n_components: usize,
    /// The spectra: one row of `bins` values per component.
    basis: Vec<Vec<f32>>,
    pub bins: usize,
    /// The TOF axis in seconds, when the file has one.
    pub tof_s: Option<Vec<f64>>,
}

impl HyperSet {
    /// Read `path`; `progress(components_done, components)` is called as
    /// the volumes come in.
    pub fn load(path: &Path, progress: impl Fn(usize, usize)) -> Result<Self> {
        let file = hdf5_metno::File::open(path)
            .with_context(|| format!("open {}", path.display()))?;
        let basis_ds = file
            .dataset("subspace_basis")
            .with_context(|| format!("{} has no subspace_basis: not a rehydrated file", path.display()))?;
        let [n_components, bins] = basis_ds.shape()[..] else {
            bail!("subspace_basis of {} is not 2-D", path.display());
        };
        let flat: Vec<f32> = basis_ds.read_raw().context("read subspace_basis")?;
        if bins == 0 || n_components == 0 || flat.len() != n_components * bins {
            bail!("subspace_basis of {} is empty", path.display());
        }
        let basis: Vec<Vec<f32>> = flat.chunks(bins).map(<[f32]>::to_vec).collect();
        let volumes = file
            .dataset("component_volumes")
            .with_context(|| format!("{} has no component_volumes", path.display()))?;
        let [n, nz, ny, nx] = volumes.shape()[..] else {
            bail!("component_volumes of {} is not 4-D", path.display());
        };
        if n != n_components {
            bail!(
                "{} component volumes for {} basis spectra in {}",
                n,
                n_components,
                path.display()
            );
        }
        let voxels = nz * ny * nx;
        let mut components: Vec<f32> = Vec::new();
        components.try_reserve_exact(n * voxels).map_err(|_| {
            anyhow::anyhow!(
                "the {n} component volumes of {nx}x{ny}x{nz} f32 = {:.1} GB do not fit in memory",
                (n * voxels * 4) as f64 / 1e9
            )
        })?;
        progress(0, n);
        for k in 0..n {
            let part = volumes
                .read_slice::<f32, _, ndarray16::Ix3>((k, .., .., ..))
                .with_context(|| format!("read component_volumes[{k}]"))?;
            let (values, _) = part.into_raw_vec_and_offset();
            if values.len() != voxels {
                bail!("component volume {k} has {} voxels, not {voxels}", values.len());
            }
            components.extend_from_slice(&values);
            progress(k + 1, n);
        }
        let tof_s = file
            .dataset("tof_s")
            .and_then(|ds| ds.read_raw::<f64>())
            .ok()
            .filter(|tof| tof.len() == bins);
        let dataset_name = file
            .dataset("dataset_name")
            .and_then(|ds| ds.read_scalar::<VarLenUnicode>())
            .map(|s| s.as_str().to_owned())
            .unwrap_or_default();
        Ok(Self {
            path: path.to_path_buf(),
            dataset_name,
            shape: [nz, ny, nx],
            components,
            n_components: n,
            basis,
            bins,
            tof_s,
        })
    }

    pub fn voxels(&self) -> usize {
        self.shape[0] * self.shape[1] * self.shape[2]
    }

    /// The bins `first..first+width` clamped to the axis.
    pub fn bin_range(&self, first: usize, width: usize) -> (usize, usize) {
        let first = first.min(self.bins.saturating_sub(1));
        let last = (first + width.max(1)).min(self.bins);
        (first, last)
    }

    /// The weight of every component for the mean over the bins
    /// `first..first+width`.
    pub fn weights(&self, first: usize, width: usize) -> Vec<f32> {
        let (first, last) = self.bin_range(first, width);
        let n = (last - first) as f32;
        self.basis
            .iter()
            .map(|row| row[first..last].iter().sum::<f32>() / n)
            .collect()
    }

    /// The attenuation volume averaged over the bins `first..first+width`:
    /// `Σ_k V_k · mean(basis[k, bins])`.
    pub fn volume(&self, first: usize, width: usize) -> Volume {
        let weights = self.weights(first, width);
        let voxels = self.voxels();
        let mut out = vec![0.0f32; voxels];
        const BLOCK: usize = 1 << 16;
        out.par_chunks_mut(BLOCK).enumerate().for_each(|(i, block)| {
            let start = i * BLOCK;
            for (k, w) in weights.iter().enumerate() {
                let src = &self.components[k * voxels + start..k * voxels + start + block.len()];
                for (o, v) in block.iter_mut().zip(src) {
                    *o += w * v;
                }
            }
        });
        let [nz, ny, nx] = self.shape;
        let data = Array3::from_shape_vec((nz, ny, nx), out).expect("sized above");
        Volume::new(data, self.path.clone(), self.bins, Selection::default())
    }

    /// The spectrum of one voxel over the TOF bins: `Σ_k V_k(voxel) ·
    /// basis[k, :]`.
    pub fn spectrum_at(&self, z: usize, y: usize, x: usize) -> Vec<f32> {
        let [_, ny, nx] = self.shape;
        let i = (z * ny + y) * nx + x;
        let voxels = self.voxels();
        let mut spectrum = vec![0.0f32; self.bins];
        for (k, row) in self.basis.iter().enumerate() {
            let v = self.components[k * voxels + i];
            for (s, b) in spectrum.iter_mut().zip(row) {
                *s += v * b;
            }
        }
        spectrum
    }

    /// The TOF of a bin, in milliseconds, when the file has the axis.
    pub fn tof_ms(&self, bin: usize) -> Option<f64> {
        self.tof_s.as_ref().and_then(|t| t.get(bin)).map(|s| s * 1e3)
    }

    /// "bin 12 of 1000 (3.456 ms)" — or "bin 12 to 15 of 1000 (…)" for a
    /// range.
    pub fn range_label(&self, first: usize, width: usize) -> String {
        let (first, last) = self.bin_range(first, width);
        let tof = match (self.tof_ms(first), self.tof_ms(last - 1)) {
            (Some(a), Some(b)) if last - first > 1 => format!(" ({a:.4} to {b:.4} ms)"),
            (Some(a), _) => format!(" ({a:.4} ms)"),
            _ => String::new(),
        };
        if last - first > 1 {
            format!("bins {} to {} of {}{tof}", first + 1, last, self.bins)
        } else {
            format!("bin {} of {}{tof}", first + 1, self.bins)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_set(path: &Path) {
        let file = hdf5_metno::File::create(path).unwrap();
        // 2 components of 2 x 1 x 3 voxels; 3 TOF bins.
        let volumes: Vec<f32> = (0..12).map(|i| i as f32).collect();
        file.new_dataset::<f32>()
            .shape((2, 2, 1, 3))
            .create("component_volumes")
            .unwrap()
            .write_raw(&volumes)
            .unwrap();
        file.new_dataset::<f32>()
            .shape((2, 3))
            .create("subspace_basis")
            .unwrap()
            .write_raw(&[1.0f32, 2.0, 0.0, 0.0, 1.0, 4.0])
            .unwrap();
        file.new_dataset::<f64>()
            .shape(3)
            .create("tof_s")
            .unwrap()
            .write_raw(&[1e-3, 2e-3, 3e-3])
            .unwrap();
        let name: VarLenUnicode = "demo".parse().unwrap();
        file.new_dataset::<VarLenUnicode>()
            .create("dataset_name")
            .unwrap()
            .write_scalar(&name)
            .unwrap();
    }

    #[test]
    fn a_rehydrated_file_gives_the_volume_of_any_tof_bin() {
        let dir = std::env::temp_dir().join(format!("volume_3d_viewer_hyper_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rehydrated.h5");
        write_set(&path);
        assert!(is_hyper_file(&path));

        let seen = std::sync::Mutex::new(Vec::new());
        let set = HyperSet::load(&path, |done, total| seen.lock().unwrap().push((done, total))).unwrap();
        assert_eq!(seen.lock().unwrap().last(), Some(&(2, 2)));
        assert_eq!(set.shape, [2, 1, 3]);
        assert_eq!(set.n_components, 2);
        assert_eq!(set.bins, 3);
        assert_eq!(set.dataset_name, "demo");
        assert_eq!(set.tof_ms(1), Some(2.0));

        // Bin 0: 1.0 * V1 + 0.0 * V2 = V1.
        let v = set.volume(0, 1);
        assert_eq!(v.data.dim(), (2, 1, 3));
        assert_eq!(v.data[[0, 0, 0]], 0.0);
        assert_eq!(v.data[[1, 0, 2]], 5.0);
        // Bin 2: 0 * V1 + 4 * V2.
        let v = set.volume(2, 1);
        assert_eq!(v.data[[0, 0, 0]], 24.0);
        assert_eq!(v.data[[1, 0, 2]], 44.0);
        // Bins 1..3 averaged: weights (1, 2.5).
        assert_eq!(set.weights(1, 2), vec![1.0, 2.5]);
        let v = set.volume(1, 2);
        assert_eq!(v.data[[0, 0, 1]], 1.0 + 2.5 * 7.0);
        // Past the end: clamped.
        assert_eq!(set.bin_range(10, 5), (2, 3));
        assert_eq!(set.range_label(0, 1), "bin 1 of 3 (1.0000 ms)");
        assert_eq!(set.range_label(1, 2), "bins 2 to 3 of 3 (2.0000 to 3.0000 ms)");
        // The spectrum of voxel (1, 0, 2): V1 = 5, V2 = 11.
        assert_eq!(set.spectrum_at(1, 0, 2), vec![5.0, 10.0 + 11.0, 44.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
