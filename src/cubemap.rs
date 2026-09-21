//! Cube face storage and texture sampling.

use std::path::{Path, PathBuf};

use clap::ValueEnum;
use image::DynamicImage;
use rayon::prelude::*;

use crate::color::Transfer;

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Filter {
    /// Fast, slightly soft under magnification.
    Bilinear,
    /// Catmull-Rom. Sharper, ~4x the taps. Recommended when the output is
    /// larger than 4x the face size.
    Bicubic,
}

pub struct CubeMap {
    /// Six faces of `size * size * 3` linear-light floats, row-major, RGB interleaved.
    faces: Vec<Vec<f32>>,
    size: usize,
    size_f: f32,
}

impl CubeMap {
    pub fn size(&self) -> usize {
        self.size
    }

    /// Load all six faces in parallel. `paths` must be ordered
    /// `[+X, -X, +Y, -Y, +Z, -Z]`.
    pub fn load(paths: &[PathBuf; 6], transfer: Transfer) -> Result<Self, String> {
        let loaded: Result<Vec<(Vec<f32>, usize)>, String> = paths
            .as_slice()
            .par_iter()
            .map(|p| load_face(p, transfer))
            .collect();
        let loaded = loaded?;

        let size = loaded[0].1;
        for (i, (_, s)) in loaded.iter().enumerate() {
            if *s != size {
                return Err(format!(
                    "face size mismatch: {} is {}x{}, but {} is {}x{}",
                    paths[i].display(),
                    s,
                    s,
                    paths[0].display(),
                    size,
                    size
                ));
            }
        }

        Ok(CubeMap {
            faces: loaded.into_iter().map(|(d, _)| d).collect(),
            size,
            size_f: size as f32,
        })
    }

    #[inline(always)]
    fn texel(&self, face: usize, x: i32, y: i32) -> [f32; 3] {
        let last = self.size as i32 - 1;
        let x = x.clamp(0, last) as usize;
        let y = y.clamp(0, last) as usize;
        let i = (y * self.size + x) * 3;
        let f = &self.faces[face];
        [f[i], f[i + 1], f[i + 2]]
    }

    /// Sample a face at normalised `[0,1]` coordinates.
    ///
    /// Coordinates are clamped at the face border. Cube edges therefore reuse
    /// the outermost texel rather than reaching into the neighbouring face;
    /// with correctly rendered faces this is sub-pixel and invisible.
    #[inline]
    pub fn sample(&self, face: usize, u: f32, v: f32, filter: Filter) -> [f32; 3] {
        // Texel-centre space: texel i covers [i, i+1), its centre sits at i+0.5.
        let x = u * self.size_f - 0.5;
        let y = v * self.size_f - 0.5;

        match filter {
            Filter::Bilinear => self.bilinear(face, x, y),
            Filter::Bicubic => self.bicubic(face, x, y),
        }
    }

    #[inline]
    fn bilinear(&self, face: usize, x: f32, y: f32) -> [f32; 3] {
        let x0 = x.floor();
        let y0 = y.floor();
        let fx = x - x0;
        let fy = y - y0;
        let (x0, y0) = (x0 as i32, y0 as i32);

        let c00 = self.texel(face, x0, y0);
        let c10 = self.texel(face, x0 + 1, y0);
        let c01 = self.texel(face, x0, y0 + 1);
        let c11 = self.texel(face, x0 + 1, y0 + 1);

        let mut out = [0.0f32; 3];
        for c in 0..3 {
            let top = c00[c] + (c10[c] - c00[c]) * fx;
            let bot = c01[c] + (c11[c] - c01[c]) * fx;
            out[c] = top + (bot - top) * fy;
        }
        out
    }

    #[inline]
    fn bicubic(&self, face: usize, x: f32, y: f32) -> [f32; 3] {
        let x0 = x.floor();
        let y0 = y.floor();
        let wx = catmull_rom_weights(x - x0);
        let wy = catmull_rom_weights(y - y0);
        let (x0, y0) = (x0 as i32, y0 as i32);

        let mut out = [0.0f32; 3];
        for j in 0..4 {
            let mut row = [0.0f32; 3];
            for i in 0..4 {
                let t = self.texel(face, x0 - 1 + i as i32, y0 - 1 + j as i32);
                row[0] += t[0] * wx[i];
                row[1] += t[1] * wx[i];
                row[2] += t[2] * wx[i];
            }
            out[0] += row[0] * wy[j];
            out[1] += row[1] * wy[j];
            out[2] += row[2] * wy[j];
        }
        // Catmull-Rom overshoots on high-contrast edges; negative light is not
        // meaningful, so clamp the lower bound only (highlights stay unbounded).
        [out[0].max(0.0), out[1].max(0.0), out[2].max(0.0)]
    }
}

#[inline]
fn catmull_rom_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        0.5 * (-t3 + 2.0 * t2 - t),
        0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
        0.5 * (-3.0 * t3 + 4.0 * t2 + t),
        0.5 * (t3 - t2),
    ]
}

fn load_face(path: &Path, transfer: Transfer) -> Result<(Vec<f32>, usize), String> {
    let img = image::open(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let (w, h) = (img.width() as usize, img.height() as usize);

    if w != h {
        return Err(format!(
            "{}: faces must be square, got {}x{}",
            path.display(),
            w,
            h
        ));
    }
    if w == 0 {
        return Err(format!("{}: empty image", path.display()));
    }

    let data: Vec<f32> = match &img {
        // 8-bit sources: exact LUT conversion, no transcendentals in the hot loop.
        DynamicImage::ImageLuma8(_)
        | DynamicImage::ImageLumaA8(_)
        | DynamicImage::ImageRgb8(_)
        | DynamicImage::ImageRgba8(_) => {
            let lut = transfer.lut();
            img.to_rgb8()
                .as_raw()
                .iter()
                .map(|&b| lut[b as usize])
                .collect()
        }
        // 16-bit / float sources keep their full precision.
        _ => img
            .to_rgb32f()
            .as_raw()
            .iter()
            .map(|&v| transfer.apply(v))
            .collect(),
    };

    Ok((data, w))
}
