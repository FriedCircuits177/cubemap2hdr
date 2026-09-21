//! Transfer-function handling.
//!
//! Radiance `.hdr` is conventionally a *linear light* container, while JPEG and
//! most PNGs are sRGB-encoded. We therefore linearise on load by default. Use
//! `--input-transfer linear` if your faces already hold linear data.

use clap::ValueEnum;

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Transfer {
    /// Inputs are sRGB-encoded (the default for JPG/PNG photographs).
    Srgb,
    /// Inputs already hold linear light; values are passed straight through.
    Linear,
}

/// Exact sRGB electro-optical transfer function (IEC 61966-2-1).
#[inline]
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_448_237 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

impl Transfer {
    /// Apply the transfer to a single normalised [0,1] sample.
    #[inline]
    pub fn apply(self, c: f32) -> f32 {
        match self {
            Transfer::Srgb => srgb_to_linear(c),
            Transfer::Linear => c,
        }
    }

    /// 256-entry lookup table for the 8-bit fast path.
    ///
    /// This is exact for 8-bit sources and removes ~150M `powf` calls on a
    /// 6 x 4096² cubemap, which is the difference between seconds and
    /// milliseconds during load.
    pub fn lut(self) -> [f32; 256] {
        let mut lut = [0.0f32; 256];
        for (i, slot) in lut.iter_mut().enumerate() {
            *slot = self.apply(i as f32 / 255.0);
        }
        lut
    }
}
