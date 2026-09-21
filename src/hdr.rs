//! Radiance RGBE (`.hdr`) writer.
//!
//! The format is small enough to implement directly, which keeps the dependency
//! list at three crates. Output uses new-style adaptive RLE (the `2 2 hi lo`
//! scanline marker), which every reader since Radiance 2.x understands and which
//! typically halves file size on real panoramas.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use rayon::prelude::*;

/// Encode a linear RGB triple as 4-byte RGBE (shared exponent).
#[inline]
pub fn float_to_rgbe(r: f32, g: f32, b: f32) -> [u8; 4] {
    let max = r.max(g).max(b);

    // Guard against NaN, negatives and denormal-range values.
    if !(max > 1e-32) {
        return [0, 0, 0, 0];
    }

    // exponent e such that max / 2^e lies in [0.5, 1.0)
    let e = max.log2().floor() as i32 + 1;
    if e < -126 {
        return [0, 0, 0, 0];
    }
    let e = e.min(127);

    let scale = 256.0 / exp2i(e);
    [
        (r * scale).clamp(0.0, 255.0) as u8,
        (g * scale).clamp(0.0, 255.0) as u8,
        (b * scale).clamp(0.0, 255.0) as u8,
        (e + 128) as u8,
    ]
}

#[inline]
fn exp2i(e: i32) -> f32 {
    // Exact for the full valid exponent range, and cheaper than powf.
    f32::from_bits(((e + 127) as u32) << 23)
}

/// Write a linear-light RGB float buffer as a Radiance `.hdr` file.
///
/// `data` must be `width * height * 3` floats, row-major, top row first.
pub fn write_hdr(
    path: &Path,
    width: u32,
    height: u32,
    data: &[f32],
    rle: bool,
) -> io::Result<()> {
    let w = width as usize;
    let h = height as usize;
    assert_eq!(data.len(), w * h * 3, "buffer size does not match dimensions");

    let file = File::create(path)?;
    let mut out = BufWriter::with_capacity(1 << 20, file);

    out.write_all(b"#?RADIANCE\n")?;
    out.write_all(b"# Created by cube2hdr\n")?;
    out.write_all(b"FORMAT=32-bit_rle_rgbe\n")?;
    out.write_all(b"EXPOSURE=1.0\n")?;
    out.write_all(b"\n")?;
    // -Y first means the first scanline stored is the top of the image.
    writeln!(out, "-Y {height} +X {width}")?;

    // RLE is only legal for these widths; fall back to flat scanlines otherwise.
    let use_rle = rle && (8..=0x7fff).contains(&w);

    // Encoding is pure and per-scanline, so it parallelises cleanly; only the
    // final ordered write is sequential.
    let scanlines: Vec<Vec<u8>> = data
        .par_chunks_exact(w * 3)
        .map(|row| encode_scanline(row, w, use_rle))
        .collect();

    for line in &scanlines {
        out.write_all(line)?;
    }
    out.flush()
}

fn encode_scanline(row: &[f32], width: usize, use_rle: bool) -> Vec<u8> {
    if !use_rle {
        let mut out = Vec::with_capacity(width * 4);
        for px in row.chunks_exact(3) {
            out.extend_from_slice(&float_to_rgbe(px[0], px[1], px[2]));
        }
        return out;
    }

    // Adaptive RLE is applied per component, so de-interleave into four planes.
    let mut planes = vec![0u8; width * 4];
    for (i, px) in row.chunks_exact(3).enumerate() {
        let rgbe = float_to_rgbe(px[0], px[1], px[2]);
        planes[i] = rgbe[0];
        planes[width + i] = rgbe[1];
        planes[2 * width + i] = rgbe[2];
        planes[3 * width + i] = rgbe[3];
    }

    let mut out = Vec::with_capacity(width * 2);
    out.extend_from_slice(&[2, 2, (width >> 8) as u8, (width & 0xff) as u8]);
    for c in 0..4 {
        encode_component(&mut out, &planes[c * width..(c + 1) * width]);
    }
    out
}

/// Radiance adaptive RLE for one component plane.
///
/// A count byte > 128 introduces a run of `count - 128` copies of the next byte;
/// a count byte <= 128 introduces that many literal bytes.
fn encode_component(out: &mut Vec<u8>, data: &[u8]) {
    let n = data.len();
    let mut i = 0;

    while i < n {
        // How far does the run starting at `i` extend? (capped at 127)
        let mut run = 1;
        while i + run < n && run < 127 && data[i + run] == data[i] {
            run += 1;
        }

        if run >= 4 {
            out.push(128 + run as u8);
            out.push(data[i]);
            i += run;
        } else {
            // Emit literals until a run of >= 4 begins, or we hit the 128 cap.
            let start = i;
            let mut j = i;
            while j < n && j - start < 128 {
                if j + 3 < n
                    && data[j] == data[j + 1]
                    && data[j] == data[j + 2]
                    && data[j] == data[j + 3]
                {
                    break;
                }
                j += 1;
            }
            debug_assert!(j > start, "literal block must make progress");
            out.push((j - start) as u8);
            out.extend_from_slice(&data[start..j]);
            i = j;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference RGBE decode, used to check the round trip.
    fn rgbe_to_float(p: [u8; 4]) -> [f32; 3] {
        if p[3] == 0 {
            return [0.0; 3];
        }
        let f = exp2i(p[3] as i32 - 128) / 256.0;
        [
            (p[0] as f32 + 0.5) * f,
            (p[1] as f32 + 0.5) * f,
            (p[2] as f32 + 0.5) * f,
        ]
    }

    #[test]
    fn rgbe_round_trip_is_within_one_percent() {
        for &v in &[1e-3f32, 0.018, 0.5, 1.0, 12.5, 400.0, 65504.0] {
            let enc = float_to_rgbe(v, v * 0.5, v * 0.25);
            let dec = rgbe_to_float(enc);
            let err = (dec[0] - v).abs() / v;
            assert!(err < 0.01, "v={v} dec={:?} err={err}", dec);
        }
    }

    #[test]
    fn zero_and_nan_encode_to_black() {
        assert_eq!(float_to_rgbe(0.0, 0.0, 0.0), [0, 0, 0, 0]);
        assert_eq!(float_to_rgbe(f32::NAN, 0.0, 0.0), [0, 0, 0, 0]);
        assert_eq!(float_to_rgbe(-1.0, -1.0, -1.0), [0, 0, 0, 0]);
    }

    #[test]
    fn rle_round_trips() {
        let mut data: Vec<u8> = vec![7; 300];
        data.extend((0u8..=255).cycle().take(200));
        data.extend(vec![0u8; 5]);
        data.push(99);

        let mut enc = Vec::new();
        encode_component(&mut enc, &data);

        // Decode.
        let mut dec = Vec::new();
        let mut i = 0;
        while i < enc.len() {
            let c = enc[i] as usize;
            i += 1;
            if c > 128 {
                dec.extend(std::iter::repeat(enc[i]).take(c - 128));
                i += 1;
            } else {
                dec.extend_from_slice(&enc[i..i + c]);
                i += c;
            }
        }
        assert_eq!(dec, data);
    }
}
