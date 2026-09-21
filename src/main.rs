//! cube2hdr -- six square cubemap faces -> one equirectangular Radiance `.hdr`.
//!
//! The conversion is output-driven (inverse mapping): for every output pixel we
//! compute the ray it represents, find the cube face that ray pierces, and
//! filter-sample it. That guarantees full coverage with no seams or holes, which
//! forward-projecting each face cannot.

mod color;
mod cubemap;
mod hdr;
mod projection;

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;
use rayon::prelude::*;

use color::Transfer;
use cubemap::{CubeMap, Filter};
use projection::{dir_to_face_uv, equirect_to_dir, FACE_NAMES};

const ABOUT: &str = "Convert six square cubemap faces into an equirectangular Radiance .hdr panorama";

const LONG_ABOUT: &str = "\
Convert six square cubemap faces (JPG/PNG) into a single equirectangular
panorama in Radiance .hdr format.

Face convention is OpenGL-style, right-handed and Y-up:

  --px  +X  right
  --nx  -X  left
  --py  +Y  up
  --ny  -Y  down
  --pz  +Z  back
  --nz  -Z  front  (lands at the horizontal centre of the panorama)

If your faces came from a tool with a different forward axis, rotate the result
with --yaw instead of renaming or re-rendering files.

Note: .hdr is a float container. Feeding it 8-bit sRGB faces produces a valid
linear-light .hdr, but it cannot invent dynamic range that was not captured.";

#[derive(Parser, Debug)]
#[command(name = "cube2hdr", version, about = ABOUT, long_about = LONG_ABOUT)]
struct Args {
    /// +X face (right)
    #[arg(long, value_name = "FILE")]
    px: PathBuf,
    /// -X face (left)
    #[arg(long, value_name = "FILE")]
    nx: PathBuf,
    /// +Y face (up)
    #[arg(long, value_name = "FILE")]
    py: PathBuf,
    /// -Y face (down)
    #[arg(long, value_name = "FILE")]
    ny: PathBuf,
    /// +Z face (back)
    #[arg(long, value_name = "FILE")]
    pz: PathBuf,
    /// -Z face (front, centre of the panorama)
    #[arg(long, value_name = "FILE")]
    nz: PathBuf,

    /// Output .hdr path
    #[arg(short, long, value_name = "FILE")]
    output: PathBuf,

    /// Output width in pixels; height is always width/2 [default: 4 x face size]
    #[arg(short, long, value_name = "PX")]
    width: Option<u32>,

    /// Transfer function of the input faces
    #[arg(long, value_enum, default_value_t = Transfer::Srgb)]
    input_transfer: Transfer,

    /// Reconstruction filter used when sampling the faces
    #[arg(long, value_enum, default_value_t = Filter::Bilinear)]
    filter: Filter,

    /// Supersampling grid per output pixel (N x N). 1 disables it
    #[arg(long, value_name = "N", default_value_t = 2)]
    samples: u32,

    /// Linear multiplier applied to the output (1.0 = unchanged)
    #[arg(long, value_name = "SCALE", default_value_t = 1.0)]
    exposure: f32,

    /// Rotate the panorama horizontally, in degrees
    #[arg(long, value_name = "DEG", default_value_t = 0.0, allow_negative_numbers = true)]
    yaw: f32,

    /// Write uncompressed scanlines instead of adaptive RLE
    #[arg(long)]
    no_rle: bool,

    /// Worker threads [default: number of logical cores]
    #[arg(short = 'j', long, value_name = "N")]
    threads: Option<usize>,

    /// Suppress progress output on stderr
    #[arg(short, long)]
    quiet: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Args) -> Result<(), String> {
    if let Some(n) = args.threads {
        if n == 0 {
            return Err("--threads must be at least 1".into());
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .map_err(|e| format!("could not configure thread pool: {e}"))?;
    }

    let samples = args.samples;
    if !(1..=8).contains(&samples) {
        return Err("--samples must be between 1 and 8".into());
    }
    if !args.exposure.is_finite() || args.exposure <= 0.0 {
        return Err("--exposure must be a positive, finite number".into());
    }

    let paths = [args.px, args.nx, args.py, args.ny, args.pz, args.nz];

    let t0 = Instant::now();
    let cube = CubeMap::load(&paths, args.input_transfer)?;
    let face_size = cube.size();
    if !args.quiet {
        eprintln!(
            "loaded 6 x {face_size}x{face_size} faces [{}] in {:.2}s",
            FACE_NAMES.join(" "),
            t0.elapsed().as_secs_f32()
        );
    }

    // 4 x face size keeps the equatorial band at roughly the source sampling
    // rate, which is the usual "no detail lost, no invented detail" choice.
    let width = args.width.unwrap_or((face_size as u32).saturating_mul(4));
    if width < 2 {
        return Err("--width must be at least 2".into());
    }
    if width % 2 != 0 {
        return Err("--width must be even so that height = width/2 is exact".into());
    }
    let height = width / 2;

    let pixels = (width as usize)
        .checked_mul(height as usize)
        .and_then(|p| p.checked_mul(3))
        .ok_or("output dimensions overflow addressable memory")?;

    if !args.quiet {
        eprintln!(
            "rendering {width}x{height} ({} filter, {samples}x{samples} samples/px)",
            match args.filter {
                Filter::Bilinear => "bilinear",
                Filter::Bicubic => "bicubic",
            }
        );
    }

    let t1 = Instant::now();
    let mut out = vec![0.0f32; pixels];

    let wf = width as f32;
    let hf = height as f32;
    let yaw = args.yaw.to_radians();
    let filter = args.filter;
    let sub = samples as f32;
    let norm = args.exposure / (samples * samples) as f32;
    let row_stride = width as usize * 3;

    out.par_chunks_mut(row_stride)
        .enumerate()
        .for_each(|(y, row)| {
            let y0 = y as f32;
            for x in 0..width as usize {
                let x0 = x as f32;
                let mut acc = [0.0f32; 3];

                for sy in 0..samples {
                    let v = y0 + (sy as f32 + 0.5) / sub;
                    for sx in 0..samples {
                        let u = x0 + (sx as f32 + 0.5) / sub;

                        let dir = equirect_to_dir(u, v, wf, hf, yaw);
                        let (face, fu, fv) = dir_to_face_uv(dir);
                        let c = cube.sample(face, fu, fv, filter);

                        acc[0] += c[0];
                        acc[1] += c[1];
                        acc[2] += c[2];
                    }
                }

                let o = x * 3;
                row[o] = acc[0] * norm;
                row[o + 1] = acc[1] * norm;
                row[o + 2] = acc[2] * norm;
            }
        });

    if !args.quiet {
        eprintln!("rendered in {:.2}s", t1.elapsed().as_secs_f32());
    }

    let t2 = Instant::now();
    hdr::write_hdr(&args.output, width, height, &out, !args.no_rle)
        .map_err(|e| format!("{}: {}", args.output.display(), e))?;

    if !args.quiet {
        let bytes = std::fs::metadata(&args.output).map(|m| m.len()).unwrap_or(0);
        eprintln!(
            "wrote {} ({:.1} MiB) in {:.2}s; total {:.2}s",
            args.output.display(),
            bytes as f64 / (1024.0 * 1024.0),
            t2.elapsed().as_secs_f32(),
            t0.elapsed().as_secs_f32()
        );
    }

    Ok(())
}
