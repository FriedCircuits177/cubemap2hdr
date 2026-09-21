//! cube2hdr -- six square cubemap faces -> one equirectangular Radiance `.hdr`.
//!
//! The conversion is output-driven (inverse mapping): for every output pixel we
//! compute the ray it represents, find the cube face that ray pierces, and
//! filter-sample it. That guarantees full coverage with no seams or holes, which
//! forward-projecting each face cannot.
//!
//! Two front-ends share one renderer:
//!   `cube2hdr convert`   -- explicit per-face paths
//!   `cube2hdr minecraft` -- discovers `panorama_0..5.png` in a directory

mod color;
mod cubemap;
mod hdr;
mod projection;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Args as ClapArgs, Parser, Subcommand};
use rayon::prelude::*;

use color::Transfer;
use cubemap::{CubeMap, Filter};
use projection::{FACE_NAMES, dir_to_face_uv, equirect_to_dir};

/// Basename used when `--output` is omitted.
const DEFAULT_OUTPUT_STEM: &str = "cube2hdr_output";

const ABOUT: &str =
    "Convert six square cubemap faces into an equirectangular Radiance .hdr panorama";

const LONG_ABOUT: &str = "\
Convert six square cubemap faces into a single equirectangular panorama in
Radiance .hdr format.

Face convention is OpenGL-style, right-handed and Y-up:

  +X  right
  -X  left
  +Y  up
  -Y  down
  +Z  back
  -Z  front  (lands at the horizontal centre of the panorama)

If your faces came from a tool with a different forward axis, rotate the result
with --yaw instead of renaming or re-rendering files.

Note: .hdr is a float container. Feeding it 8-bit sRGB faces produces a valid
linear-light .hdr, but it cannot invent dynamic range that was not captured.";

const MINECRAFT_LONG_ABOUT: &str = "\
Convert a Minecraft panorama directory into an equirectangular .hdr panorama.

Looks for these files in DIR (default: the current directory) and maps them to
cube faces automatically:

  panorama_0.png  front   -> +Z  (centre of the panorama)
  panorama_1.png  right   -> +X
  panorama_2.png  back    -> -Z
  panorama_3.png  left    -> -X
  panorama_4.png  up      -> +Y  (sky)
  panorama_5.png  down    -> -Y  (ground)

Processing is otherwise identical to `cube2hdr convert`.";

/// Minecraft asset filenames, in the order they appear on disk.
const MINECRAFT_FILES: [&str; 6] = [
    "panorama_0.png",
    "panorama_1.png",
    "panorama_2.png",
    "panorama_3.png",
    "panorama_4.png",
    "panorama_5.png",
];

/// Human-readable role of each `panorama_N.png`, same ordering as above.
const MINECRAFT_ROLES: [&str; 6] = ["front", "right", "back", "left", "up", "down"];

/// Maps our internal face order [+X, -X, +Y, -Y, +Z, -Z] onto the indices of
/// `MINECRAFT_FILES`: 0=front(+Z), 1=right(+X), 2=back(-Z), 3=left(-X),
/// 4=up(+Y), 5=down(-Y).
const MINECRAFT_FACE_ORDER: [usize; 6] = [1, 3, 4, 5, 0, 2];

#[derive(Parser, Debug)]
#[command(name = "cube2hdr", version, about = ABOUT, long_about = LONG_ABOUT)]
#[command(subcommand_required = true, arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Convert six explicitly named cube faces
    Convert(ConvertArgs),
    /// Convert a Minecraft panorama_0..5.png set from a directory
    #[command(long_about = MINECRAFT_LONG_ABOUT)]
    Minecraft(MinecraftArgs),
}

#[derive(ClapArgs, Debug)]
struct ConvertArgs {
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

    #[command(flatten)]
    common: RenderArgs,
}

#[derive(ClapArgs, Debug)]
struct MinecraftArgs {
    /// Directory containing panorama_0.png .. panorama_5.png
    #[arg(value_name = "DIR", default_value = ".")]
    dir: PathBuf,

    #[command(flatten)]
    common: RenderArgs,
}

/// Options shared by every sub-command.
#[derive(ClapArgs, Debug)]
struct RenderArgs {
    /// Output .hdr path [default: cube2hdr_output.hdr, auto-numbered if taken]
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,

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
    #[arg(
        long,
        value_name = "DEG",
        default_value_t = 0.0,
        allow_negative_numbers = true
    )]
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
    let cli = Cli::parse();

    let result = match cli.command {
        Command::Convert(a) => {
            let faces = [a.px, a.nx, a.py, a.ny, a.pz, a.nz];
            render(faces, a.common)
        }
        Command::Minecraft(a) => match minecraft_faces(&a.dir, a.common.quiet) {
            Ok(faces) => render(faces, a.common),
            Err(e) => Err(e),
        },
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Resolve `panorama_0..5.png` in `dir` and reorder them into internal face
/// order [+X, -X, +Y, -Y, +Z, -Z].
fn minecraft_faces(dir: &Path, quiet: bool) -> Result<[PathBuf; 6], String> {
    if !dir.is_dir() {
        return Err(format!("{}: not a directory", dir.display()));
    }

    let on_disk: Vec<PathBuf> = MINECRAFT_FILES.iter().map(|f| dir.join(f)).collect();

    let missing: Vec<&str> = MINECRAFT_FILES
        .iter()
        .zip(&on_disk)
        .filter(|(_, p)| !p.is_file())
        .map(|(n, _)| *n)
        .collect();

    if !missing.is_empty() {
        return Err(format!(
            "{}: missing Minecraft panorama file(s): {}",
            dir.display(),
            missing.join(", ")
        ));
    }

    if !quiet {
        eprintln!("found Minecraft panorama set in {}", dir.display());
        for (i, face) in MINECRAFT_FACE_ORDER.iter().enumerate() {
            eprintln!(
                "  {} <- {} ({})",
                FACE_NAMES[i], MINECRAFT_FILES[*face], MINECRAFT_ROLES[*face]
            );
        }
    }

    let ordered = MINECRAFT_FACE_ORDER.map(|i| on_disk[i].clone());
    Ok(ordered)
}

/// Pick the output path: the user's choice, or `cube2hdr_output.hdr` with a
/// `_1`, `_2`, ... suffix appended until an unused name is found.
fn resolve_output(requested: Option<PathBuf>) -> Result<PathBuf, String> {
    if let Some(p) = requested {
        return Ok(p);
    }

    let first = PathBuf::from(format!("{DEFAULT_OUTPUT_STEM}.hdr"));
    if !first.exists() {
        return Ok(first);
    }

    for n in 1..=u32::MAX {
        let candidate = PathBuf::from(format!("{DEFAULT_OUTPUT_STEM}_{n}.hdr"));
        if !candidate.exists() {
            return Ok(candidate);
        }
    }

    Err("could not find an unused default output filename".into())
}

/// Faces are in internal order: [+X, -X, +Y, -Y, +Z, -Z].
fn render(faces: [PathBuf; 6], args: RenderArgs) -> Result<(), String> {
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

    // Resolve the destination before doing the expensive work, so a bad path
    // fails immediately rather than after a minute of rendering.
    let output = resolve_output(args.output)?;

    let t0 = Instant::now();
    let cube = CubeMap::load(&faces, args.input_transfer)?;
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
    hdr::write_hdr(&output, width, height, &out, !args.no_rle)
        .map_err(|e| format!("{}: {}", output.display(), e))?;

    if !args.quiet {
        let bytes = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
        eprintln!(
            "wrote {} ({:.1} MiB) in {:.2}s; total {:.2}s",
            output.display(),
            bytes as f64 / (1024.0 * 1024.0),
            t2.elapsed().as_secs_f32(),
            t0.elapsed().as_secs_f32()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minecraft_mapping_is_the_documented_one() {
        // Internal order is [+X, -X, +Y, -Y, +Z, -Z].
        let expect = [
            ("+X", "panorama_1.png", "right"),
            ("-X", "panorama_3.png", "left"),
            ("+Y", "panorama_4.png", "up"),
            ("-Y", "panorama_5.png", "down"),
            ("-Z", "panorama_0.png", "front"),
            ("+Z", "panorama_2.png", "back"),
        ];
        for (i, (axis, file, role)) in expect.iter().enumerate() {
            let src = MINECRAFT_FACE_ORDER[i];
            assert_eq!(FACE_NAMES[i], *axis);
            assert_eq!(MINECRAFT_FILES[src], *file);
            assert_eq!(MINECRAFT_ROLES[src], *role);
        }
    }

    #[test]
    fn explicit_output_is_passed_through() {
        let p = PathBuf::from("somewhere/custom.hdr");
        assert_eq!(resolve_output(Some(p.clone())).unwrap(), p);
    }
}
