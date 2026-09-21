//! Projection maths.
//!
//! Convention (OpenGL / `cmft`-compatible, right-handed, Y-up):
//!
//! ```text
//!   face 0  +X   right
//!   face 1  -X   left
//!   face 2  +Y   up
//!   face 3  -Y   down
//!   face 4  +Z   back
//!   face 5  -Z   front   <-- ends up at the horizontal centre of the panorama
//! ```
//!
//! The equirectangular image spans longitude theta in [-pi, pi] left-to-right and
//! latitude phi in [+pi/2, -pi/2] top-to-bottom. If your faces come from a tool
//! with a different forward axis, `--yaw` rotates the output without any
//! re-authoring of the inputs.

use std::f32::consts::{FRAC_PI_2, PI, TAU};

pub const FACE_NAMES: [&str; 6] = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"];

/// Map a (sub)pixel position in the equirectangular output to a unit direction.
///
/// `u` and `v` are in *continuous* pixel space, i.e. the caller has already
/// added the sample offset within the pixel footprint.
#[inline]
pub fn equirect_to_dir(u: f32, v: f32, width: f32, height: f32, yaw: f32) -> [f32; 3] {
    let theta = (u / width) * TAU - PI + yaw;
    let phi = FRAC_PI_2 - (v / height) * PI;

    let (sin_phi, cos_phi) = phi.sin_cos();
    let (sin_theta, cos_theta) = theta.sin_cos();

    [cos_phi * sin_theta, sin_phi, -cos_phi * cos_theta]
}

/// Select the cube face a direction hits and return the face-local uv in [0,1].
///
/// `u` runs left-to-right and `v` top-to-bottom within the face image, matching
/// the row-major storage of the decoded source images.
#[inline]
pub fn dir_to_face_uv(d: [f32; 3]) -> (usize, f32, f32) {
    let [x, y, z] = d;
    let (ax, ay, az) = (x.abs(), y.abs(), z.abs());

    // (face index, s coordinate, t coordinate, major axis magnitude)
    let (face, sc, tc, ma) = if ax >= ay && ax >= az {
        if x > 0.0 {
            (0usize, -z, -y, ax)
        } else {
            (1usize, z, -y, ax)
        }
    } else if ay >= az {
        if y > 0.0 {
            (2usize, x, z, ay)
        } else {
            (3usize, x, -z, ay)
        }
    } else if z > 0.0 {
        (4usize, x, -y, az)
    } else {
        (5usize, -x, -y, az)
    };

    // `ma` is zero only for a zero-length direction, which we never generate.
    let inv = 0.5 / ma;
    (face, sc * inv + 0.5, tc * inv + 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn face_centres_map_to_uv_centre() {
        let cases = [
            ([1.0, 0.0, 0.0], 0usize),
            ([-1.0, 0.0, 0.0], 1),
            ([0.0, 1.0, 0.0], 2),
            ([0.0, -1.0, 0.0], 3),
            ([0.0, 0.0, 1.0], 4),
            ([0.0, 0.0, -1.0], 5),
        ];
        for (dir, expected) in cases {
            let (face, u, v) = dir_to_face_uv(dir);
            assert_eq!(face, expected, "wrong face for {dir:?}");
            assert!(close(u, 0.5) && close(v, 0.5), "uv not centred: {u} {v}");
        }
    }

    #[test]
    fn panorama_centre_looks_forward() {
        // Centre pixel of a 4x2 image -> theta = 0 -> -Z.
        let d = equirect_to_dir(2.0, 1.0, 4.0, 2.0, 0.0);
        assert!(close(d[0], 0.0) && close(d[1], 0.0) && close(d[2], -1.0), "{d:?}");
        assert_eq!(dir_to_face_uv(d).0, 5);
    }

    #[test]
    fn top_row_looks_up() {
        let d = equirect_to_dir(2.0, 0.0, 4.0, 2.0, 0.0);
        assert!(close(d[1], 1.0), "{d:?}");
        assert_eq!(dir_to_face_uv(d).0, 2);
    }

    #[test]
    fn directions_are_unit_length() {
        for y in 0..17 {
            for x in 0..33 {
                let d = equirect_to_dir(x as f32 + 0.5, y as f32 + 0.5, 33.0, 17.0, 0.3);
                let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                assert!(close(len, 1.0), "len {len}");
            }
        }
    }
}
