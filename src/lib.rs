//! Rust functions for mapping between 1D and 2D space using the Hilbert curve, and its approximations.
//! 
//! When working with images and matrices, use the `h2xy_discrete` and `xy2h_discrete` functions:
//! 
//! ```rust
//! use hilbert_2d::{h2xy_discrete, xy2h_discrete, Variant};
//! 
//! let (x, y) = h2xy_discrete(7, 2, Variant::Hilbert); // (1, 2)
//! let h = xy2h_discrete(2, 1, 2, Variant::Hilbert); // 13
//! ```
//! 
//! When performing real-valued calculations, use the continuous functions instead:
//! 
//! ```rust
//! use hilbert_2d::{h2xy_continuous_f64, Variant};
//! 
//! // Approaches the bottom-left corner
//! let (x1, y1) = h2xy_continuous_f64(0.0, Variant::Hilbert); 
//! // Approaches the bottom-right corner
//! let (x2, y2) = h2xy_continuous_f64(1.0, Variant::Hilbert); 
//! ```
//! 
//! Some of the pattern variants of the Hilbert curve have also been implemented:
//! 
//! ```rust
//! use hilbert_2d::{h2xy_continuous_f64, Variant};
//! 
//! // In the Liu L1 variant, both ends of the curve approach the center of the square
//! let (x1, y1) = h2xy_continuous_f64(0.0, Variant::Liu1); // (~0.5, ~0.5)
//! let (x2, y2) = h2xy_continuous_f64(1.0, Variant::Liu1); // (~0.5, ~0.5)
//! ```

#![doc(html_root_url = "https://docs.rs/hilbert_2d/1.0.0")]

/// Number of bits in `usize` for the current platform
const USIZE_BITS: u32 = usize::MAX.count_ones();

/// Highest order of the Hilbert curve that can be calculated in the current platform
const ORDER_MAX: u32 = USIZE_BITS / 2;

/// Lookup tables for the 1D->2D conversions.
const LUTS_H2XY: [[(usize, usize); 4]; 8] = [
    // 1 ― 2
    // |   |
    // 0   3  Index 0b000
    [(0, 0), (0, 1), (1, 1), (1, 0)],
    // 3 ― 2
    //     |
    // 0 ― 1  Index 0b001
    [(0, 0), (1, 0), (1, 1), (0, 1)],
    // 1 ― 0
    // |
    // 2 ― 3  Index 0b010
    [(1, 1), (0, 1), (0, 0), (1, 0)],
    // 3   0
    // |   |
    // 2 ― 1  Index 0b011
    [(1, 1), (1, 0), (0, 0), (0, 1)],
    // 2 ― 1
    // |   |
    // 3   0  Index 0b100
    [(1, 0), (1, 1), (0, 1), (0, 0)],
    // 2 ― 3
    // |
    // 1 ― 0  Index 0b101
    [(1, 0), (0, 0), (0, 1), (1, 1)],
    // 0 ― 1
    //     |
    // 3 ― 2  Index 0b110
    [(0, 1), (1, 1), (1, 0), (0, 0)],
    // 0   3
    // |   |
    // 1 ― 2  Index 0b111
    [(0, 1), (0, 0), (1, 0), (1, 1)],
];

/// Lookup tables for the 2D->1D conversions.
const LUTS_YX2H: [[[usize; 2]; 2]; 8] = [
    // 1 ― 2
    // |   |
    // 0   3  Index 0b000
    [[0, 3], [1, 2]],
    // 3 ― 2
    //     |
    // 0 ― 1  Index 0b001
    [[0, 1], [3, 2]],
    // 1 ― 0
    // |
    // 2 ― 3  Index 0b010
    [[2, 3], [1, 0]],
    // 3   0
    // |   |
    // 2 ― 1  Index 0b011
    [[2, 1], [3, 0]],
    // 2 ― 1
    // |   |
    // 3   0  Index 0b100
    [[3, 0], [2, 1]],
    // 2 ― 3
    // |
    // 1 ― 0  Index 0b101
    [[1, 0], [2, 3]],
    // 0 ― 1
    //     |
    // 3 ― 2  Index 0b110
    [[3, 2], [0, 1]],
    // 0   3
    // |   |
    // 1 ― 2  Index 0b111
    [[1, 2], [0, 3]],
];

/// Indicates the pattern variant of the Hilbert curve to be constructed.
///
/// Different patterns of the Hilbert curve have been demonstrated, and they can also be constructed by the methods
/// present in this library.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum Variant {
    /// The traditional pattern, as presented by David Hilbert. [(Hilbert, D., 1935)](https://doi.org/10.1007/BF01199431)
    Hilbert,
    /// The pattern variant introduced by E. H. Moore. [(Moore, E.H., 1900)](https://doi.org/10.1090/S0002-9947-1900-1500526-4)
    Moore,
    /// The pattern L1, as presented by Xian Liu. [(Liu, X., 2004)](https://doi.org/10.1016/S0096-3003(02)00808-1)
    Liu1,
    /// The pattern L2, as presented by Xian Liu. [(Liu, X., 2004)](https://doi.org/10.1016/S0096-3003(02)00808-1)
    Liu2,
    /// The pattern L3, as presented by Xian Liu. [(Liu, X., 2004)](https://doi.org/10.1016/S0096-3003(02)00808-1)
    Liu3,
    /// The pattern L4, as presented by Xian Liu. [(Liu, X., 2004)](https://doi.org/10.1016/S0096-3003(02)00808-1)
    Liu4,
}

/// Determines the index of the next lookup table to be used.
///
/// The conversion from/to the Hilbert curve is done using a matrix, that is transformed after each subsequent order,
/// based on the curve quadrant for that transition. We, instead, use a set of lookup tables containing all possible
/// transition matrices, and simply change the lookup index to indicate which matrix to use next.
const fn next_lut_index(lut_index: usize, quadrant: usize) -> usize {
    match quadrant {
        0 => lut_index ^ 0b001,
        3 => lut_index ^ 0b010,
        _ => lut_index,
    }
}

/// Determines the index of the lookup table to be used in the transition from order 1 to order 2.
///
/// See [`next_lut_index`](fn.next_lut_index.html) for more details.
const fn next_lut_index_variant(
    lut_index: usize,
    cur_quadrant: usize,
    variant: Variant,
) -> usize {
    match variant {
        Variant::Moore => match cur_quadrant {
            0 | 1 => (lut_index ^ 0b10) ^ 0b111,
            _ => (lut_index ^ 0b01) ^ 0b111,
        },
        Variant::Liu1 => match cur_quadrant {
            0 | 3 => (lut_index ^ 0b01) ^ 0b10,
            _ => lut_index,
        },
        Variant::Liu2 => match cur_quadrant {
            1 => (lut_index ^ 0b10) ^ 0b111,
            2 => (lut_index ^ 0b01) ^ 0b111,
            _ => lut_index ^ 0b111,
        },
        Variant::Liu3 => match cur_quadrant {
            0 => lut_index ^ 0b01,
            3 => (lut_index ^ 0b01) ^ 0b10,
            _ => lut_index,
        },
        Variant::Liu4 => match cur_quadrant {
            0 => lut_index ^ 0b111,
            1 => (lut_index ^ 0b10) ^ 0b111,
            _ => (lut_index ^ 0b01) ^ 0b111,
        },
        _ => next_lut_index(lut_index, cur_quadrant),
    }
}

/// Maps from a 1D index to a 2D coordinate, using a discrete approximation of the Hilbert curve. Recommended for
/// images and matrices.
///
/// Given `h`, this method calculates the `(x, y)` coordinates for that index, in the Hilbert curve approximation of
/// `order`.
///
/// The value of `h` must be in the range `0 <= h < 2^(2 * order)`. The coordinates returned will be in the range
/// `0 <= x/y < 2^order`.
///
/// With `b` being the number of bits in a `usize` for the target platform, the highest approximation order achievable
/// by this method is `b / 2`. Therefore, the parameter `order` must be in the range `1 <= order <= b / 2`.
///
/// The pattern of the Hilbert curve to be constructed must be indicated by the `variant` parameter. See [`Variant`] 
/// for more details.
///
/// # Examples
///
/// Basic usage:
///
/// ```rust
/// use hilbert_2d::{h2xy_discrete, Variant};
///
/// // Hilbert curve of order 2:
/// //  5 ―― 6    9 ― 10
/// //  |    |    |    |
/// //  4    7 ―― 8   11
/// //  |              |
/// //  3 ―― 2   13 ― 12
/// //       |    |
/// //  0 ―― 1   14 ― 15
/// let (x, y) = h2xy_discrete(7, 2, Variant::Hilbert);
///
/// assert_eq!(x, 1);
/// assert_eq!(y, 2);
/// ```
///
/// [`Variant`]: enum.Variant.html
///
pub const fn h2xy_discrete(h: usize, order: usize, variant: Variant) -> (usize, usize) {
    // Records how many >>'s are required to isolate the quadrant crumbs for each subsequent order, starting at 1
    let mut steps = 2 * (order - 1);
    if steps as u32 >= USIZE_BITS {
        steps = USIZE_BITS as usize - 2;
    }
    // Extract the crumb for the quadrant of order 1 from `curve_p`
    let mut cur_quadrant = (h >> steps) & 0b11;
    // Retrieve the square coordinates for that quadrant, using the root LUT
    let (mut square_x, mut square_y) = LUTS_H2XY[0b000][cur_quadrant];

    // Stop here if the given order is 1
    if steps == 0 {
        return (square_x, square_y);
    }

    // Depending on the curve variant, the method of choosing the LUT for the `order 1 -> order 2` transition changes
    let mut lut_index = next_lut_index_variant(0b000, cur_quadrant, variant);

    loop {
        // Extract the crumb for the next quadrant from `curve_p`
        steps -= 2;
        cur_quadrant = (h >> steps) & 0b11;

        // Retrieve the square coordinates for that quadrant, using the current LUT
        square_x = (square_x << 1) | LUTS_H2XY[lut_index][cur_quadrant].0;
        square_y = (square_y << 1) | LUTS_H2XY[lut_index][cur_quadrant].1;

        if steps > 0 {
            // From `order 2 -> order 3` and beyond, the same method of choosing the next LUT is used
            lut_index = next_lut_index(lut_index, cur_quadrant);
        } else {
            break;
        }
    }

    (square_x, square_y)
}

/// Maps from a 2D coordinate to an 1D index, using a discrete approximation of the Hilbert curve. Recommended for
/// images and matrices.
///
/// Given `x` and `y`, this method calculates the `h` index for that coordinate, in the Hilbert curve approximation of
/// `order`.
///
/// The value of `x` and `y` must be in the range `0 <= x/y < 2^order`. The index returned will be in the range
/// `0 <= h < 2^(2 * order)`.
///
/// With `b` being the number of bits in a `usize` for the target platform, the highest approximation order achievable
/// by this method is `b / 2`. Therefore, the parameter `order` must be in the range `1 <= order <= b / 2`.
///
/// The pattern of the Hilbert curve to be constructed must be indicated by the `variant` parameter. See
/// [`Variant`] for more details.
///
/// # Examples
///
/// Basic usage:
///
/// ```rust
/// use hilbert_2d::{xy2h_discrete, Variant};
///
/// // Hilbert curve of order 2:
/// //  5 ―― 6    9 ― 10
/// //  |    |    |    |
/// //  4    7 ―― 8   11
/// //  |              |
/// //  3 ―― 2   13 ― 12
/// //       |    |
/// //  0 ―― 1   14 ― 15
/// let h = xy2h_discrete(2, 1, 2, Variant::Hilbert);
///
/// assert_eq!(h, 13);
/// ```
///
/// [`Variant`]: enum.Variant.html
///
pub const fn xy2h_discrete(x: usize, y: usize, order: usize, variant: Variant) -> usize {
    // Records how many >>'s are required to isolate the quadrant bits for each subsequent order, starting at 1
    let mut steps = order - 1;
    if steps as u32 >= ORDER_MAX {
        steps = ORDER_MAX as usize - 1;
    }

    // Extract the bits for the square region from the `square_x` and `square_x` coordinates
    let mut q_x = (x >> steps) & 0b1;
    let mut q_y = (y >> steps) & 0b1;

    // Retrieve the quadrant of order 1 equivalent to that square region
    let mut cur_quadrant = LUTS_YX2H[0b000][q_y][q_x];
    let mut curve_p = cur_quadrant;

    // Stop here if the given order is 1
    if steps == 0 {
        return curve_p;
    }

    // Depending on the curve variant, the method of choosing the LUT for the `order 1 -> order 2` transition changes
    let mut lut_index = next_lut_index_variant(0b000, cur_quadrant, variant);

    loop {
        // Extract the bits for the next square region from the `square_x` and `square_x` coordinates
        steps -= 1;
        q_x = (x >> steps) & 0b1;
        q_y = (y >> steps) & 0b1;

        // Retrieve the quadrant equivalent to that square region, using the current LUT
        cur_quadrant = LUTS_YX2H[lut_index][q_y][q_x];
        curve_p = (curve_p << 2) | cur_quadrant;

        if steps > 0 {
            // From `order 2 -> order 3` and beyond, the same method of choosing the next LUT is used
            lut_index = next_lut_index(lut_index, cur_quadrant);
        } else {
            break;
        }
    }

    curve_p
}

/// Maps from a 1D value to an approximate 2D coordinate, using the closest approachable limit of the Hilbert curve. 
/// Recommended for real-valued calculations.
///
/// Given a value `h`, this method calculates an approximation of it's corresponding limit point `(x, y)`, in the
/// Hilbert curve of the highest achievable order for the target platform.
///
/// The value of `h` must be in the range `0.0 <= h < 1.0`. A value `h >= 1.0` can be given, but functionally, it will
/// be clipped to the closest value to one possible. The coordinates returned will be in the range `0.0 <= x/y < 1.0`.
///
/// The pattern of the Hilbert curve to be constructed must be indicated by the `variant` parameter. See [`Variant`]
/// for more details.
///
/// Internally, this method uses [`h2xy_discrete`] for the calculation, coupled with a trick using the fractional parts
/// of binary/quaternary numbers. In short, the precision of the calculation, as well as the closest approximation
/// possible, both depend on the number of bits in a `usize` for the target platform. See [`h2xy_discrete`] for more
/// details.
///
/// # Examples
///
/// Basic usage:
///
/// ```rust
/// use hilbert_2d::{h2xy_continuous_f32, Variant};
///
/// // In the traditional pattern, the edges of the Hilbert curve are horizontally distant...
/// let (x_a, _) = h2xy_continuous_f32(0.0, Variant::Hilbert);
/// let (x_b, _) = h2xy_continuous_f32(1.0, Variant::Hilbert);
/// assert!(x_b - x_a > 0.999);
///
/// // ...but in the Moore pattern, the edges of the curve close up.
/// let (x_a, _) = h2xy_continuous_f32(0.0, Variant::Moore);
/// let (x_b, _) = h2xy_continuous_f32(1.0, Variant::Moore);
/// assert!(x_b - x_a < 0.001);
/// ```
///
/// [`h2xy_discrete`]: fn.h2xy_discrete.html
/// [`Variant`]: enum.Variant.html
///
pub fn h2xy_continuous_f32(h: f32, variant: Variant) -> (f32, f32) {
    // Convert decimal `h` into fractional representation, and calculate the square position
    let (square_x, square_y) = h2xy_discrete(f32_to_binfrac(h), ORDER_MAX as usize, variant);
    // Convert the fractional result to decimal and return
    (
        binfrac_to_f32(square_x << ORDER_MAX),
        binfrac_to_f32(square_y << ORDER_MAX),
    )
}

/// Maps from a 1D value to an approximate 2D coordinate, using the closest approachable limit of the Hilbert curve. 
/// Recommended for real-valued calculations.
///
/// Given a value `h`, this method calculates an approximation of it's corresponding limit point `(x, y)`, in the
/// Hilbert curve of the highest achievable order for the target platform.
///
/// The value of `h` must be in the range `0.0 <= h < 1.0`. A value `h >= 1.0` can be given, but functionally, it will
/// be clipped to the closest value to one possible. The coordinates returned will be in the range `0.0 <= x/y < 1.0`.
///
/// The pattern of the Hilbert curve to be constructed must be indicated by the `variant` parameter. See [`Variant`]
/// for more details.
///
/// Internally, this method uses [`h2xy_discrete`] for the calculation, coupled with a trick using the fractional parts
/// of binary/quaternary numbers. In short, the precision of the calculation, as well as the closest approximation
/// possible, both depend on the number of bits in a `usize` for the target platform. See [`h2xy_discrete`] for more
/// details.
///
/// # Examples
///
/// Basic usage:
///
/// ```rust
/// use hilbert_2d::{h2xy_continuous_f64, Variant};
///
/// // In the traditional pattern, the edges of the Hilbert curve are horizontally distant...
/// let (x_a, _) = h2xy_continuous_f64(0.0, Variant::Hilbert);
/// let (x_b, _) = h2xy_continuous_f64(1.0, Variant::Hilbert);
/// assert!(x_b - x_a > 0.999);
///
/// // ...but in the Moore pattern, the edges of the curve close up.
/// let (x_a, _) = h2xy_continuous_f64(0.0, Variant::Moore);
/// let (x_b, _) = h2xy_continuous_f64(1.0, Variant::Moore);
/// assert!(x_b - x_a < 0.001);
/// ```
///
/// [`h2xy_discrete`]: fn.h2xy_discrete.html
/// [`Variant`]: enum.Variant.html
///
pub fn h2xy_continuous_f64(h: f64, variant: Variant) -> (f64, f64) {
    // Convert decimal `h` into fractional representation, and calculate the square position
    let (square_x, square_y) = h2xy_discrete(f64_to_binfrac(h), ORDER_MAX as usize, variant);
    // Convert the fractional result to decimal and return
    (
        binfrac_to_f64(square_x << ORDER_MAX),
        binfrac_to_f64(square_y << ORDER_MAX),
    )
}

/// Maps from a 2D coordinate to an approximate 1D value, using the closest approachable limit of the Hilbert curve. 
/// Recommended for real-valued calculations.
///
/// Given `x` and `y`, this method calculates an approximation of it's corresponding limit `h`, in the Hilbert curve
/// of the highest achievable order for the target platform.
///
/// The value of `x` and `y` must be in the range `0.0 <= x/y < 1.0`. A value `x/y >= 1.0` can be given, but
/// functionally, it will be clipped to the closest value to one possible. The value returned will be in the range
/// `0.0 <= h <= 1.0`.
///
/// The pattern of the Hilbert curve to be constructed must be indicated by the `variant` parameter. See [`Variant`]
/// for more details.
///
/// Internally, this method uses [`xy2h_discrete`] for the calculation, coupled with a trick using the fractional parts
/// of binary/quaternary numbers. In short, the precision of the calculation, as well as the closest approximation
/// possible, both depend on the number of bits in a `usize` for the target platform. See [`xy2h_discrete`] for more
/// details.
///
/// # Examples
///
/// Basic usage:
///
/// ```rust
/// use hilbert_2d::{xy2h_continuous_f32, Variant};
///
/// // In the third pattern presented by Liu, the curve begins at the lower left corner...
/// let h = xy2h_continuous_f32(0.0, 0.0, Variant::Liu3);
/// assert!(h < 0.001);
///
/// // ...but ends near the center of the square.
/// let h = xy2h_continuous_f32(0.5001, 0.4999, Variant::Liu3);
/// assert!(h > 0.999);
/// ```
///
/// [`xy2h_discrete`]: fn.xy2h_discrete.html
/// [`Variant`]: enum.Variant.html
///
pub fn xy2h_continuous_f32(x: f32, y: f32, variant: Variant) -> f32 {
    // Convert the x/y coordinates into a factional representation, and calculate the index
    let h = xy2h_discrete(
        f32_to_binfrac(x) >> ORDER_MAX,
        f32_to_binfrac(y) >> ORDER_MAX,
        ORDER_MAX as usize,
        variant,
    );
    // Convert the fractional result back into decimal, and return
    binfrac_to_f32(h)
}

/// Maps from a 2D coordinate to an approximate 1D value, using the closest approachable limit of the Hilbert curve. 
/// Recommended for real-valued calculations.
///
/// Given `x` and `y`, this method calculates an approximation of it's corresponding limit `h`, in the Hilbert curve
/// of the highest achievable order for the target platform.
///
/// The value of `x` and `y` must be in the range `0.0 <= x/y < 1.0`. A value `x/y >= 1.0` can be given, but
/// functionally, it will be clipped to the closest value to one possible. The value returned will be in the range
/// `0.0 <= h <= 1.0`.
///
/// The pattern of the Hilbert curve to be constructed must be indicated by the `variant` parameter. See [`Variant`]
/// for more details.
///
/// Internally, this method uses [`xy2h_discrete`] for the calculation, coupled with a trick using the fractional parts
/// of binary/quaternary numbers. In short, the precision of the calculation, as well as the closest approximation
/// possible, both depend on the number of bits in a `usize` for the target platform. See [`xy2h_discrete`] for more
/// details.
///
/// # Examples
///
/// Basic usage:
///
/// ```rust
/// use hilbert_2d::{xy2h_continuous_f64, Variant};
///
/// // In the third pattern presented by Liu, the curve begins at the lower left corner...
/// let h = xy2h_continuous_f64(0.0, 0.0, Variant::Liu3);
/// assert!(h < 0.001);
///
/// // ...but ends near the center of the square.
/// let h = xy2h_continuous_f64(0.5001, 0.4999, Variant::Liu3);
/// assert!(h > 0.999);
/// ```
///
/// [`xy2h_discrete`]: fn.xy2h_discrete.html
/// [`Variant`]: enum.Variant.html
///
pub fn xy2h_continuous_f64(x: f64, y: f64, variant: Variant) -> f64 {
    // Convert the x/y coordinates into a factional representation, and calculate the index
    let h = xy2h_discrete(
        f64_to_binfrac(x) >> ORDER_MAX,
        f64_to_binfrac(y) >> ORDER_MAX,
        ORDER_MAX as usize,
        variant,
    );
    // Convert the fractional result back into decimal, and return
    binfrac_to_f64(h)
}

/// Calculates the lowest decimal number that could be represented in a binary fraction with N decimal places, where N
/// is the number of bits in `usize`.
fn lowest_decimal_f32() -> f32 {
    2.0f32.powi(-(USIZE_BITS as i32))
}

/// Calculates the lowest decimal number that could be represented in a binary fraction with N decimal places, where N
/// is the number of bits in `usize`.
fn lowest_decimal_f64() -> f64 {
    2.0f64.powi(-(USIZE_BITS as i32))
}

/// Convert the fractional part of a binary decimal number to a 32-bit floating point number.
fn binfrac_to_f32(frac: usize) -> f32 {
    lowest_decimal_f32() * (frac as f32)
}

/// Convert the fractional part of a binary decimal number to a 64-bit floating point number.
fn binfrac_to_f64(frac: usize) -> f64 {
    lowest_decimal_f64() * (frac as f64)
}

/// Extracts the fractional part, in binary, of a 32-bit floating point number.
fn f32_to_binfrac(dec: f32) -> usize {
    (dec / lowest_decimal_f32()) as usize
}

/// Extracts the fractional part, in binary, of a 64-bit floating point number.
fn f64_to_binfrac(dec: f64) -> usize {
    (dec / lowest_decimal_f64()) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn lut_transition_standard() {
        // LUT transition for the Hilbert variant, with the curve 0b011 as the seed
        assert_eq!(next_lut_index(0b011, 0), 0b010);
        assert_eq!(next_lut_index(0b011, 1), 0b011);
        assert_eq!(next_lut_index(0b011, 2), 0b011);
        assert_eq!(next_lut_index(0b011, 3), 0b001);
    }

    #[test]
    fn lut_transition_variant() {
        // LUT transition for the Moore variant, with the curve 0b111 as the seed
        assert_eq!(next_lut_index_variant(0b111, 0, Variant::Moore), 0b010);
        assert_eq!(next_lut_index_variant(0b111, 1, Variant::Moore), 0b010);
        assert_eq!(next_lut_index_variant(0b111, 2, Variant::Moore), 0b001);
        assert_eq!(next_lut_index_variant(0b111, 3, Variant::Moore), 0b001);
        // LUT transition for the Liu1 variant, with the curve 0b111 as the seed
        assert_eq!(next_lut_index_variant(0b111, 0, Variant::Liu1), 0b100);
        assert_eq!(next_lut_index_variant(0b111, 1, Variant::Liu1), 0b111);
        assert_eq!(next_lut_index_variant(0b111, 2, Variant::Liu1), 0b111);
        assert_eq!(next_lut_index_variant(0b111, 3, Variant::Liu1), 0b100);
        // LUT transition for the Liu2 variant, with the curve 0b111 as the seed
        assert_eq!(next_lut_index_variant(0b111, 0, Variant::Liu2), 0b000);
        assert_eq!(next_lut_index_variant(0b111, 1, Variant::Liu2), 0b010);
        assert_eq!(next_lut_index_variant(0b111, 2, Variant::Liu2), 0b001);
        assert_eq!(next_lut_index_variant(0b111, 3, Variant::Liu2), 0b000);
        // LUT transition for the Liu3 variant, with the curve 0b111 as the seed
        assert_eq!(next_lut_index_variant(0b111, 0, Variant::Liu3), 0b110);
        assert_eq!(next_lut_index_variant(0b111, 1, Variant::Liu3), 0b111);
        assert_eq!(next_lut_index_variant(0b111, 2, Variant::Liu3), 0b111);
        assert_eq!(next_lut_index_variant(0b111, 3, Variant::Liu3), 0b100);
        // LUT transition for the Liu4 variant, with the curve 0b111 as the seed
        assert_eq!(next_lut_index_variant(0b111, 0, Variant::Liu4), 0b000);
        assert_eq!(next_lut_index_variant(0b111, 1, Variant::Liu4), 0b010);
        assert_eq!(next_lut_index_variant(0b111, 2, Variant::Liu4), 0b001);
        assert_eq!(next_lut_index_variant(0b111, 3, Variant::Liu4), 0b001);
    }

    #[test]
    fn map_discrete_hilbert() {
        // Test for the central coordinates of a order 3 curve
        assert_eq!(h2xy_discrete(8, 3, Variant::Hilbert), (2, 2));
        assert_eq!(h2xy_discrete(9, 3, Variant::Hilbert), (3, 2));
        assert_eq!(xy2h_discrete(4, 2, 3, Variant::Hilbert), 54);
        assert_eq!(xy2h_discrete(5, 2, 3, Variant::Hilbert), 55);
        assert_eq!(h2xy_discrete(11, 3, Variant::Hilbert), (2, 3));
        assert_eq!(h2xy_discrete(10, 3, Variant::Hilbert), (3, 3));
        assert_eq!(xy2h_discrete(4, 3, 3, Variant::Hilbert), 53);
        assert_eq!(xy2h_discrete(5, 3, 3, Variant::Hilbert), 52);
        assert_eq!(xy2h_discrete(2, 4, 3, Variant::Hilbert), 30);
        assert_eq!(xy2h_discrete(3, 4, 3, Variant::Hilbert), 31);
        assert_eq!(h2xy_discrete(32, 3, Variant::Hilbert), (4, 4));
        assert_eq!(h2xy_discrete(33, 3, Variant::Hilbert), (5, 4));
        assert_eq!(xy2h_discrete(2, 5, 3, Variant::Hilbert), 29);
        assert_eq!(xy2h_discrete(3, 5, 3, Variant::Hilbert), 28);
        assert_eq!(h2xy_discrete(35, 3, Variant::Hilbert), (4, 5));
        assert_eq!(h2xy_discrete(34, 3, Variant::Hilbert), (5, 5));
    }

    #[test]
    fn map_discrete_moore() {
        // Test for the central coordinates of a order 3 curve
        assert_eq!(h2xy_discrete(13, 3, Variant::Moore), (2, 2));
        assert_eq!(h2xy_discrete(14, 3, Variant::Moore), (3, 2));
        assert_eq!(xy2h_discrete(4, 2, 3, Variant::Moore), 49);
        assert_eq!(xy2h_discrete(5, 2, 3, Variant::Moore), 50);
        assert_eq!(h2xy_discrete(12, 3, Variant::Moore), (2, 3));
        assert_eq!(h2xy_discrete(15, 3, Variant::Moore), (3, 3));
        assert_eq!(xy2h_discrete(4, 3, 3, Variant::Moore), 48);
        assert_eq!(xy2h_discrete(5, 3, 3, Variant::Moore), 51);
        assert_eq!(xy2h_discrete(2, 4, 3, Variant::Moore), 19);
        assert_eq!(xy2h_discrete(3, 4, 3, Variant::Moore), 16);
        assert_eq!(h2xy_discrete(47, 3, Variant::Moore), (4, 4));
        assert_eq!(h2xy_discrete(44, 3, Variant::Moore), (5, 4));
        assert_eq!(xy2h_discrete(2, 5, 3, Variant::Moore), 18);
        assert_eq!(xy2h_discrete(3, 5, 3, Variant::Moore), 17);
        assert_eq!(h2xy_discrete(46, 3, Variant::Moore), (4, 5));
        assert_eq!(h2xy_discrete(45, 3, Variant::Moore), (5, 5));
    }

    #[test]
    fn map_discrete_liu1() {
        // Test for the central coordinates of a order 3 curve
        assert_eq!(h2xy_discrete(2, 3, Variant::Liu1), (2, 2));
        assert_eq!(h2xy_discrete(3, 3, Variant::Liu1), (3, 2));
        assert_eq!(xy2h_discrete(4, 2, 3, Variant::Liu1), 60);
        assert_eq!(xy2h_discrete(5, 2, 3, Variant::Liu1), 61);
        assert_eq!(h2xy_discrete(1, 3, Variant::Liu1), (2, 3));
        assert_eq!(h2xy_discrete(0, 3, Variant::Liu1), (3, 3));
        assert_eq!(xy2h_discrete(4, 3, 3, Variant::Liu1), 63);
        assert_eq!(xy2h_discrete(5, 3, 3, Variant::Liu1), 62);
        assert_eq!(xy2h_discrete(2, 4, 3, Variant::Liu1), 30);
        assert_eq!(xy2h_discrete(3, 4, 3, Variant::Liu1), 31);
        assert_eq!(h2xy_discrete(32, 3, Variant::Liu1), (4, 4));
        assert_eq!(h2xy_discrete(33, 3, Variant::Liu1), (5, 4));
        assert_eq!(xy2h_discrete(2, 5, 3, Variant::Liu1), 29);
        assert_eq!(xy2h_discrete(3, 5, 3, Variant::Liu1), 28);
        assert_eq!(h2xy_discrete(35, 3, Variant::Liu1), (4, 5));
        assert_eq!(h2xy_discrete(34, 3, Variant::Liu1), (5, 5));
    }

    #[test]
    fn map_discrete_liu2() {
        // Test for the central coordinates of a order 3 curve
        assert_eq!(h2xy_discrete(13, 3, Variant::Liu2), (2, 2));
        assert_eq!(h2xy_discrete(12, 3, Variant::Liu2), (3, 2));
        assert_eq!(xy2h_discrete(4, 2, 3, Variant::Liu2), 51);
        assert_eq!(xy2h_discrete(5, 2, 3, Variant::Liu2), 50);
        assert_eq!(h2xy_discrete(14, 3, Variant::Liu2), (2, 3));
        assert_eq!(h2xy_discrete(15, 3, Variant::Liu2), (3, 3));
        assert_eq!(xy2h_discrete(4, 3, 3, Variant::Liu2), 48);
        assert_eq!(xy2h_discrete(5, 3, 3, Variant::Liu2), 49);
        assert_eq!(xy2h_discrete(2, 4, 3, Variant::Liu2), 19);
        assert_eq!(xy2h_discrete(3, 4, 3, Variant::Liu2), 16);
        assert_eq!(h2xy_discrete(47, 3, Variant::Liu2), (4, 4));
        assert_eq!(h2xy_discrete(44, 3, Variant::Liu2), (5, 4));
        assert_eq!(xy2h_discrete(2, 5, 3, Variant::Liu2), 18);
        assert_eq!(xy2h_discrete(3, 5, 3, Variant::Liu2), 17);
        assert_eq!(h2xy_discrete(46, 3, Variant::Liu2), (4, 5));
        assert_eq!(h2xy_discrete(45, 3, Variant::Liu2), (5, 5));
    }

    #[test]
    fn map_discrete_liu3() {
        // Test for the central coordinates of a order 3 curve
        assert_eq!(h2xy_discrete(8, 3, Variant::Liu3), (2, 2));
        assert_eq!(h2xy_discrete(9, 3, Variant::Liu3), (3, 2));
        assert_eq!(xy2h_discrete(4, 2, 3, Variant::Liu3), 60);
        assert_eq!(xy2h_discrete(5, 2, 3, Variant::Liu3), 61);
        assert_eq!(h2xy_discrete(11, 3, Variant::Liu3), (2, 3));
        assert_eq!(h2xy_discrete(10, 3, Variant::Liu3), (3, 3));
        assert_eq!(xy2h_discrete(4, 3, 3, Variant::Liu3), 63);
        assert_eq!(xy2h_discrete(5, 3, 3, Variant::Liu3), 62);
        assert_eq!(xy2h_discrete(2, 4, 3, Variant::Liu3), 30);
        assert_eq!(xy2h_discrete(3, 4, 3, Variant::Liu3), 31);
        assert_eq!(h2xy_discrete(32, 3, Variant::Liu3), (4, 4));
        assert_eq!(h2xy_discrete(33, 3, Variant::Liu3), (5, 4));
        assert_eq!(xy2h_discrete(2, 5, 3, Variant::Liu3), 29);
        assert_eq!(xy2h_discrete(3, 5, 3, Variant::Liu3), 28);
        assert_eq!(h2xy_discrete(35, 3, Variant::Liu3), (4, 5));
        assert_eq!(h2xy_discrete(34, 3, Variant::Liu3), (5, 5));
    }

    #[test]
    fn map_discrete_liu4() {
        // Test for the central coordinates of a order 3 curve
        assert_eq!(h2xy_discrete(13, 3, Variant::Liu4), (2, 2));
        assert_eq!(h2xy_discrete(12, 3, Variant::Liu4), (3, 2));
        assert_eq!(xy2h_discrete(4, 2, 3, Variant::Liu4), 49);
        assert_eq!(xy2h_discrete(5, 2, 3, Variant::Liu4), 50);
        assert_eq!(h2xy_discrete(14, 3, Variant::Liu4), (2, 3));
        assert_eq!(h2xy_discrete(15, 3, Variant::Liu4), (3, 3));
        assert_eq!(xy2h_discrete(4, 3, 3, Variant::Liu4), 48);
        assert_eq!(xy2h_discrete(5, 3, 3, Variant::Liu4), 51);
        assert_eq!(xy2h_discrete(2, 4, 3, Variant::Liu4), 19);
        assert_eq!(xy2h_discrete(3, 4, 3, Variant::Liu4), 16);
        assert_eq!(h2xy_discrete(47, 3, Variant::Liu4), (4, 4));
        assert_eq!(h2xy_discrete(44, 3, Variant::Liu4), (5, 4));
        assert_eq!(xy2h_discrete(2, 5, 3, Variant::Liu4), 18);
        assert_eq!(xy2h_discrete(3, 5, 3, Variant::Liu4), 17);
        assert_eq!(h2xy_discrete(46, 3, Variant::Liu4), (4, 5));
        assert_eq!(h2xy_discrete(45, 3, Variant::Liu4), (5, 5));
    }

    #[test]
    fn map_continuous_hilbert() {
        // Max acceptable error, based on a couple of naive tests of my own
        // A proper analysis of the precision loss in the `continuous` methods is still required
        let err32 = 0.000163;
        let err64 = 0.00000000722;
        
        let (tx, ty, th) = (0.33333334, 0.33333334, 0.13333334);
        let h = xy2h_continuous_f32(tx, ty, Variant::Hilbert);
        let (x, y) = h2xy_continuous_f32(th, Variant::Hilbert);
        assert_approx_eq!(h, th, err32);
        assert_approx_eq!(x, tx, err32);
        assert_approx_eq!(y, ty, err32);

        let (tx, ty, th) = (0.33333334, 0.6666667, 0.46666667);
        let h = xy2h_continuous_f32(tx, ty, Variant::Hilbert);
        let (x, y) = h2xy_continuous_f32(th, Variant::Hilbert);
        assert_approx_eq!(h, th, err32);
        assert_approx_eq!(x, tx, err32);
        assert_approx_eq!(y, ty, err32);

        let (tx, ty, th) = (0.6666666666666666, 0.3333333333333333, 0.8666666666666667);
        let h = xy2h_continuous_f64(tx, ty, Variant::Hilbert);
        let (x, y) = h2xy_continuous_f64(th, Variant::Hilbert);
        assert_approx_eq!(h, th, err64);
        assert_approx_eq!(x, tx, err64);
        assert_approx_eq!(y, ty, err64);

        let (tx, ty, th) = (0.6666666666666666, 0.6666666666666666, 0.5333333333333333);
        let h = xy2h_continuous_f64(tx, ty, Variant::Hilbert);
        let (x, y) = h2xy_continuous_f64(th, Variant::Hilbert);
        assert_approx_eq!(h, th, err64);
        assert_approx_eq!(x, tx, err64);
        assert_approx_eq!(y, ty, err64);
    }

    #[test]
    fn map_continuous_moore() {
        // Max acceptable error, based on a couple of naive tests of my own
        // A proper analysis of the precision loss in the `continuous` methods is still required
        let err32 = 0.000163;
        let err64 = 0.00000000722;
        
        let (tx, ty, th) = (0.33333334, 0.33333334, 0.21666667);
        let h = xy2h_continuous_f32(tx, ty, Variant::Moore);
        let (x, y) = h2xy_continuous_f32(th, Variant::Moore);
        assert_approx_eq!(h, th, err32);
        assert_approx_eq!(x, tx, err32);
        assert_approx_eq!(y, ty, err32);

        let (tx, ty, th) = (0.33333334, 0.6666667, 0.28333333);
        let h = xy2h_continuous_f32(tx, ty, Variant::Moore);
        let (x, y) = h2xy_continuous_f32(th, Variant::Moore);
        assert_approx_eq!(h, th, err32);
        assert_approx_eq!(x, tx, err32);
        assert_approx_eq!(y, ty, err32);

        let (tx, ty, th) = (0.6666666666666666, 0.3333333333333333, 0.7833333333333333);
        let h = xy2h_continuous_f64(tx, ty, Variant::Moore);
        let (x, y) = h2xy_continuous_f64(th, Variant::Moore);
        assert_approx_eq!(h, th, err64);
        assert_approx_eq!(x, tx, err64);
        assert_approx_eq!(y, ty, err64);

        let (tx, ty, th) = (0.6666666666666666, 0.6666666666666666, 0.7166666666666667);
        let h = xy2h_continuous_f64(tx, ty, Variant::Moore);
        let (x, y) = h2xy_continuous_f64(th, Variant::Moore);
        assert_approx_eq!(h, th, err64);
        assert_approx_eq!(x, tx, err64);
        assert_approx_eq!(y, ty, err64);
    }

    #[test]
    fn map_continuous_liu1() {
        // Max acceptable error, based on a couple of naive tests of my own
        // A proper analysis of the precision loss in the `continuous` methods is still required
        let err32 = 0.000163;
        let err64 = 0.00000000722;
        
        let (tx, ty, th) = (0.22222222, 0.22222222, 0.12564103);
        let h = xy2h_continuous_f32(tx, ty, Variant::Liu1);
        let (x, y) = h2xy_continuous_f32(th, Variant::Liu1);
        assert_approx_eq!(h, th, err32);
        assert_approx_eq!(x, tx, err32);
        assert_approx_eq!(y, ty, err32);

        let (tx, ty, th) = (0.22222222, 0.44444445, 0.22758524);
        let h = xy2h_continuous_f32(tx, ty, Variant::Liu1);
        let (x, y) = h2xy_continuous_f32(th, Variant::Liu1);
        assert_approx_eq!(h, th, err32);
        assert_approx_eq!(x, tx, err32);
        assert_approx_eq!(y, ty, err32);

        let (tx, ty, th) = (0.4444444444444444, 0.2222222222222222, 0.06545535120101877);
        let h = xy2h_continuous_f64(tx, ty, Variant::Liu1);
        let (x, y) = h2xy_continuous_f64(th, Variant::Liu1);
        assert_approx_eq!(h, th, err64);
        assert_approx_eq!(x, tx, err64);
        assert_approx_eq!(y, ty, err64);

        let (tx, ty, th) = (0.4444444444444444, 0.4444444444444444, 0.002564102564102564);
        let h = xy2h_continuous_f64(tx, ty, Variant::Liu1);
        let (x, y) = h2xy_continuous_f64(th, Variant::Liu1);
        assert_approx_eq!(h, th, err64);
        assert_approx_eq!(x, tx, err64);
        assert_approx_eq!(y, ty, err64);
    }

    #[test]
    fn map_continuous_liu2() {
        // Max acceptable error, based on a couple of naive tests of my own
        // A proper analysis of the precision loss in the `continuous` methods is still required
        let err32 = 0.000163;
        let err64 = 0.00000000722;
        
        let (tx, ty, th) = (0.22222222, 0.22222222, 0.124358974);
        let h = xy2h_continuous_f32(tx, ty, Variant::Liu2);
        let (x, y) = h2xy_continuous_f32(th, Variant::Liu2);
        assert_approx_eq!(h, th, err32);
        assert_approx_eq!(x, tx, err32);
        assert_approx_eq!(y, ty, err32);

        let (tx, ty, th) = (0.22222222, 0.44444445, 0.02241476);
        let h = xy2h_continuous_f32(tx, ty, Variant::Liu2);
        let (x, y) = h2xy_continuous_f32(th, Variant::Liu2);
        assert_approx_eq!(h, th, err32);
        assert_approx_eq!(x, tx, err32);
        assert_approx_eq!(y, ty, err32);

        let (tx, ty, th) = (0.4444444444444444, 0.2222222222222222, 0.18454464879898125);
        let h = xy2h_continuous_f64(tx, ty, Variant::Liu2);
        let (x, y) = h2xy_continuous_f64(th, Variant::Liu2);
        assert_approx_eq!(h, th, err64);
        assert_approx_eq!(x, tx, err64);
        assert_approx_eq!(y, ty, err64);

        let (tx, ty, th) = (0.4444444444444444, 0.4444444444444444, 0.24743589743589745);
        let h = xy2h_continuous_f64(tx, ty, Variant::Liu2);
        let (x, y) = h2xy_continuous_f64(th, Variant::Liu2);
        assert_approx_eq!(h, th, err64);
        assert_approx_eq!(x, tx, err64);
        assert_approx_eq!(y, ty, err64);
    }

    #[test]
    fn map_continuous_liu3() {
        // Max acceptable error, based on a couple of naive tests of my own
        // A proper analysis of the precision loss in the `continuous` methods is still required
        let err32 = 0.000163;
        let err64 = 0.00000000722;
        
        let (tx, ty, th) = (0.22222222, 0.22222222, 0.041025642);
        let h = xy2h_continuous_f32(tx, ty, Variant::Liu3);
        let (x, y) = h2xy_continuous_f32(th, Variant::Liu3);
        assert_approx_eq!(h, th, err32);
        assert_approx_eq!(x, tx, err32);
        assert_approx_eq!(y, ty, err32);

        let (tx, ty, th) = (0.22222222, 0.44444445, 0.18914248);
        let h = xy2h_continuous_f32(tx, ty, Variant::Liu3);
        let (x, y) = h2xy_continuous_f32(th, Variant::Liu3);
        assert_approx_eq!(h, th, err32);
        assert_approx_eq!(x, tx, err32);
        assert_approx_eq!(y, ty, err32);

        let (tx, ty, th) = (0.4444444444444444, 0.2222222222222222, 0.10511851937285181);
        let h = xy2h_continuous_f64(tx, ty, Variant::Liu3);
        let (x, y) = h2xy_continuous_f64(th, Variant::Liu3);
        assert_approx_eq!(h, th, err64);
        assert_approx_eq!(x, tx, err64);
        assert_approx_eq!(y, ty, err64);

        let (tx, ty, th) = (0.4444444444444444, 0.4444444444444444, 0.1641025641025641);
        let h = xy2h_continuous_f64(tx, ty, Variant::Liu3);
        let (x, y) = h2xy_continuous_f64(th, Variant::Liu3);
        assert_approx_eq!(h, th, err64);
        assert_approx_eq!(x, tx, err64);
        assert_approx_eq!(y, ty, err64);
    }

    #[test]
    fn map_continuous_liu4() {
        // Max acceptable error, based on a couple of naive tests of my own
        // A proper analysis of the precision loss in the `continuous` methods is still required
        let err32 = 0.000163;
        let err64 = 0.00000000722;
        
        let (tx, ty, th) = (0.22222222, 0.22222222, 0.124358974);
        let h = xy2h_continuous_f32(tx, ty, Variant::Liu4);
        let (x, y) = h2xy_continuous_f32(th, Variant::Liu4);
        assert_approx_eq!(h, th, err32);
        assert_approx_eq!(x, tx, err32);
        assert_approx_eq!(y, ty, err32);

        let (tx, ty, th) = (0.22222222, 0.44444445, 0.02241476);
        let h = xy2h_continuous_f32(tx, ty, Variant::Liu4);
        let (x, y) = h2xy_continuous_f32(th, Variant::Liu4);
        assert_approx_eq!(h, th, err32);
        assert_approx_eq!(x, tx, err32);
        assert_approx_eq!(y, ty, err32);

        let (tx, ty, th) = (0.4444444444444444, 0.2222222222222222, 0.18454464879898125);
        let h = xy2h_continuous_f64(tx, ty, Variant::Liu4);
        let (x, y) = h2xy_continuous_f64(th, Variant::Liu4);
        assert_approx_eq!(h, th, err64);
        assert_approx_eq!(x, tx, err64);
        assert_approx_eq!(y, ty, err64);

        let (tx, ty, th) = (0.4444444444444444, 0.4444444444444444, 0.24743589743589745);
        let h = xy2h_continuous_f64(tx, ty, Variant::Liu4);
        let (x, y) = h2xy_continuous_f64(th, Variant::Liu4);
        assert_approx_eq!(h, th, err64);
        assert_approx_eq!(x, tx, err64);
        assert_approx_eq!(y, ty, err64);
    }

    #[test]
    fn extract_binary_fractional_f32() {
        // Test conversion for every single decimal place able to be represented.
        // The conversion is not completely precise, but it's imprecision should match that of the f32 multiplication.
        let mut frac = 0b1usize.reverse_bits();
        let mut dec = 0.5f32;
        while frac != 0 {
            assert_eq!(frac, f32_to_binfrac(dec));
            assert_approx_eq!(dec, binfrac_to_f32(frac));
            frac >>= 1;
            dec *= 0.5;
        }
    }

    #[test]
    fn extract_binary_fractional_f64() {
        // Test conversion for every single decimal place able to be represented.
        // The conversion is not completely precise, but it's imprecision should match that of the f64 multiplication.
        let mut frac = 0b1usize.reverse_bits();
        let mut dec = 0.5f64;
        while frac != 0 {
            assert_eq!(frac, f64_to_binfrac(dec));
            assert_approx_eq!(dec, binfrac_to_f64(frac));
            frac >>= 1;
            dec *= 0.5;
        }
    }
}
