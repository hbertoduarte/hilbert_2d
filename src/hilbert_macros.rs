macro_rules! hilbert_impl {
    ( $n:literal, $SelfT:ident, $const_bits:ident ) => {
        #[doc = concat!("Discrete functions for the ", $n, " unsigned integer type.")]
        pub mod $SelfT {
            use super::{next_lut_index, next_lut_index_variant, Variant};

            /// Number of bits in `
            #[doc = stringify!($SelfT)]
            /// ` for the current platform
            pub(crate) const $const_bits: u32 = <$SelfT>::MAX.count_ones();

            /// Highest order of the Hilbert curve that can be calculated for `
            #[doc = stringify!($SelfT)]
            /// ` in the current platform
            pub(crate) const ORDER_MAX: u32 = $const_bits / 2;

            /// Lookup tables for the 1D->2D conversions.
            const LUTS_H2XY: [[($SelfT, $SelfT); 4]; 8] = [
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
            const LUTS_YX2H: [[[$SelfT; 2]; 2]; 8] = [
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

            /// Maps from a 1D index to a 2D coordinate, using a discrete approximation of the
            /// Hilbert curve. Recommended for images and matrices.
            ///
            /// Given `h`, this method calculates the `(x, y)` coordinates for that index, in the
            /// Hilbert curve approximation of `order`.
            ///
            /// The value of `h` must be in the range `0 <= h < 2^(2 * order)`. The coordinates
            /// returned will be in the range `0 <= x/y < 2^order`.
            ///
            /// With `b` being the number of bits in a `
            #[doc = stringify!($SelfT)]
            /// ` for the target platform, the highest approximation order achievable
            /// by this method is `b / 2`. Therefore, the parameter `order` must be in the range
            /// `1 <= order <= b / 2`.
            ///
            /// The pattern of the Hilbert curve to be constructed must be indicated by the
            /// `variant` parameter. See [`Variant`] for more details.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```rust
            #[doc = concat!("use hilbert_2d::{Variant, ", stringify!($SelfT), "::h2xy_discrete};")]
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
            /// [`Variant`]: ../enum.Variant.html
            ///
            pub const fn h2xy_discrete(
                h: $SelfT,
                order: $SelfT,
                variant: Variant,
            ) -> ($SelfT, $SelfT) {
                // Records how many >>'s are required to isolate the quadrant crumbs for each subsequent order, starting at 1
                let mut steps = 2 * (order - 1);
                if steps as u32 >= $const_bits {
                    steps = $const_bits as $SelfT - 2;
                }
                // Extract the crumb for the quadrant of order 1 from `curve_p`
                let u_h = h as usize;
                let mut cur_quadrant = (u_h >> steps) & 0b11;
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
                    cur_quadrant = (u_h >> steps) & 0b11;

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

            /// Maps from a 2D coordinate to an 1D index, using a discrete approximation of the
            /// Hilbert curve. Recommended for images and matrices.
            ///
            /// Given `x` and `y`, this method calculates the `h` index for that coordinate, in the
            /// Hilbert curve approximation of `order`.
            ///
            /// The value of `x` and `y` must be in the range `0 <= x/y < 2^order`. The index
            /// returned will be in the range `0 <= h < 2^(2 * order)`.
            ///
            /// With `b` being the number of bits in a `
            #[doc = stringify!($SelfT)]
            /// ` for the target platform, the highest approximation order achievable
            /// by this method is `b / 2`. Therefore, the parameter `order` must be in the range
            /// `1 <= order <= b / 2`.
            ///
            /// The pattern of the Hilbert curve to be constructed must be indicated by the
            /// `variant` parameter. See [`Variant`] for more details.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```rust
            #[doc = concat!("use hilbert_2d::{Variant, ", stringify!($SelfT), "::xy2h_discrete};")]
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
            /// [`Variant`]: ../enum.Variant.html
            ///
            pub const fn xy2h_discrete(
                x: $SelfT,
                y: $SelfT,
                order: $SelfT,
                variant: Variant,
            ) -> $SelfT {
                // Records how many >>'s are required to isolate the quadrant bits for each subsequent order, starting at 1
                let mut steps = order - 1;
                if steps as u32 >= ORDER_MAX {
                    steps = ORDER_MAX as $SelfT - 1;
                }

                // Extract the bits for the square region from the `square_x` and `square_x` coordinates
                let u_x = x as usize;
                let u_y = y as usize;
                let mut q_x = (u_x >> steps) & 0b1;
                let mut q_y = (u_y >> steps) & 0b1;

                // Retrieve the quadrant of order 1 equivalent to that square region
                let mut cur_quadrant = LUTS_YX2H[0b000][q_y][q_x];
                let mut curve_p = cur_quadrant;

                // Stop here if the given order is 1
                if steps == 0 {
                    return curve_p;
                }

                // Depending on the curve variant, the method of choosing the LUT for the `order 1 -> order 2` transition changes
                let mut lut_index = next_lut_index_variant(0b000, cur_quadrant as usize, variant);

                loop {
                    // Extract the bits for the next square region from the `square_x` and `square_x` coordinates
                    steps -= 1;
                    q_x = (u_x >> steps) & 0b1;
                    q_y = (u_y >> steps) & 0b1;

                    // Retrieve the quadrant equivalent to that square region, using the current LUT
                    cur_quadrant = LUTS_YX2H[lut_index][q_y][q_x];
                    curve_p = (curve_p << 2) | cur_quadrant;

                    if steps > 0 {
                        // From `order 2 -> order 3` and beyond, the same method of choosing the next LUT is used
                        lut_index = next_lut_index(lut_index, cur_quadrant as usize);
                    } else {
                        break;
                    }
                }

                curve_p
            }

            #[cfg(test)]
            mod tests {
                use super::*;

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
            }
        }
    };
}
