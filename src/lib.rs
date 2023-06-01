#![doc = include_str!("../README.md")]

use ndarray::{s, ArrayView1, ArrayViewMut1, ArrayViewMut2, Axis};
use std::ops::{AddAssign, DivAssign, MulAssign};

#[cfg(not(feature = "f64"))]
type FLTYPE = f32;
#[cfg(feature = "f64")]
type FLTYPE = f64;

const TWO: usize = 2;
const NORM_COEFF: FLTYPE = std::f64::consts::FRAC_1_SQRT_2 as FLTYPE;

pub mod wavelet {
    //! Thanks to [Wavelet Browser](https://wavelets.pybytes.com/)!
    //!
    //! > *但你们为什么要把分解滤波器给反过来？？*
    //! >
    //! > 所以，这里的所有小波也都是反过来的。懒得改了，反正不影响使用就对了。
    //!
    //! The number of wavelets are currently limited, and none of them has more than **12** coefficients.
    //!
    //! *(Because I'm lazy :)*
    use super::*;

    mod bior;
    mod coif;
    mod db;
    mod sym;

    pub use bior::*;
    pub use coif::*;
    pub use db::*;
    pub use sym::*;
    pub const HAAR: Wavelet = Wavelet {
        decomp_low: &[0.7071067811865476, 0.7071067811865476],
        decomp_high: &[-0.7071067811865476, 0.7071067811865476],
        recons_low: &[0.7071067811865476, 0.7071067811865476],
        recons_high: &[-0.7071067811865476, 0.7071067811865476],
    };
}

/// Check [`wavelet`] to see all the wavelets provided.
///
/// Each filter of a single wavelet must be of equal length!
///
/// ``` plaintext
///  |<------------->|- window_size (N.coeffs)
///  ↓               ↓
///  (A B C;D E;F G H)
///        ↑   ↑     ↑
///        |   |<--->|- half_padding_length
///        |   |
///        |<->|------- *Sliding*!
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Wavelet {
    pub decomp_low: &'static [FLTYPE],
    pub decomp_high: &'static [FLTYPE],
    pub recons_low: &'static [FLTYPE],
    pub recons_high: &'static [FLTYPE],
}
impl Wavelet {
    #[inline]
    pub const fn window_size(&self) -> usize {
        self.decomp_low.len()
    }
    #[inline]
    pub const fn half_padding_length(&self) -> usize {
        (self.decomp_low.len() - 2) >> 1
    }
}

/// Forward wavelet transform, 1D, only once, inplace.
///
/// ``` plaintext
/// [      Signal      ] => [ Approx ][ Detail ]
/// ```
///
/// # Hard requirements
///
/// - `buffer_size` >= `signal_size + window_size - TWO`
#[doc(alias = "dwt")]
pub fn decompose(
    mut signal: ArrayViewMut1<FLTYPE>,
    mut buffer: ArrayViewMut1<FLTYPE>,
    wavelet: Wavelet,
) {
    let signal_size = signal.len();
    let window_size = wavelet.window_size();

    let expected_buffer_size = signal_size + window_size - TWO; // 这个公式不是很直观……但它没错就对了

    /* 填充 */

    let mut step_buffer = 0;
    while step_buffer + signal_size < expected_buffer_size {
        buffer
            .slice_mut(s![step_buffer..step_buffer + signal_size])
            .assign(&signal);
        step_buffer += signal_size;
    }
    buffer
        .slice_mut(s![step_buffer..expected_buffer_size])
        .into_iter()
        .zip(&signal) // 用迭代器不是比列公式方便的多？
        .for_each(|(buf, sig)| *buf = *sig);

    /* 卷积 */

    let (low_pass, high_pass) = (
        ArrayView1::from_shape(window_size, wavelet.decomp_low).unwrap(),
        ArrayView1::from_shape(window_size, wavelet.decomp_high).unwrap(),
    );

    let half = signal.len() / TWO;
    for (step_signal, step_buffer) in (0..signal_size).step_by(TWO).enumerate() {
        let slice_signal = buffer.slice(s![step_buffer..step_buffer + window_size]);

        signal[step_signal] = slice_signal.dot(&low_pass); // 就地操作
        signal[half + step_signal] = slice_signal.dot(&high_pass);
    }

    /* 归一化 */

    signal.mul_assign(NORM_COEFF);
}

/// Inverse wavelet transform, 1D, only once, inplace.
///
/// ``` plaintext
/// [ Approx ][ Detail ] => [ Original signal ]
/// ```
///
/// # Hard requirements
///
/// - `buffer_size` >= `signal_size + window_size - TWO`
#[doc(alias = "idwt")]
pub fn reconstruct(
    mut signal: ArrayViewMut1<FLTYPE>,
    mut buffer: ArrayViewMut1<FLTYPE>,
    wavelet: Wavelet,
) {
    let signal_size = signal.len();
    let window_size = wavelet.window_size();

    let expected_buffer_size = signal_size + window_size - TWO;

    /* 反归一化 */

    let half = signal_size / TWO;
    signal.div_assign(NORM_COEFF);

    /* 卷积 */

    buffer.slice_mut(s![..expected_buffer_size]).fill(0.);
    let (low_pass, high_pass) = (
        ArrayView1::from_shape(window_size, wavelet.recons_low).unwrap(),
        ArrayView1::from_shape(window_size, wavelet.recons_high).unwrap(),
    );

    for (mut step_buffer, approx_n) in signal.slice(s![..half]).into_iter().enumerate() {
        step_buffer *= 2;
        buffer
            .slice_mut(s![step_buffer..step_buffer + window_size])
            .add_assign(&(&low_pass * *approx_n));
    }
    for (mut step_buffer, detail_n) in signal.slice(s![half..]).into_iter().enumerate() {
        step_buffer *= 2;
        buffer
            .slice_mut(s![step_buffer..step_buffer + window_size])
            .add_assign(&(&high_pass * *detail_n));
    }

    /* “折叠”！！*/

    signal.fill(0.);

    let mut step_signal = 0;
    while step_signal + signal_size < expected_buffer_size {
        signal.add_assign(&buffer.slice(s![step_signal..step_signal + signal_size]));
        step_signal += signal_size;
    }
    signal
        .iter_mut()
        .zip(buffer.slice(s![step_signal..expected_buffer_size]))
        .for_each(|(sig, buf)| *sig += *buf);
}

/// Forward wavelet transform, 1D, completely, inplace.
///
/// ``` plaintext
/// [            Signal            ] => [A₂][D₂][  D₁  ][      D₀      ]
/// ```
///
/// # Hard requirements
///
/// - `signal_size` should be exactly a power of 2, otherwise it will panic in debug builds.
/// - `buffer_size` >= `signal_size + window_size - TWO`
#[doc(alias = "dwt")]
pub fn completely_decompose(
    mut signal: ArrayViewMut1<FLTYPE>,
    mut buffer: ArrayViewMut1<FLTYPE>,
    wavelet: Wavelet,
) {
    let mut signal_size = signal.len();
    debug_assert!(signal_size.is_power_of_two());

    while signal_size >= TWO {
        decompose(
            signal.slice_mut(s![..signal_size]),
            buffer.view_mut(),
            wavelet,
        );
        signal_size >>= 1;
    }
}

/// Inverse wavelet transform, 1D, completely, inplace.
///
/// ``` plaintext
/// [A₂][D₂][  D₁  ][      D₀      ] => [        Original signal        ]
/// ```
///
/// # Hard requirements
///
/// - `signal_size` should be exactly a power of 2, otherwise it will panic in debug builds.
/// - `buffer_size` >= `signal_size + window_size - TWO`
#[doc(alias = "idwt")]
pub fn completely_reconstruct(
    mut signal: ArrayViewMut1<FLTYPE>,
    mut buffer: ArrayViewMut1<FLTYPE>,
    wavelet: Wavelet,
) {
    let signal_size = signal.len();
    debug_assert!(signal_size.is_power_of_two());

    let mut stage = TWO;
    while stage <= signal_size {
        reconstruct(signal.slice_mut(s![..stage]), buffer.view_mut(), wavelet);
        stage <<= 1;
    }
}

/// Forward wavelet transform, 2D, only twice, inplace.
///
/// ``` plaintext
/// +-----------+    +-----+-----+
/// |           |    |  A  |  H  |
/// | 2D Signal | => +-----+-----+
/// |           |    |  V  |  D  |
/// +-----------+    +-----+-----+
/// ```
///
/// Horizontal firstly, then vertical.
///
/// # Hard requirements
///
/// - `buffer_size` >= `signal_side_length + window_size - TWO`
#[doc(alias = "dwt2")]
pub fn decompose_2d(
    mut signal_2d: ArrayViewMut2<FLTYPE>,
    mut buffer: ArrayViewMut1<FLTYPE>,
    wavelet: Wavelet,
) {
    signal_2d
        .rows_mut()
        .into_iter()
        .for_each(|row| decompose(row, buffer.view_mut(), wavelet));

    signal_2d
        .columns_mut()
        .into_iter()
        .for_each(|col| decompose(col, buffer.view_mut(), wavelet));
}

/// Inverse wavelet transform, 2D, only twice, inplace.
///
/// ``` plaintext
/// +-----+-----+    +-----------+
/// |  A  |  H  |    |           |
/// +-----+-----+ => | 2D Signal |
/// |  V  |  D  |    |           |
/// +-----+-----+    +-----------+
/// ```
///
/// Vertical firstly, then horizontal.
///
/// # Hard requirements
///
/// - `buffer_size` >= `signal_side_length + window_size - TWO`
#[doc(alias = "idwt2")]
pub fn reconstruct_2d(
    mut signal_2d: ArrayViewMut2<FLTYPE>,
    mut buffer: ArrayViewMut1<FLTYPE>,
    wavelet: Wavelet,
) {
    signal_2d
        .columns_mut()
        .into_iter()
        .for_each(|col| reconstruct(col, buffer.view_mut(), wavelet));

    signal_2d
        .rows_mut()
        .into_iter()
        .for_each(|row| reconstruct(row, buffer.view_mut(), wavelet));
}

/// Forward wavelet transform, 2D, completely, inplace.
///
/// ``` plaintext
/// +-------------------+    +----+----+---------+
/// |                   |    |----| H₁ |         |
/// |                   |    +----+----+    H₀   |
/// |                   |    | V₁ | D₁ |         |
/// |     2D Signal     | => +----+----+---------+
/// |                   |    |         |         |
/// |                   |    |    V₀   |    D₀   |
/// |                   |    |         |         |
/// +-------------------+    +---------+---------+ ...
/// ```
///
/// Horizontal firstly, then vertical.
///
/// # Hard requirements
///
/// - `signal_shape` should be exactly a **square**, otherwise it will panic in debug builds.
/// - `signal_side_length` should be exactly a power of 2, otherwise it will panic in debug builds.
/// - `buffer_size` >= `signal_side_length + window_size - TWO`
#[doc(alias = "dwt2")]
pub fn completely_decompose_2d(
    mut signal_2d: ArrayViewMut2<FLTYPE>,
    mut buffer: ArrayViewMut1<FLTYPE>,
    wavelet: Wavelet,
) {
    let height = signal_2d.len_of(Axis(0));
    let mut width = signal_2d.len_of(Axis(1));
    debug_assert!(width == height);
    debug_assert!(width.is_power_of_two());

    while width >= TWO {
        decompose_2d(
            signal_2d.slice_mut(s![..width, ..width]),
            buffer.view_mut(),
            wavelet,
        );
        width >>= 1;
    }
}

/// Inverse wavelet transform, 2D, completely, inplace.
///
/// ``` plaintext
/// +----+----+---------+    +-------------------+
/// |----| H₁ |         |    |                   |
/// +----+----+    H₀   |    |                   |
/// | V₁ | D₁ |         |    |     Original      |
/// +----+----+---------+ => |                   |
/// |         |         |    |     2D Signal     |
/// |    V₀   |    D₀   |    |                   |
/// |         |         |    |                   |
/// +---------+---------+    +-------------------+
/// ```
///
/// Vertical firstly, then horizontal.
///
/// # Hard requirements
///
/// - `signal_shape` should be exactly a **square**, otherwise it will panic in debug builds.
/// - `signal_side_length` should be exactly a power of 2, otherwise it will panic in debug builds.
/// - `buffer_size` >= `signal_side_length + window_size - TWO`
#[doc(alias = "idwt2")]
pub fn completely_reconstruct_2d(
    mut signal_2d: ArrayViewMut2<FLTYPE>,
    mut buffer: ArrayViewMut1<FLTYPE>,
    wavelet: Wavelet,
) {
    let height = signal_2d.len_of(Axis(0));
    let width = signal_2d.len_of(Axis(1));
    debug_assert!(width == height);
    debug_assert!(width.is_power_of_two());

    let mut stage = TWO;
    while stage <= width {
        reconstruct_2d(
            signal_2d.slice_mut(s![..stage, ..stage]),
            buffer.view_mut(),
            wavelet,
        );
        stage <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{s, Array1, Array2, Axis};

    #[test]
    fn norm_test() {
        // let wavelet = wavelet::RBIO_3_1;
        let wavelet = wavelet::BIOR_3_1;
        let mut signal = Array1::<FLTYPE>::from_vec(vec![0., 0., 0., 0., 255., 255., 255., 255.]);
        let mut buffer = Array1::<FLTYPE>::zeros(signal.len() + wavelet.window_size() - TWO);

        completely_decompose(signal.view_mut(), buffer.view_mut(), wavelet);
        println!("{signal:>5.1}");

        signal.for_each(|v| assert!(*v >= -128. && *v <= 128.));
    }

    #[test]
    fn auto_2d() {
        let wavelet = wavelet::BIOR_3_5;
        #[rustfmt::skip]
        let raw = Array2::from_shape_vec((8, 8), vec![
            52., 55., 61., 66., 70., 61., 64., 73.,
            63., 59., 55., 90., 109.,85., 69., 72.,
            62., 59., 68., 113.,144.,104.,66., 73.,
            63., 58., 71., 122.,154.,106.,70., 69.,
            67., 61., 68., 104.,126.,88., 68., 70.,
            79., 65., 60., 70., 77., 68., 58., 75.,
            85., 71., 64., 59., 55., 61., 65., 83.,
            87., 79., 69., 68., 65., 76., 78., 94.,
        ]).unwrap();
        let mut signal_2d = raw.clone();
        let mut buffer =
            Array1::<FLTYPE>::zeros(signal_2d.len_of(Axis(0)) + wavelet.window_size() - TWO);

        completely_decompose_2d(signal_2d.view_mut(), buffer.view_mut(), wavelet);
        println!("{signal_2d:>5.1}");
        completely_reconstruct_2d(signal_2d.view_mut(), buffer.view_mut(), wavelet);

        raw.into_iter()
            .zip(signal_2d)
            .for_each(|(a, b)| assert_abs_diff_eq!(a, b, epsilon = 0.0001));
    }

    #[test]
    fn manual_2d() {
        let wavelet = wavelet::BIOR_3_1;
        #[rustfmt::skip]
        let raw = Array2::<FLTYPE>::from_shape_vec((6, 10), vec![
            0., 0., 0.,99.,99.,99.,99., 0., 0., 0.,
            0., 0.,99.,99.,99.,99.,99.,99., 0., 0.,
            0.,99.,99.,99.,99.,99.,99.,99.,99., 0.,
            0.,99.,99.,99.,99.,99.,99.,99.,99., 0.,
            0., 0.,99.,99.,99.,99.,99.,99., 0., 0.,
            0., 0., 0.,99.,99.,99.,99., 0., 0., 0.,
        ]).unwrap();
        let mut signal_2d = raw.clone();
        let mut buffer =
            Array1::<FLTYPE>::zeros(signal_2d.len_of(Axis(1)) + wavelet.window_size() - TWO);

        decompose_2d(signal_2d.view_mut(), buffer.view_mut(), wavelet);
        println!("{signal_2d:>5.1}");
        reconstruct_2d(signal_2d.view_mut(), buffer.view_mut(), wavelet);

        raw.into_iter()
            .zip(signal_2d)
            .for_each(|(a, b)| assert_abs_diff_eq!(a, b, epsilon = 0.0001));
    }

    #[test]
    fn auto_1d() {
        let wavelet = wavelet::BIOR_2_2;
        let raw = Array1::<FLTYPE>::from_vec(vec![0., 10., 100., 200., 250., 30., 20., 10.]);
        let mut signal = raw.clone();
        let mut buffer = Array1::<FLTYPE>::zeros(signal.len() + wavelet.window_size() - TWO);

        completely_decompose(signal.view_mut(), buffer.view_mut(), wavelet);
        println!("{signal:>5.1}");
        completely_reconstruct(signal.view_mut(), buffer.view_mut(), wavelet);

        raw.into_iter()
            .zip(signal)
            .for_each(|(a, b)| assert_abs_diff_eq!(a, b, epsilon = 0.0001));
    }

    #[test]
    fn manual_1d() {
        let wavelet = wavelet::HAAR;
        let raw = Array1::<FLTYPE>::from_vec(vec![31., 41., 59., 26., 53., 58., 97., 93.]);
        let mut signal = raw.clone();
        let mut buffer = Array1::<FLTYPE>::zeros(signal.len() + wavelet.window_size() - TWO);

        decompose(signal.slice_mut(s![..8]), buffer.view_mut(), wavelet);
        decompose(signal.slice_mut(s![..4]), buffer.view_mut(), wavelet);
        decompose(signal.slice_mut(s![..2]), buffer.view_mut(), wavelet);
        println!("{signal:>5.1}");
        reconstruct(signal.slice_mut(s![..2]), buffer.view_mut(), wavelet);
        reconstruct(signal.slice_mut(s![..4]), buffer.view_mut(), wavelet);
        reconstruct(signal.slice_mut(s![..8]), buffer.view_mut(), wavelet);

        raw.into_iter()
            .zip(signal)
            .for_each(|(a, b)| assert_abs_diff_eq!(a, b, epsilon = 0.0001));
    }
}
