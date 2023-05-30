# omni-wave

[![](https://img.shields.io/crates/v/omni-wave)](https://crates.io/crates/omni-wave)
[![](https://img.shields.io/crates/d/omni-wave)](https://crates.io/crates/omni-wave)
[![](https://img.shields.io/crates/l/omni-wave)](#)
[![](https://img.shields.io/docsrs/omni-wave)](https://docs.rs/omni-wave)
[![](https://img.shields.io/github/stars/eternal-io/omni-wave?style=social)](https://github.com/eternal-io/omni-wave)

Easy to use Discrete Wavelet Transform library, no need to worry about padding, and a variety of wavelets are available.

``` rust
# use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, Axis};
use omni_wave::{completely_decompose_2d, completely_reconstruct_2d, wavelet};

let wave = wavelet::BIOR_3_1;
let raw = Array2::<f32>::from_shape_vec((8, 8),
vec![0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0.,99.,99., 0., 0., 0.,
     0., 0.,99.,99.,99.,99., 0., 0.,
     0.,99.,99.,99.,99.,99.,99., 0.,
     0.,99.,99.,99.,99.,99.,99., 0.,
     0., 0.,99.,99.,99.,99., 0., 0.,
     0., 0., 0.,99.,99., 0., 0., 0.,
     0., 0., 0., 0., 0., 0., 0., 0.,]).unwrap();

let mut signal_2d = raw.clone();
let mut buffer = Array1::<f32>::zeros(signal_2d.len_of(Axis(0)) + wave.window_size() - 2);

completely_decompose_2d(signal_2d.view_mut(), buffer.view_mut(), wave);
completely_reconstruct_2d(signal_2d.view_mut(), buffer.view_mut(), wave);

raw.into_iter()
    .zip(signal_2d)
    .for_each(|(a, b)| assert_abs_diff_eq!(a, b, epsilon = 0.0001));
```

## Knowledges

#### Signal

The data need to transform. **The length should be even.**
Failure to meet the length requirement may not result a panic, but the behavior of functions will be undefined.

The left half of the input will be considered as *Approx*, while the right half will be considered as *Detail*.

#### Padding

Our filling method named `periodic` (in [PyWavelets](https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html#naming-conventions)),
`ppd` (in Matlab) or `wrap` (in numpy.pad).

``` plaintext
[ A.B.C.D.E.F.G.H ] a.b.c.d ...
  ↑^^^^^^^^^^^^^^   ↑^^^^^^
  Original signal   Padding: automatically fill & detach!
```

#### Window

The number of wavelet coefficients.
