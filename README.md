# omni-wave

[![](https://img.shields.io/crates/v/omni-wave)](https://crates.io/crates/omni-wave)
[![](https://img.shields.io/crates/d/omni-wave)](https://crates.io/crates/omni-wave)
[![](https://img.shields.io/crates/l/omni-wave)](#)
[![](https://img.shields.io/docsrs/omni-wave)](https://docs.rs/omni-wave)
[![](https://img.shields.io/github/stars/eternal-io/omni-wave?style=social)](https://github.com/eternal-io/omni-wave)

Easy to use DWT (Discrete Wavelet Transform) library, no need to worry about padding, and a variety of wavelets are available.

***CAUTION: My knowledge can't vouch for it being "correct", but it does "concentrate energy" and "reconstruct perfectly".***

``` rust
use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, Axis};
use omni_wave::*;

let wavelet = wavelet::BIOR_3_1;
let love = Array2::from_shape_vec((8, 8),
vec![0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0.,99., 0., 0.,99., 0., 0.,
     0.,99.,99.,99.,99.,99.,99., 0.,
     0.,99.,99.,99.,99.,99.,99., 0.,
     0.,99.,99.,99.,99.,99.,99., 0.,
     0., 0.,99.,99.,99.,99., 0., 0.,
     0., 0., 0.,99.,99., 0., 0., 0.,
     0., 0., 0., 0., 0., 0., 0., 0.,]).unwrap();

let mut signal = love.clone();
let mut buffer = Array1::zeros(signal.len_of(Axis(0)) + wavelet.window_size() - 2); // The minimum length of a buffer

completely_decompose_2d(signal.view_mut(), buffer.view_mut(), wavelet);
completely_reconstruct_2d(signal.view_mut(), buffer.view_mut(), wavelet);

love.into_iter()
    .zip(signal)
    .for_each(|(a, b)| assert_abs_diff_eq!(a, b, epsilon = 0.0001));
```

## Features

- `f64` - The default primitive type used for calculations is `f32`. Enable this feature to switch to `f64`.

## Knowledges

#### Signal

The data need to transform. **The length should be even.**
Failure to meet the length requirement may not result a panic, but the behavior of functions will be undefined.

The left half of the input will be considered as *Approx*, while the right half will be considered as *Detail*.

#### Padding

The extension of a signal when processing. Our filling method named `periodic` (in [PyWavelets](https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html#naming-conventions)), `ppd` (in Matlab) or `wrap` (in numpy.pad).

``` plaintext
[ A.B.C.D.E.F.G.H ] a.b.c.d ...
  ↑^^^^^^^^^^^^^^   ↑^^^^^^
  Original signal   Padding: automatically fill & detach.
```

Originally, I planned to provide more extension mode, but found that the first coefficient of wavelets such as `bior2.2` is zero... Wouldn't the information be lost in this way? ? So currently only `periodic` mode is provided. *Let me know if you have a better suggestion.*

#### Buffer

Temporary buffer for calculations. For performance, it is strongly recommended that it be contiguous in memory.

#### `window_size`

The number of wavelet coefficients.
