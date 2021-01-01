use std::f64::consts::PI as PIf64;
use std::iter::FromIterator;
use std::ops::Div;

use ndarray::{prelude::*, ScalarOperand};
use num_traits::{FromPrimitive, ToPrimitive};
use rustfft::num_traits::Float;

pub enum WindowType {
    Hann,
    BoxCar,
}

#[inline]
pub fn calc_normalized_win<T>(
    win_type: WindowType,
    size: usize,
    norm_factor: impl ToPrimitive,
) -> Array1<T>
where
    T: Float + FromPrimitive + Div + ScalarOperand,
{
    match win_type {
        WindowType::Hann => hann(size, false) / T::from(norm_factor).unwrap(),
        WindowType::BoxCar => Array1::from_elem(size, T::one() / T::from(norm_factor).unwrap()),
    }
}

#[inline]
pub fn hann<T: Float + FromPrimitive>(size: usize, symmetric: bool) -> Array1<T> {
    cosine_window(
        T::from(0.5).unwrap(),
        T::from(0.5).unwrap(),
        T::zero(),
        T::zero(),
        size,
        symmetric,
    )
}

fn cosine_window<T>(a: T, b: T, c: T, d: T, size: usize, symmetric: bool) -> Array1<T>
where
    T: Float + FromPrimitive,
{
    assert!(size > 1);
    let pi = T::from_f64(PIf64).unwrap();
    let size2 = if symmetric { size } else { size + 1 };
    let cos_fn = |i| {
        let x = pi * T::from_usize(i).unwrap() / T::from_usize(size2 - 1).unwrap();
        let b_ = b * (T::from_u8(2).unwrap() * x).cos();
        let c_ = c * (T::from_u8(4).unwrap() * x).cos();
        let d_ = d * (T::from_u8(6).unwrap() * x).cos();
        (a - b_) + (c_ - d_)
    };
    Array::from_iter((0..size2).map(cos_fn).take(size))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hann_window_works() {
        assert_eq!(hann::<f32>(4, false), arr1(&[0f32, 0.5, 1., 0.5]));
    }
}
