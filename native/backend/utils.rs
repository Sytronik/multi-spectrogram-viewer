use std::mem::MaybeUninit;

use ndarray::{prelude::*, Data, RemoveAxis, Slice, Zip};
use rustfft::{
    num_complex::Complex,
    num_traits::{
        identities::{One, Zero},
        Float, Num,
    },
    FftNum,
};

use super::realfft::RealFFT;

pub fn calc_proper_n_fft(win_length: usize) -> usize {
    2usize.pow((win_length as f32).log2().ceil() as u32)
}

pub trait Impulse {
    fn impulse(size: usize, location: usize) -> Self;
}

impl<A> Impulse for Array1<A>
where
    A: Clone + Zero + One,
{
    fn impulse(size: usize, location: usize) -> Self {
        let mut new = Array1::<A>::zeros((size,));
        new[location] = A::one();
        new
    }
}

pub fn rfft<A, S, D>(input: ArrayBase<S, D>) -> Array1<Complex<A>>
where
    A: FftNum + Float,
    S: Data<Elem = A>,
    D: Dimension,
{
    let n_fft = input.shape()[0];
    let mut r2c = RealFFT::<A>::new(n_fft).unwrap();
    let mut output = Array1::<Complex<A>>::zeros(n_fft / 2 + 1);
    r2c.process(
        input.into_owned().as_slice_mut().unwrap(),
        output.as_slice_mut().unwrap(),
    )
    .unwrap();

    output
}

pub enum PadMode<T> {
    Constant(T),
    Reflect,
}

pub fn pad<A, D>(
    array: ArrayView<A, D>,
    (n_pad_left, n_pad_right): (usize, usize),
    axis: Axis,
    mode: PadMode<A>,
) -> Array<A, D>
where
    A: Copy + Num,
    D: Dimension + RemoveAxis,
{
    let mut shape = array.raw_dim();
    shape[axis.index()] += n_pad_left + n_pad_right;
    let mut result = Array::maybe_uninit(shape);

    let s_result_main = if n_pad_right > 0 {
        Slice::from(n_pad_left as isize..-(n_pad_right as isize))
    } else {
        Slice::from(n_pad_left as isize..)
    };
    Zip::from(&array).apply_assign_into(result.slice_axis_mut(axis, s_result_main), A::clone);
    match mode {
        PadMode::Constant(constant) => {
            result
                .slice_axis_mut(axis, Slice::from(..n_pad_left))
                .mapv_inplace(|_| MaybeUninit::new(constant));
            if n_pad_right > 0 {
                result
                    .slice_axis_mut(axis, Slice::from(-(n_pad_right as isize)..))
                    .mapv_inplace(|_| MaybeUninit::new(constant));
            }
        }
        PadMode::Reflect => {
            let pad_left = array
                .axis_iter(axis)
                .skip(1)
                .chain(array.axis_iter(axis).rev().skip(1))
                .cycle()
                .take(n_pad_left);
            result
                .axis_iter_mut(axis)
                .take(n_pad_left)
                .rev()
                .zip(pad_left)
                .for_each(|(y, x)| Zip::from(x).apply_assign_into(y, A::clone));

            if n_pad_right > 0 {
                let pad_right = array
                    .axis_iter(axis)
                    .rev()
                    .skip(1)
                    .chain(array.axis_iter(axis).skip(1))
                    .cycle()
                    .take(n_pad_right);
                result
                    .axis_iter_mut(axis)
                    .rev()
                    .take(n_pad_right)
                    .rev()
                    .zip(pad_right)
                    .for_each(|(y, x)| Zip::from(x).apply_assign_into(y, A::clone));
            }
        }
    }
    unsafe { result.assume_init() }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::{arr1, arr2, Array1};
    use rustfft::num_complex::Complex;

    #[test]
    fn rfft_wrapper_works() {
        assert_eq!(
            rfft(Array1::<f32>::impulse(4, 0)),
            arr1(&[Complex::<f32>::new(1., 0.); 3])
        );
    }

    #[test]
    fn pad_works() {
        assert_eq!(
            pad(
                arr2(&[[1, 2, 3]]).view(),
                (1, 2),
                Axis(0),
                PadMode::Constant(10)
            ),
            arr2(&[[10, 10, 10], [1, 2, 3], [10, 10, 10], [10, 10, 10]])
        );
        assert_eq!(
            pad(arr2(&[[1, 2, 3]]).view(), (3, 4), Axis(1), PadMode::Reflect),
            arr2(&[[2, 3, 2, 1, 2, 3, 2, 1, 2, 3]])
        );
    }
}