use ndarray::{prelude::*, ScalarOperand};
use rayon::prelude::*;
use rustfft::{num_complex::Complex, num_traits::Float, FftNum};
use std::ops::*;

use super::mel;
use super::realfft::RealFFT;
use super::utils::{pad, PadMode};
use super::windows::{calc_normalized_win, WindowType};

#[derive(Clone, Copy, Debug)]
pub enum FreqScale {
    Linear,
    Mel,
}

pub fn perform_stft<A>(
    input: ArrayView1<A>,
    win_length: usize,
    hop_length: usize,
    n_fft: usize,
    window: Option<CowArray<A, Ix1>>,
    fft_module: Option<&mut RealFFT<A>>,
    parallel: bool,
) -> Array2<Complex<A>>
where
    A: FftNum + Float + DivAssign + ScalarOperand,
{
    let n_pad_left = (n_fft - win_length) / 2;
    let n_pad_right = (((n_fft - win_length) as f32) / 2.).ceil() as usize;

    let window = if let Some(w) = window {
        assert_eq!(w.len(), win_length);
        w
    } else {
        CowArray::from(calc_normalized_win(WindowType::Hann, win_length, n_fft))
    };

    let to_frames_wrapper =
        |x| to_windowed_frames(x, window.view(), hop_length, (n_pad_left, n_pad_right));
    let front_wav = pad(
        input.slice(s![..(win_length - 1)]),
        (win_length / 2, 0),
        Axis(0),
        PadMode::Reflect,
    );
    let mut front_frames = to_frames_wrapper(front_wav.view());

    let mut first_i = front_frames.len() * hop_length - win_length / 2;
    let mut frames: Vec<Array1<A>> = to_frames_wrapper(input.slice(s![first_i..]));

    first_i += frames.len() * hop_length;
    let i_back_wav_start = first_i.min(input.len() - win_length / 2 - 1);

    let mut back_wav = pad(
        input.slice(s![i_back_wav_start..]),
        (0, win_length / 2),
        Axis(0),
        PadMode::Reflect,
    );
    back_wav.slice_collapse(s![(first_i - i_back_wav_start).max(0)..]);
    let mut back_frames = to_frames_wrapper(back_wav.view());

    let n_frames = front_frames.len() + frames.len() + back_frames.len();
    let mut output = Array2::<Complex<A>>::zeros((n_frames, n_fft / 2 + 1));
    let out_frames: Vec<&mut [Complex<A>]> = output
        .axis_iter_mut(Axis(0))
        .map(|x| x.into_slice().unwrap())
        .collect();

    if parallel {
        let in_frames = front_frames
            .par_iter_mut()
            .chain(frames.par_iter_mut())
            .chain(back_frames.par_iter_mut());
        in_frames.zip(out_frames).for_each(|(x, y)| {
            let mut fft_module = RealFFT::<A>::new(n_fft).unwrap();
            let x = x.as_slice_mut().unwrap();
            fft_module.process(x, y).unwrap();
        });
    } else {
        let mut new_module;
        let fft_module = match fft_module {
            Some(m) => m,
            None => {
                new_module = RealFFT::<A>::new(n_fft).unwrap();
                &mut new_module
            }
        };
        let in_frames = front_frames
            .iter_mut()
            .chain(frames.iter_mut())
            .chain(back_frames.iter_mut());
        in_frames.zip(out_frames).for_each(|(x, y)| {
            let x = x.as_slice_mut().unwrap();
            fft_module.process(x, y).unwrap();
        });
    }

    output
}

#[inline]
pub fn calc_up_ratio(sr: u32, max_sr: u32, freq_scale: FreqScale) -> f32 {
    match freq_scale {
        FreqScale::Linear => max_sr as f32 / sr as f32,
        FreqScale::Mel => mel::from_hz(max_sr as f32 / 2.) / mel::from_hz(sr as f32 / 2.),
    }
}

#[inline]
fn to_windowed_frames<A: Float>(
    input: ArrayView1<A>,
    window: ArrayView1<A>,
    hop_length: usize,
    (n_pad_left, n_pad_right): (usize, usize),
) -> Vec<Array1<A>> {
    input
        .windows(window.len())
        .into_iter()
        .step_by(hop_length)
        .map(|x| {
            pad(
                (&x * &window).view(),
                (n_pad_left, n_pad_right),
                Axis(0),
                PadMode::Constant(A::zero()),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use ndarray::{arr2, Array1};
    use rustfft::num_complex::Complex;

    use super::super::utils::Impulse;
    use super::*;
    #[test]
    fn stft_works() {
        let impulse = Array1::<f32>::impulse(4, 2);
        assert_eq!(
            perform_stft(impulse.view(), 4, 2, 4, None, None, false),
            arr2(&[
                [
                    Complex::<f32>::new(0., 0.),
                    Complex::<f32>::new(0., 0.),
                    Complex::<f32>::new(0., 0.)
                ],
                [
                    Complex::<f32>::new(1. / 4., 0.),
                    Complex::<f32>::new(-1. / 4., 0.),
                    Complex::<f32>::new(1. / 4., 0.)
                ],
                [
                    Complex::<f32>::new(1. / 4., 0.),
                    Complex::<f32>::new(-1. / 4., 0.),
                    Complex::<f32>::new(1. / 4., 0.)
                ]
            ])
        );
    }
}
