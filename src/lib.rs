use ndarray::prelude::*;
use ndarray::{stack, RemoveAxis, Slice};
use rustfft::num_complex::Complex;
use rustfft::num_traits::identities::*;
use rustfft::num_traits::{Float, Num};
use rustfft::FFTnum;

mod realfft;
mod windows;
mod audio;
use realfft::RealToComplex;

trait Impulse {
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

pub fn rfft<A>(input: &Array1<A>) -> Array1<Complex<A>>
where
    A: FFTnum + Float,
{
    let n_fft = input.shape()[0];
    let mut r2c = RealToComplex::<A>::new(n_fft).unwrap();
    let mut output = Array1::<Complex<A>>::zeros(n_fft / 2 + 1);
    r2c.process(&mut input.to_vec(), output.as_slice_mut().unwrap())
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
    match mode {
        PadMode::Constant(constant) => {
            let mut shape_left = array.raw_dim();
            let mut shape_right = array.raw_dim();
            shape_left[axis.index()] = n_pad_left;
            shape_right[axis.index()] = n_pad_right;
            let pad_left = Array::from_elem(shape_left, constant);
            let pad_right = Array::from_elem(shape_right, constant);
            stack![axis, pad_left.view(), array, pad_right.view()]
        }
        PadMode::Reflect => {
            let pad_left = array.slice_axis(
                axis, Slice::new(1, Some(n_pad_left as isize + 1), -1)
            );
            let pad_right = array.slice_axis(
                axis, Slice::new(-(n_pad_right as isize + 1), Some(-1), -1)
            );
            stack![axis, pad_left, array, pad_right]
        }
    }
}

pub fn stft<A>(input: &Array1<A>, win_length: usize, hop_length: usize) -> Array2<Complex<A>>
where
    A: FFTnum + Float,
{
    let n_fft = 2usize.pow((win_length as f32).log2().ceil() as u32);
    let n_frames = (input.len() - win_length) / hop_length + 1;
    let n_pad_left = (n_fft - win_length) / 2;
    let n_pad_right = (((n_fft - win_length) as f32) / 2.).ceil() as usize;

    let window = windows::hann(win_length, false);
    let win_pad_fn = |x| {
        pad(
            (&x * &window).view(),
            (n_pad_left, n_pad_right),
            Axis(0),
            PadMode::Constant(A::zero()),
        )
    };
    let mut input: Vec<Array1<A>> = input
        .windows(win_length)
        .into_iter()
        .step_by(hop_length)
        .map(win_pad_fn)
        .collect();

    let mut spec = Array2::<Complex<A>>::zeros((n_fft / 2 + 1, n_frames));
    let spec_view_mut: Vec<&mut [Complex<A>]> = spec
        .axis_iter_mut(Axis(1))
        .map(|x| x.into_slice().unwrap())
        .collect();

    let mut r2c = RealToComplex::<A>::new(n_fft).unwrap();
    for (x, y) in input.iter_mut().zip(spec_view_mut) {
        let x = x.as_slice_mut().unwrap();
        r2c.process(x, y).unwrap();
    }

    spec
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use crate::realfft::{ComplexToReal, RealToComplex};
    use crate::*;
    use ndarray::{arr1, arr2, Array1};
    use rustfft::num_complex::Complex;
    use rustfft::num_traits::Zero;
    use rustfft::FFTplanner;

    fn compare_complex(a: &[Complex<f64>], b: &[Complex<f64>], tol: f64) -> bool {
        a.iter().zip(b.iter()).fold(true, |eq, (val_a, val_b)| {
            eq && (val_a.re - val_b.re).abs() < tol && (val_a.im - val_b.im).abs() < tol
        })
    }

    fn compare_float<T: Float>(a: &[T], b: &[T], tol: T) -> bool {
        a.iter()
            .zip(b.iter())
            .fold(true, |eq, (val_a, val_b)| eq && (*val_a - *val_b).abs() < tol)
    }

    // Compare RealToComplex with standard FFT
    #[test]
    fn real_to_complex() {
        let mut indata = vec![0.0f64; 256];
        indata[0] = 1.0;
        indata[3] = 0.5;
        let mut indata_c = indata
            .iter()
            .map(|val| Complex::from(val))
            .collect::<Vec<Complex<f64>>>();
        let mut fft_planner = FFTplanner::<f64>::new(false);
        let fft = fft_planner.plan_fft(256);

        let mut r2c = RealToComplex::<f64>::new(256).unwrap();
        let mut out_a: Vec<Complex<f64>> = vec![Complex::zero(); 129];
        let mut out_b: Vec<Complex<f64>> = vec![Complex::zero(); 256];

        fft.process(&mut indata_c, &mut out_b);
        r2c.process(&mut indata, &mut out_a).unwrap();
        assert!(compare_complex(&out_a[0..129], &out_b[0..129], 1.0e-9));
    }

    // Compare ComplexToReal with standard iFFT
    #[test]
    fn complex_to_real() {
        let mut indata = vec![Complex::<f64>::zero(); 256];
        indata[0] = Complex::new(1.0, 0.0);
        indata[1] = Complex::new(1.0, 0.4);
        indata[255] = Complex::new(1.0, -0.4);
        indata[3] = Complex::new(0.3, 0.2);
        indata[253] = Complex::new(0.3, -0.2);

        let mut fft_planner = FFTplanner::<f64>::new(true);
        let fft = fft_planner.plan_fft(256);

        let mut c2r = ComplexToReal::<f64>::new(256).unwrap();
        let mut out_a: Vec<f64> = vec![0.0; 256];
        let mut out_b: Vec<Complex<f64>> = vec![Complex::zero(); 256];

        c2r.process(&indata[0..129], &mut out_a).unwrap();
        fft.process(&mut indata, &mut out_b);

        let out_b_r = out_b.iter().map(|val| 0.5 * val.re).collect::<Vec<f64>>();
        assert!(compare_float(&out_a, &out_b_r, 1.0e-9));
    }

    #[test]
    fn impulse_works() {
        assert_eq!(Array1::<f32>::impulse(4, 0), arr1(&[1., 0., 0., 0.]));
    }

    #[test]
    fn rfft_wrapper_works() {
        assert_eq!(
            rfft(&Array1::<f32>::impulse(4, 0)),
            arr1(&[Complex::<f32>::new(1., 0.); 3])
        );
    }

    #[test]
    fn hann_window_works() {
        assert_eq!(windows::hann::<f32>(4, false), arr1(&[0., 0.5, 1., 0.5]));
    }

    #[test]
    fn pad_works() {
        assert_eq!(
            pad(arr2(&[[1, 2, 3]]).view(), (1, 2), Axis(0), PadMode::Constant(10)),
            arr2(&[[10, 10, 10], 
                   [1, 2, 3], 
                   [10, 10, 10], 
                   [10, 10, 10] ])
        );
        assert_eq!(
            pad(arr2(&[[1, 2, 3]]).view(), (1, 2), Axis(1), PadMode::Reflect),
            arr2(&[[2, 1, 2, 3, 2, 1]])
        );
    }

    #[test]
    fn stft_works() {
        assert_eq!(
            stft(&Array1::<f32>::impulse(4, 2), 4, 2),
            arr2(&[
                [Complex::<f32>::new(1., 0.)],
                [Complex::<f32>::new(-1., 0.)],
                [Complex::<f32>::new(1., 0.)]
            ])
        );
    }
    
    #[test]
    fn open_audio_works() {
        let (wav, sr) = audio::open_audio_file(Path::new("samples/sample.wav")).unwrap();
        let arr = arr2(&[
            [-1.919269561767578125e-05,
            2.510547637939453125e-04,
            2.177953720092773438e-04,
            8.809566497802734375e-05,
            1.561641693115234375e-05,
            1.788139343261718750e-05,
            1.298189163208007812e-04,
            1.105070114135742188e-04,
            -1.615285873413085938e-04,
            -4.312992095947265625e-04,
            -4.181861877441406250e-04,
            -1.516342163085937500e-04,
            -3.480911254882812500e-05,
            -2.431869506835937500e-05,
            -1.041889190673828125e-04,
            -1.143217086791992188e-04],
        ]);
        assert_eq!(sr, 48000);
        assert_eq!(wav.shape(), &[1, 320911]);
        assert_eq!(
            compare_float(
                &[wav.iter().fold(f32::MIN, |s, &x| if x > s {x} else {s} )],
                &[0.1715821],
                f32::EPSILON,
            ),
            true
        );
        assert_eq!(
            compare_float(
                wav.as_slice().unwrap(),
                arr.as_slice().unwrap(),
                f32::EPSILON,
            ),
            true
        );
    }

}
