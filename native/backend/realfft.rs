use std::error;
use std::f64::consts::PI as PIf64;
use std::fmt;

use rustfft::{
    num_complex::Complex,
    num_traits::{Float, Zero},
    FftNum, FftPlanner,
};

type Res<T> = Result<T, Box<dyn error::Error>>;

/// Custom error returned by FFTs
#[derive(Debug)]
pub struct FftError {
    desc: String,
}

impl fmt::Display for FftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.desc)
    }
}

impl error::Error for FftError {
    fn description(&self) -> &str {
        &self.desc
    }
}

impl FftError {
    pub fn new(desc: &str) -> Self {
        FftError {
            desc: desc.to_owned(),
        }
    }
}

/// An FFT that takes a real-valued input vector of length 2*N and transforms it to a complex
/// spectrum of length N+1.
#[readonly::make]
pub struct RealFFT<T: FftNum + Float> {
    sin_cos: Vec<(T, T)>,
    pub length: usize,
    fft: std::sync::Arc<dyn rustfft::Fft<T>>,
    buffer_out: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,
}

/// An FFT that takes a real-valued input vector of length 2*N and transforms it to a complex
/// spectrum of length N+1.
#[allow(dead_code)]
#[readonly::make]
pub struct InvRealFFT<T: FftNum + Float> {
    sin_cos: Vec<(T, T)>,
    pub length: usize,
    fft: std::sync::Arc<dyn rustfft::Fft<T>>,
    buffer_in: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,
}

fn zip4<A, B, C, D>(
    a: A,
    b: B,
    c: C,
    d: D,
) -> impl Iterator<Item = (A::Item, B::Item, C::Item, D::Item)>
where
    A: IntoIterator,
    B: IntoIterator,
    C: IntoIterator,
    D: IntoIterator,
{
    a.into_iter()
        .zip(b.into_iter().zip(c.into_iter().zip(d)))
        .map(|(w, (x, (y, z)))| (w, x, y, z))
}

impl<T> RealFFT<T>
where
    T: FftNum + Float,
{
    /// Create a new RealToComplex FFT for input data of a given length. Returns an error if the length is not even.
    pub fn new(length: usize) -> Res<Self> {
        if length % 2 > 0 {
            return Err(Box::new(FftError::new("Length must be even")));
        }
        let buffer_out = vec![Complex::zero(); length / 2 + 1];
        let mut sin_cos = Vec::with_capacity(length / 2);
        let pi = T::from_f64(PIf64).unwrap();
        let halflength = T::from_usize(length / 2).unwrap();
        for k in 0..length / 2 {
            let k = T::from_usize(k).unwrap();
            let sin = (k * pi / halflength).sin();
            let cos = (k * pi / halflength).cos();
            sin_cos.push((sin, cos));
        }
        let mut fft_planner = FftPlanner::<T>::new();
        let fft = fft_planner.plan_fft_forward(length / 2);
        let scratch = vec![Complex::zero(); fft.get_outofplace_scratch_len()];
        Ok(RealFFT {
            sin_cos,
            length,
            fft,
            buffer_out,
            scratch,
        })
    }

    /// Transform a vector of 2*N real-valued samples, storing the result in the N+1 element long complex output vector.
    /// The input buffer is used as scratch space, so the contents of input should be considered garbage after calling.
    pub fn process(&mut self, input: &mut [T], output: &mut [Complex<T>]) -> Res<()> {
        if input.len() != self.length {
            return Err(Box::new(FftError::new(
                format!(
                    "Wrong length of input, expected {}, got {}",
                    self.length,
                    input.len()
                )
                .as_str(),
            )));
        }
        if output.len() != (self.length / 2 + 1) {
            return Err(Box::new(FftError::new(
                format!(
                    "Wrong length of output, expected {}, got {}",
                    self.length / 2 + 1,
                    input.len()
                )
                .as_str(),
            )));
        }
        let fftlen = self.length / 2;
        //for (val, buf) in input.chunks(2).take(fftlen).zip(self.buffer_in.iter_mut()) {
        //    *buf = Complex::new(val[0], val[1]);
        //}
        let mut buf_in = unsafe {
            let ptr = input.as_mut_ptr() as *mut Complex<T>;
            let len = input.len();
            std::slice::from_raw_parts_mut(ptr, len / 2)
        };

        // FFT and store result in buffer_out
        self.fft.process_outofplace_with_scratch(
            &mut buf_in,
            &mut self.buffer_out[0..fftlen],
            &mut self.scratch,
        );

        self.buffer_out[fftlen] = self.buffer_out[0];

        for (&buf, &buf_rev, &(sin, cos), out) in zip4(
            &self.buffer_out,
            self.buffer_out.iter().rev(),
            &self.sin_cos,
            &mut output[..],
        ) {
            let xr = T::from(0.5).unwrap()
                * ((buf.re + buf_rev.re) + cos * (buf.im + buf_rev.im)
                    - sin * (buf.re - buf_rev.re));
            let xi = T::from(0.5).unwrap()
                * ((buf.im - buf_rev.im)
                    - sin * (buf.im + buf_rev.im)
                    - cos * (buf.re - buf_rev.re));
            *out = Complex::new(xr, xi);
        }
        output[fftlen] = Complex::new(self.buffer_out[0].re - self.buffer_out[0].im, T::zero());
        Ok(())
    }
}

/// Create a new ComplexToReal iFFT for output data of a given length. Returns an error if the length is not even.
#[allow(dead_code)]
impl<T> InvRealFFT<T>
where
    T: FftNum + Float,
{
    pub fn new(length: usize) -> Res<Self> {
        if length % 2 > 0 {
            return Err(Box::new(FftError::new("Length must be even")));
        }
        let buffer_in = vec![Complex::zero(); length / 2];
        let mut sin_cos = Vec::with_capacity(length / 2);
        let pi = T::from_f64(std::f64::consts::PI).unwrap();
        let halflength = T::from_usize(length / 2).unwrap();
        for k in 0..length / 2 {
            let k = T::from_usize(k).unwrap();
            let sin = (k * pi / halflength).sin();
            let cos = (k * pi / halflength).cos();
            sin_cos.push((sin, cos));
        }
        let mut fft_planner = FftPlanner::<T>::new();
        let fft = fft_planner.plan_fft_inverse(length / 2);
        let scratch = vec![Complex::zero(); fft.get_outofplace_scratch_len()];
        Ok(InvRealFFT {
            sin_cos,
            length,
            fft,
            buffer_in,
            scratch,
        })
    }

    /// Transform a complex spectrum of N+1 values and store the real result in the 2*N long output.
    pub fn process(&mut self, input: &[Complex<T>], output: &mut [T]) -> Res<()> {
        if input.len() != (self.length / 2 + 1) {
            return Err(Box::new(FftError::new(
                format!(
                    "Wrong length of input, expected {}, got {}",
                    self.length / 2 + 1,
                    input.len()
                )
                .as_str(),
            )));
        }
        if output.len() != self.length {
            return Err(Box::new(FftError::new(
                format!(
                    "Wrong length of output, expected {}, got {}",
                    self.length,
                    input.len()
                )
                .as_str(),
            )));
        }

        for (&buf, &buf_rev, &(sin, cos), fft_input) in zip4(
            input,
            input.iter().rev(),
            &self.sin_cos,
            &mut self.buffer_in[..],
        ) {
            let xr =
                (buf.re + buf_rev.re) - cos * (buf.im + buf_rev.im) - sin * (buf.re - buf_rev.re);
            let xi =
                (buf.im - buf_rev.im) + cos * (buf.re - buf_rev.re) - sin * (buf.im + buf_rev.im);
            *fft_input = Complex::new(xr, xi);
        }

        // FFT and store result in buffer_out
        let mut buf_out = unsafe {
            let ptr = output.as_mut_ptr() as *mut Complex<T>;
            let len = output.len();
            std::slice::from_raw_parts_mut(ptr, len / 2)
        };
        self.fft.process_outofplace_with_scratch(
            &mut self.buffer_in,
            &mut buf_out,
            &mut self.scratch,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rustfft::num_complex::Complex;
    use rustfft::num_traits::Zero;
    use rustfft::FftPlanner;

    // Compare RealToComplex with standard FFT
    #[test]
    fn real_to_complex() {
        let mut indata = vec![0.0f64; 256];
        indata[0] = 1.0;
        indata[3] = 0.5;
        let mut inout_b = indata
            .iter()
            .map(|val| Complex::from(val))
            .collect::<Vec<Complex<f64>>>();
        let mut fft_planner = FftPlanner::<f64>::new();
        let fft = fft_planner.plan_fft_forward(256);

        let mut r2c = RealFFT::<f64>::new(256).unwrap();
        let mut out_a = vec![Complex::<f64>::zero(); 129];

        fft.process(&mut inout_b);
        r2c.process(&mut indata, &mut out_a).unwrap();
        assert_abs_diff_eq!(&out_a[0..129], &inout_b[0..129], epsilon = 1e-15);
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

        let mut fft_planner = FftPlanner::<f64>::new();
        let fft = fft_planner.plan_fft_inverse(256);

        let mut c2r = InvRealFFT::<f64>::new(256).unwrap();
        let mut out_a = vec![0f64; 256];

        c2r.process(&indata[0..129], &mut out_a).unwrap();
        fft.process(&mut indata);

        let out_b: Vec<f64> = indata.iter().map(|val| 0.5 * val.re).collect();
        assert_abs_diff_eq!(&out_a[..], &out_b[..], epsilon = 1e-15);
    }
}
