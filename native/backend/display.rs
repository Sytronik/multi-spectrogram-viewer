use std::{iter, mem::MaybeUninit};
// use std::time::Instant;

use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use resize::{self, Pixel::GrayF32};
use tiny_skia::{Canvas, FillRule, LineCap, Paint, PathBuilder, PixmapMut, Rect, Stroke};

pub type ResizeType = resize::Type;

const BLACK: [u8; 3] = [0; 3];
const WHITE: [u8; 3] = [255; 3];
const THR_LONG_HEIGHT: f32 = 2.;
const THR_N_CONSEQ_LONG_H: usize = 5;
const WAV_STROKE_WIDTH: f32 = 1.75;
pub const COLORMAP: [[u8; 3]; 10] = [
    [0, 0, 4],
    [27, 12, 65],
    [74, 12, 107],
    [120, 28, 109],
    [165, 44, 96],
    [207, 68, 70],
    [237, 105, 37],
    [251, 155, 6],
    [247, 209, 61],
    [252, 255, 164],
];
pub const WAVECOLOR: [u8; 3] = [200, 21, 103];
pub const MAX_SIZE: u32 = 8192;

#[inline]
fn interpolate(rgba1: &[u8], rgba2: &[u8], ratio: f32) -> Vec<u8> {
    rgba1
        .iter()
        .zip(rgba2.iter())
        .map(|(&a, &b)| (ratio * a as f32 + (1. - ratio) * b as f32).round() as u8)
        .collect()
}

fn convert_grey_to_rgb(x: f32) -> Vec<u8> {
    if x < 0. {
        return BLACK.to_vec();
    }
    if x >= 1. {
        return WHITE.to_vec();
    }
    let position = x * COLORMAP.len() as f32;
    let index = position.floor() as usize;
    let rgba1 = if index >= COLORMAP.len() - 1 {
        &WHITE
    } else {
        &COLORMAP[index + 1]
    };
    interpolate(rgba1, &COLORMAP[index], position - index as f32)
}

pub fn convert_spec_to_grey(
    spec: ArrayView2<f32>,
    up_ratio: f32,
    max: f32,
    min: f32,
) -> Array2<f32> {
    // spec: T x F
    // return: grey image with F(inverted) x T
    let width = spec.shape()[0];
    let height = (spec.shape()[1] as f32 * up_ratio).round() as usize;
    let mut grey = Array2::maybe_uninit((height, width));
    grey.indexed_iter_mut().for_each(|((i, j), x)| {
        if height - 1 - i < spec.raw_dim()[1] {
            *x = MaybeUninit::new((spec[[j, height - 1 - i]] - min) / (max - min));
        } else {
            *x = MaybeUninit::new(0.);
        }
    });
    unsafe { grey.assume_init() }
}

pub fn draw_blended_spec_wav(
    spec_grey: ArrayView2<f32>,
    wav: ArrayView1<f32>,
    width: u32,
    height: u32,
    amp_range: (f32, f32),
    fast_resize: bool,
    blend: f64,
) -> Vec<u8> {
    let mut result = vec![0u8; width as usize * height as usize * 4];
    let pixmap = PixmapMut::from_bytes(&mut result[..], width, height).unwrap();
    let mut canvas = Canvas::from(pixmap);

    // spec
    if blend > 0. {
        // let start = Instant::now();

        colorize_grey_with_size_to(
            canvas.pixmap().data_mut(),
            spec_grey,
            width,
            height,
            fast_resize,
        );

        // println!("drawing spec: {:?}", start.elapsed());
    }

    if blend < 1. {
        // black
        // let start = Instant::now();
        if blend < 0.5 {
            let rect = Rect::from_xywh(0., 0., width as f32, height as f32).unwrap();
            let mut paint = Paint::default();
            paint.set_color_rgba8(0, 0, 0, (255. * (1. - 2. * blend)).round() as u8);
            canvas.fill_rect(rect, &paint);
        }
        // println!("drawing blackbox: {:?}", start.elapsed());

        // wave
        // let start = Instant::now();
        draw_wav_to(
            canvas.pixmap().data_mut(),
            wav,
            width,
            height,
            (255. * (2. - 2. * blend).min(1.)).round() as u8,
            amp_range,
        );
        // println!("drawing wav: {:?}", start.elapsed());
    }
    result
}

pub fn colorize_grey_with_size_to(
    output: &mut [u8],
    grey: ArrayView2<f32>,
    width: u32,
    height: u32,
    fast_resize: bool,
) {
    let resizetype = if fast_resize {
        ResizeType::Point
    } else {
        ResizeType::Lanczos3
    };
    let mut resizer = resize::new(
        grey.shape()[1] as usize,
        grey.shape()[0] as usize,
        width as usize,
        height as usize,
        GrayF32,
        resizetype,
    );
    let mut resized = vec![0f32; (width * height) as usize];
    resizer.resize(grey.as_slice().unwrap(), &mut resized[..]);
    resized
        .into_iter()
        .zip(output.chunks_exact_mut(4))
        .for_each(|(x, y)| {
            y[..3].copy_from_slice(&convert_grey_to_rgb(x));
            y[3] = 255;
        });
}

pub fn colorize_grey_with_size(
    grey: ArrayView2<f32>,
    width: u32,
    height: u32,
    fast_resize: bool,
) -> Vec<u8> {
    let resizetype = if fast_resize {
        ResizeType::Point
    } else {
        ResizeType::Lanczos3
    };
    let mut resizer = resize::new(
        grey.shape()[1] as usize,
        grey.shape()[0] as usize,
        width as usize,
        height as usize,
        GrayF32,
        resizetype,
    );
    let mut resized = vec![0f32; (width * height) as usize];
    resizer.resize(grey.as_slice().unwrap(), &mut resized[..]);
    resized
        .into_iter()
        .flat_map(|x| convert_grey_to_rgb(x).into_iter().chain(iter::once(255)))
        .collect()
}

fn draw_wav_directly(wav_avg: &[f32], canvas: &mut Canvas, paint: &Paint) {
    // println!("avg rendering. short height ratio: {}", n_short_height as f32 / width as f32);
    let path = {
        let mut pb = PathBuilder::new();
        pb.move_to(0., wav_avg[0]);
        for (x, &y) in wav_avg.iter().enumerate().skip(1) {
            pb.line_to(x as f32, y);
        }
        if wav_avg.len() == 1 {
            pb.line_to(0., wav_avg[0]);
        }
        pb.finish().unwrap()
    };

    let mut stroke = Stroke::default();
    stroke.width = WAV_STROKE_WIDTH;
    stroke.line_cap = LineCap::Round;
    canvas.stroke_path(&path, paint, &stroke);
}

fn draw_wav_topbottom(
    top_envelope: &[f32],
    bottom_envelope: &[f32],
    canvas: &mut Canvas,
    paint: &Paint,
) {
    // println!("top-bottom rendering. short height ratio: {}", n_short_height as f32 / width as f32);
    let path = {
        let mut pb = PathBuilder::new();
        pb.move_to(0., top_envelope[0]);
        for (x, &y) in top_envelope.iter().enumerate().skip(1) {
            pb.line_to(x as f32, y);
        }
        for (x, &y) in bottom_envelope.iter().enumerate().rev() {
            pb.line_to(x as f32, y);
        }
        pb.close();
        pb.finish().unwrap()
    };

    canvas.fill_path(&path, paint, FillRule::Winding);
}

pub fn draw_wav_to(
    output: &mut [u8],
    wav: ArrayView1<f32>,
    width: u32,
    height: u32,
    alpha: u8,
    amp_range: (f32, f32),
) {
    let pixmap = PixmapMut::from_bytes(output, width, height).unwrap();
    let mut canvas = Canvas::from(pixmap);

    let mut paint = Paint::default();
    let [r, g, b] = WAVECOLOR;
    paint.set_color_rgba8(r, g, b, alpha);
    paint.anti_alias = true;
    if amp_range.1 - amp_range.0 < 1e-16 {
        let rect = Rect::from_xywh(0., 0., width as f32, height as f32).unwrap();
        canvas.fill_rect(rect, &paint);
        return;
    }

    let amp_to_height_px = |x: f32| {
        ((amp_range.1 - x) * height as f32 / (amp_range.1 - amp_range.0))
            .max(0.)
            .min(height as f32)
    };
    let samples_per_px = wav.len() as f32 / width as f32;

    // need upsampling
    if samples_per_px < 2. {
        let mut upsampled = Array1::<f32>::zeros(width as usize);
        // naive upsampling
        let mut resizer = resize::new(
            wav.len(),
            1,
            width as usize,
            1,
            GrayF32,
            ResizeType::Triangle,
        );
        resizer.resize(wav.as_slice().unwrap(), upsampled.as_slice_mut().unwrap());
        upsampled.mapv_inplace(amp_to_height_px);
        draw_wav_directly(upsampled.as_slice().unwrap(), &mut canvas, &paint);
        return;
    }
    let mut top_envelope = Vec::<f32>::with_capacity(width as usize);
    let mut bottom_envelope = Vec::<f32>::with_capacity(width as usize);
    let mut wav_avg = Vec::<f32>::with_capacity(width as usize);
    let mut n_conseq_long_h = 0usize;
    let mut max_n_conseq = 0usize;
    for i_px in (0..width).into_iter() {
        let i_start = ((i_px as f32 - 0.5) * samples_per_px).round().max(0.) as usize;
        let i_end = (((i_px as f32 + 0.5) * samples_per_px).round() as usize).min(wav.len());
        let wav_slice = wav.slice(s![i_start..i_end]);
        let mut top = amp_to_height_px(*wav_slice.max().unwrap());
        let mut bottom = amp_to_height_px(*wav_slice.min().unwrap());
        let avg = amp_to_height_px(wav_slice.mean().unwrap());
        let diff = THR_LONG_HEIGHT + top - bottom;
        if diff < 0. {
            n_conseq_long_h += 1;
        } else {
            max_n_conseq = max_n_conseq.max(n_conseq_long_h);
            n_conseq_long_h = 0;
            top -= diff / 2.;
            bottom += diff / 2.;
        }
        top_envelope.push(top);
        bottom_envelope.push(bottom);
        wav_avg.push(avg);
    }
    max_n_conseq = max_n_conseq.max(n_conseq_long_h);
    if max_n_conseq > THR_N_CONSEQ_LONG_H {
        draw_wav_topbottom(&top_envelope[..], &bottom_envelope[..], &mut canvas, &paint);
    } else {
        draw_wav_directly(&wav_avg[..], &mut canvas, &paint);
    }
}

pub fn get_colormap_rgba() -> Vec<u8> {
    COLORMAP
        .iter()
        .flat_map(|x| x.iter().cloned().chain(iter::once(255)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use image::RgbImage;
    use resize::Pixel::RGB24;

    #[test]
    fn show_colorbar() {
        let (width, height) = (50, 500);
        let colormap: Vec<u8> = COLORMAP.iter().rev().flatten().cloned().collect();
        let mut imvec = vec![0u8; width * height * 3];
        let mut resizer = resize::new(1, 10, width, height, RGB24, ResizeType::Triangle);
        resizer.resize(&colormap, &mut imvec);

        RgbImage::from_raw(width as u32, height as u32, imvec)
            .unwrap()
            .save("samples/colorbar.png")
            .unwrap();
    }
}
