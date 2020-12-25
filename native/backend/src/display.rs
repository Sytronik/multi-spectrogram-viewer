use std::error::Error as stdError;
use std::time::Instant;

use image::{
    imageops::{resize, FilterType},
    ImageBuffer, Luma, Rgba,
};
use imageproc::pixelops::interpolate;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use tiny_skia::{Canvas, FillRule, LineCap, Paint, PathBuilder, PixmapMut, Rect, Stroke};

pub type GreyF32Image = ImageBuffer<Luma<f32>, Vec<f32>>;

pub const COLORMAP: [[u8; 4]; 10] = [
    [0, 0, 4, 255],
    [27, 12, 65, 255],
    [74, 12, 107, 255],
    [120, 28, 109, 255],
    [165, 44, 96, 255],
    [207, 68, 70, 255],
    [237, 105, 37, 255],
    [251, 155, 6, 255],
    [247, 209, 61, 255],
    [252, 255, 164, 255],
];
pub const WAVECOLOR: [u8; 4] = [200, 21, 103, 255];

fn convert_grey_to_color(x: &Luma<f32>) -> Rgba<u8> {
    assert!(x.0[0] >= 0.);
    let position = (COLORMAP.len() as f32) * x.0[0];
    let index = position.floor() as usize;
    if index >= COLORMAP.len() - 1 {
        Rgba(COLORMAP[COLORMAP.len() - 1])
    } else {
        let ratio = position - index as f32;
        interpolate(Rgba(COLORMAP[index + 1]), Rgba(COLORMAP[index]), ratio)
    }
}

pub fn spec_to_grey(spec: ArrayView2<f32>, up_ratio: f32, max: f32, min: f32) -> GreyF32Image {
    let height = (spec.shape()[1] as f32 * up_ratio).round() as u32;
    GreyF32Image::from_fn(spec.shape()[0] as u32, height, |x, y| {
        if y >= height - spec.shape()[1] as u32 {
            let db = spec[[x as usize, (height - 1 - y) as usize]];
            Luma([((db - min) / (max - min)).max(0.).min(1.)])
        } else {
            Luma([0.])
        }
    })
}

pub fn blend_spec_wav(
    output: &mut [u8],
    spec_grey: &GreyF32Image,
    wav: ArrayView1<f32>,
    width: u32,
    height: u32,
    blend: f64,
) -> Result<(), Box<dyn stdError>> {
    let pixmap = PixmapMut::from_bytes(output, width, height).unwrap();
    let mut canvas = Canvas::from(pixmap);

    // spec
    if blend > 0. {
        let start = Instant::now();
        grey_to_rgb(canvas.pixmap().data_mut(), spec_grey, width, height);
        println!("drawing spec: {:?}", start.elapsed());
    }

    if blend < 1. {
        // black
        let start = Instant::now();
        if blend < 0.5 {
            let rect = Rect::from_xywh(0., 0., width as f32, height as f32).unwrap();
            let mut paint = Paint::default();
            paint.set_color_rgba8(0, 0, 0, (255. * (1. - 2. * blend)) as u8);
            canvas.fill_rect(rect, &paint);
        }
        println!("drawing blackbox: {:?}", start.elapsed());

        // wave
        let start = Instant::now();
        draw_wav(
            canvas.pixmap().data_mut(),
            wav,
            width,
            height,
            (255. * (2. - 2. * blend).min(1.)) as u8,
            (-1., 1.),
        );
        println!("drawing wav: {:?}", start.elapsed());
    }

    Ok(())
}

pub fn grey_to_rgb(output: &mut [u8], grey: &GreyF32Image, width: u32, height: u32) {
    let resized = resize(grey, width, height, FilterType::Lanczos3);
    resized
        .pixels()
        .zip(output.chunks_exact_mut(4))
        .for_each(|(x, y)| {
            let [r, g, b, a] = convert_grey_to_color(x).0;
            y[0] = r;
            y[1] = g;
            y[2] = b;
            y[3] = a;
        });
}

pub fn draw_wav(
    output: &mut [u8],
    wav: ArrayView1<f32>,
    width: u32,
    height: u32,
    alpha: u8,
    amp_range: (f32, f32),
) {
    let amp_to_height_px =
        |x: f32| ((amp_range.1 - x) * height as f32 / (amp_range.1 - amp_range.0));
    let samples_per_px = wav.len() as f32 / width as f32;
    let mut max_envelope = Vec::<f32>::with_capacity(width as usize);
    let mut min_envelope = Vec::<f32>::with_capacity(width as usize);
    let mut avg_envelope = Vec::<f32>::with_capacity(width as usize);
    let mut n_short_height = 0u32;
    for i_px in (0..width as i32).into_iter() {
        let i_start = ((i_px as f32 - 0.5) * samples_per_px).round().max(0.) as usize;
        let i_end = (((i_px as f32 + 0.5) * samples_per_px).round() as usize).min(wav.len());
        let wav_slice = wav.slice(s![i_start..i_end]);
        let max = *wav_slice.max().unwrap();
        let min = *wav_slice.min().unwrap();
        let avg = wav_slice.mean().unwrap();
        max_envelope.push(max);
        min_envelope.push(min);
        avg_envelope.push(avg);
        if amp_to_height_px(min) - amp_to_height_px(max) < 3. {
            n_short_height += 1;
        }
    }
    let pixmap = PixmapMut::from_bytes(output, width, height).unwrap();
    let mut canvas = Canvas::from(pixmap);

    let mut paint = Paint::default();
    let [r, g, b, _] = WAVECOLOR;
    paint.set_color_rgba8(r, g, b, alpha);
    paint.anti_alias = true;
    if width == 1 || n_short_height < width / 3 {
        // println!("min-max rendering. short height ratio: {}", n_short_height as f32 / width as f32);
        let path = {
            let mut pb = PathBuilder::new();
            pb.move_to(0., amp_to_height_px(max_envelope[0]));
            for (x, &y) in max_envelope.iter().enumerate().skip(1) {
                pb.line_to(x as f32, amp_to_height_px(y));
            }
            for (x, &y) in min_envelope.iter().enumerate().rev() {
                pb.line_to(x as f32, amp_to_height_px(y));
            }
            pb.close();
            pb.finish().unwrap()
        };

        canvas.fill_path(&path, &paint, FillRule::Winding);
    } else {
        // println!("avg rendering. short height ratio: {}", n_short_height as f32 / width as f32);
        let path = {
            let mut pb = PathBuilder::new();
            pb.move_to(0., amp_to_height_px(avg_envelope[0]));
            for (x, &y) in avg_envelope.iter().enumerate().skip(1) {
                pb.line_to(x as f32, amp_to_height_px(y));
            }
            pb.finish().unwrap()
        };

        let mut stroke = Stroke::default();
        stroke.width = 1.75;
        stroke.line_cap = LineCap::Round;
        canvas.stroke_path(&path, &paint, &stroke);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use image::{Rgba, RgbaImage};

    #[test]
    fn show_colorbar() {
        let colormap: Vec<Rgba<u8>> = COLORMAP.iter().map(|&x| Rgba(x)).collect();
        let mut im =
            RgbaImage::from_fn(50, colormap.len() as u32, |_, y| Rgba(COLORMAP[y as usize]));
        im = resize(&im, 50, 500, FilterType::Triangle);
        im.save("../../samples/colorbar.png").unwrap();
    }
}
