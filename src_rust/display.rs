use image::{imageops, ImageBuffer, Luma, Rgb, RgbImage};
use ndarray::prelude::*;

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

fn convert_grey_to_color(x: f32) -> Rgb<u8> {
    assert!(x >= 0.);
    let position = (COLORMAP.len() as f32) * x;
    let index = position.floor() as usize;
    if index >= COLORMAP.len() - 1 {
        Rgb(COLORMAP[COLORMAP.len() - 1])
    } else {
        let ratio = position - index as f32;
        let mut color = [0u8; 3];
        for (i, (&a, &b)) in COLORMAP[index].iter().zip(COLORMAP[index + 1].iter()).enumerate() {
            color[i] = (ratio * b as f32 + (1. - ratio) * a as f32).round() as u8;
        }
        Rgb(color)
    }
}

pub fn spec_to_image(
    spec_db: ArrayView2<f32>,
    nwidth: u32,
    nheight: u32,
    max_db: f32,
    min_db: f32,
) -> RgbImage {
    let im = ImageBuffer::<Luma<f32>, Vec<f32>>::from_fn(
        spec_db.shape()[0] as u32,
        spec_db.shape()[1] as u32,
        |x, y| {
            Luma([
                ((spec_db[[x as usize, spec_db.shape()[1] - y as usize - 1]] - min_db)
                    / (max_db - min_db))
                    .max(0.)
                    .min(1.),
            ])
        },
    );
    let im = imageops::resize(&im, nwidth, nheight, imageops::FilterType::Lanczos3);
    RgbImage::from_fn(nwidth, nheight, |x, y| {
        convert_grey_to_color(im.get_pixel(x, y)[0])
    })
}

pub fn colorbar(length: u32) -> RgbImage {
    let colormap: Vec<Rgb<u8>> = COLORMAP.iter().map(|&x| Rgb(x)).collect();
    let im = RgbImage::from_fn(50, colormap.len() as u32, |_, y| Rgb(COLORMAP[y as usize]));
    imageops::resize(&im, 50, length, imageops::FilterType::Triangle)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn colorbar_works() {
        let im = colorbar(500);
        im.save("colorbar.png").unwrap();
    }
}
