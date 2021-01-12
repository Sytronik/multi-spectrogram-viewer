use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io;
use std::iter;
use std::path::PathBuf;
use std::time::Instant;

use approx::abs_diff_ne;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use rayon::prelude::*;

mod audio;
mod decibel;
pub mod display;
mod mel;
mod realfft;
mod stft;
pub mod utils;
mod windows;

use decibel::DeciBelInplace;
use stft::{calc_up_ratio, perform_stft, FreqScale};
use utils::calc_proper_n_fft;
use windows::{calc_normalized_win, WindowType};

pub use display::COLORMAP;
pub type IdChVec = Vec<(usize, usize)>;
pub type IdChArr = [(usize, usize)];
pub type IdChSet = HashSet<(usize, usize)>;
pub type IdChMap<T> = HashMap<(usize, usize), T>;
pub type SrMap<T> = HashMap<u32, T>;

const MIN_WIDTH: u32 = 1;

#[derive(Debug)]
pub struct SpecSetting {
    win_ms: f32,
    t_overlap: usize,
    f_overlap: usize,
    freq_scale: FreqScale,
    db_range: f32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DrawOption {
    pub px_per_sec: f64,
    pub height: u32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DrawOptionForWav {
    pub amp_range: (f32, f32),
}

pub enum ImageKind {
    Spec,
    Wav(DrawOptionForWav),
}

#[readonly::make]
pub struct AudioTrack {
    pub sr: u32,
    pub n_ch: usize,
    path: PathBuf,
    wavs: Array2<f32>,
    win_length: usize,
    hop_length: usize,
    n_fft: usize,
}

impl AudioTrack {
    pub fn new(path: String, setting: &SpecSetting) -> io::Result<Self> {
        let (wavs, sr) = audio::open_audio_file(path.as_str())?;
        let n_ch = wavs.shape()[0];
        // let wav = wav.slice_move(s![144000..144000 + 4096]);
        let win_length = setting.win_ms * sr as f32 / 1000.;
        let hop_length = (win_length / setting.t_overlap as f32).round() as usize;
        let win_length = hop_length * setting.t_overlap;
        let n_fft = calc_proper_n_fft(win_length) * setting.f_overlap;
        Ok(AudioTrack {
            path: PathBuf::from(path),
            wavs,
            sr,
            n_ch,
            win_length,
            hop_length,
            n_fft,
        })
    }

    pub fn reload(&mut self, setting: &SpecSetting) -> io::Result<()> {
        let new = AudioTrack::new(self.path.as_path().display().to_string(), setting)?;
        *self = new;
        Ok(())
    }

    #[inline]
    pub fn get_wav(&self, ch: usize) -> ArrayView1<f32> {
        self.wavs.index_axis(Axis(0), ch)
    }

    #[inline]
    pub fn path_string(&self) -> String {
        self.path.as_os_str().to_string_lossy().into_owned()
    }

    #[inline]
    pub fn filename(&self) -> String {
        self.path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned()
    }

    #[inline]
    pub fn wavlen(&self) -> usize {
        self.wavs.shape()[1]
    }

    #[inline]
    pub fn sec(&self) -> f64 {
        self.wavs.shape()[1] as f64 / self.sr as f64
    }

    #[inline]
    pub fn calc_width(&self, px_per_sec: f64) -> u32 {
        (px_per_sec * self.wavs.shape()[1] as f64 / self.sr as f64)
            .max(MIN_WIDTH as f64)
            .round() as u32
    }
}

impl fmt::Debug for AudioTrack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AudioTrack {{\n\
                path: {},\n sr: {} Hz, n_ch: {}, length: {}, sec: {}\n\
                win_length: {}, hop_length: {}, n_fft: {}\n\
            }}",
            self.path.to_str().unwrap(),
            self.sr,
            self.n_ch,
            self.wavs.shape()[1],
            self.sec(),
            self.win_length,
            self.hop_length,
            self.n_fft,
        )
    }
}

#[readonly::make]
pub struct TrackManager {
    pub tracks: HashMap<usize, AudioTrack>,
    pub max_db: f32,
    pub min_db: f32,
    pub max_sec: f64,
    pub max_sr: u32,
    setting: SpecSetting,
    windows: SrMap<Array1<f32>>,
    mel_fbs: SrMap<Array2<f32>>,
    specs: IdChMap<Array2<f32>>,
    spec_greys: IdChMap<Array2<f32>>,
    id_max_sec: usize,
}

impl TrackManager {
    pub fn new() -> Self {
        TrackManager {
            tracks: HashMap::new(),
            setting: SpecSetting {
                win_ms: 40.,
                t_overlap: 4,
                f_overlap: 1,
                freq_scale: FreqScale::Mel,
                db_range: 120.,
            },
            windows: HashMap::new(),
            mel_fbs: HashMap::new(),
            specs: HashMap::new(),
            spec_greys: HashMap::new(),
            max_db: -f32::INFINITY,
            min_db: f32::INFINITY,
            max_sec: 0.,
            id_max_sec: 0,
            max_sr: 0,
        }
    }

    pub fn add_tracks(&mut self, id_list: &[usize], path_list: Vec<String>) -> io::Result<bool> {
        let mut new_sr_set = HashSet::<(u32, usize, usize)>::new();
        for (&id, path) in id_list.iter().zip(path_list.into_iter()) {
            let track = AudioTrack::new(path, &self.setting)?;
            let sec = track.sec();
            if sec > self.max_sec {
                self.max_sec = sec;
                self.id_max_sec = id;
            }
            if self.windows.get(&track.sr).is_none() {
                new_sr_set.insert((track.sr, track.win_length, track.n_fft));
            }
            self.tracks.insert(id, track);
        }

        self.update_specs(id_list, new_sr_set);
        Ok(self.update_greys(Some(id_list)))
    }

    pub fn remove_tracks(&mut self, id_list: &[usize]) -> bool {
        for id in id_list.iter() {
            let (_, removed) = self.tracks.remove_entry(&id).unwrap();
            for ch in (0..removed.n_ch).into_iter() {
                self.specs.remove(&(*id, ch));
                self.spec_greys.remove(&(*id, ch));
            }
            if self.tracks.par_iter().all(|(_, tr)| tr.sr != removed.sr) {
                self.windows.remove(&removed.sr);
                self.mel_fbs.remove(&removed.sr);
            }
        }
        if id_list.contains(&self.id_max_sec) {
            let (id, max_sec) = self
                .tracks
                .par_iter()
                .map(|(&id, track)| (id, track.sec()))
                .reduce(
                    || (0, 0.),
                    |(id_max, max), (id, sec)| {
                        if sec > max {
                            (id, sec)
                        } else {
                            (id_max, max)
                        }
                    },
                );
            self.id_max_sec = id;
            self.max_sec = max_sec;
        }

        self.update_greys(None)
    }

    pub fn get_entire_images(
        &self,
        id_ch_tuples: &IdChArr,
        option: DrawOption,
        kind: ImageKind,
    ) -> IdChMap<Array3<u8>> {
        let start = Instant::now();
        let DrawOption { px_per_sec, height } = option;
        let mut result = IdChMap::with_capacity(id_ch_tuples.len());
        result.par_extend(id_ch_tuples.par_iter().map(|&(id, ch)| {
            let track = self.tracks.get(&id).unwrap();
            let width = track.calc_width(px_per_sec);
            let arr = match kind {
                ImageKind::Spec => {
                    let vec = display::colorize_grey_with_size(
                        self.spec_greys.get(&(id, ch)).unwrap().view(),
                        width,
                        height,
                        false,
                    );
                    Array3::from_shape_vec((height as usize, width as usize, 4), vec).unwrap()
                }
                ImageKind::Wav(option_for_wav) => {
                    let mut arr = Array3::zeros((height as usize, width as usize, 4));
                    display::draw_wav_to(
                        arr.as_slice_mut().unwrap(),
                        track.get_wav(ch),
                        width,
                        height,
                        255,
                        option_for_wav.amp_range,
                    );
                    arr
                }
            };
            ((id, ch), arr)
        }));
        println!("draw entire: {:?}", start.elapsed());
        result
    }

    pub fn get_part_images(
        &self,
        id_ch_tuples: &IdChArr,
        sec: f64,
        width: u32,
        option: DrawOption,
        kind: ImageKind,
        fast_resize_vec: Option<Vec<bool>>,
    ) -> IdChMap<Vec<u8>> {
        let start = Instant::now();
        let DrawOption { px_per_sec, height } = option;
        let mut result = IdChMap::with_capacity(id_ch_tuples.len());
        let par_iter = id_ch_tuples.par_iter().enumerate().map(|(i, &(id, ch))| {
            // let par_iter = id_ch_tuples.iter().enumerate().map(|(i, &(id, ch))| {
            let (pad_left, drawing_width, pad_right) =
                self.calc_drawing_pad_width_of(id, sec, width, px_per_sec);

            let create_empty_im_entry =
                || ((id, ch), vec![0u8; width as usize * height as usize * 4]);
            if drawing_width == 0 {
                return create_empty_im_entry();
            }

            let arr = match kind {
                ImageKind::Spec => {
                    let grey_sub = match self.crop_grey_of(id, ch, sec, width, px_per_sec) {
                        Some(x) => x,
                        None => return create_empty_im_entry(),
                    };
                    let vec = display::colorize_grey_with_size(
                        grey_sub.view(),
                        drawing_width,
                        height,
                        match fast_resize_vec {
                            Some(ref vec) => vec[i],
                            None => false,
                        },
                    );
                    Array3::from_shape_vec((height as usize, drawing_width as usize, 4), vec)
                        .unwrap()
                }
                ImageKind::Wav(option_for_wav) => {
                    let wav_slice = match self.slice_wav_of(id, ch, sec, width, px_per_sec) {
                        Some(x) => x,
                        None => return create_empty_im_entry(),
                    };
                    let mut arr = Array3::zeros((height as usize, drawing_width as usize, 4));
                    display::draw_wav_to(
                        arr.as_slice_mut().unwrap(),
                        wav_slice,
                        drawing_width,
                        height,
                        255,
                        option_for_wav.amp_range,
                    );
                    arr
                }
            };

            if width == drawing_width {
                ((id, ch), arr.into_raw_vec())
            } else {
                let arr = utils::pad(
                    arr.view(),
                    (pad_left as usize, pad_right as usize),
                    Axis(1),
                    utils::PadMode::Constant(0),
                );
                ((id, ch), arr.into_raw_vec())
            }
        });
        result.par_extend(par_iter);

        println!("draw: {:?}", start.elapsed());
        result
    }

    pub fn get_overview_of(&self, id: usize, width: u32, height: u32) -> Vec<u8> {
        let track = self.tracks.get(&id).unwrap();
        let ch_h = height / track.n_ch as u32;
        let i_start = (height % track.n_ch as u32 / 2 * width * 4) as usize;
        let i_end = i_start + (track.n_ch as u32 * ch_h * width * 4) as usize;
        let mut result = vec![0u8; (height * width * 4) as usize];
        result[i_start..i_end]
            .par_chunks_exact_mut((ch_h * width * 4) as usize)
            .enumerate()
            .for_each(|(ch, x)| {
                display::draw_wav_to(&mut x[..], track.get_wav(ch), width, ch_h, 255, (-1., 1.))
            });
        result
    }

    pub fn get_spec_image_of(&self, id: usize, ch: usize, width: u32, height: u32) -> Vec<u8> {
        display::colorize_grey_with_size(
            self.spec_greys.get(&(id, ch)).unwrap().view(),
            width,
            height,
            false,
        )
    }

    pub fn get_wav_image_of(
        &self,
        id: usize,
        ch: usize,
        width: u32,
        height: u32,
        amp_range: (f32, f32),
    ) -> Vec<u8> {
        let mut result = vec![0u8; (width * height * 4) as usize];
        display::draw_wav_to(
            &mut result[..],
            self.tracks.get(&id).unwrap().get_wav(ch),
            width,
            height,
            255,
            amp_range,
        );
        result
    }

    pub fn get_blended_image_of(
        &self,
        id: usize,
        ch: usize,
        width: u32,
        height: u32,
        option_for_wav: DrawOptionForWav,
        blend: f64,
    ) -> Vec<u8> {
        display::draw_blended_spec_wav(
            self.spec_greys.get(&(id, ch)).unwrap().view(),
            self.tracks.get(&id).unwrap().get_wav(ch),
            width,
            height,
            option_for_wav.amp_range,
            false,
            blend,
        )
    }

    fn calc_drawing_pad_width_of(
        &self,
        id: usize,
        sec: f64,
        width: u32,
        px_per_sec: f64,
    ) -> (u32, u32, u32) {
        let track = self.tracks.get(&id).unwrap();

        let total_width = (px_per_sec * track.wavlen() as f64 / track.sr as f64).max(1.);
        let pad_left = ((-sec * px_per_sec).max(0.).round() as u32).min(width);
        let pad_right = ((sec * px_per_sec + width as f64 - total_width)
            .max(0.)
            .round() as u32)
            .min(width - pad_left);

        let drawing_width = width - pad_left - pad_right;
        (pad_left, drawing_width, pad_right)
    }

    #[inline]
    pub fn id_ch_tuples(&self) -> IdChVec {
        self.specs.keys().cloned().collect()
    }

    #[inline]
    pub fn id_ch_tuples_from(&self, id_list: &[usize]) -> IdChVec {
        id_list
            .iter()
            .flat_map(|&id| {
                let n_ch = self.tracks.get(&id).unwrap().n_ch;
                iter::repeat(id).zip((0..n_ch).into_iter())
            })
            .collect()
    }

    pub fn calc_hz_of(&self, id: usize, relative_freq: f32) -> f32 {
        let half_sr = self.tracks.get(&id).unwrap().sr as f32 / 2.;

        match self.setting.freq_scale {
            FreqScale::Linear => half_sr * relative_freq,
            FreqScale::Mel => mel::to_hz(mel::from_hz(half_sr) * relative_freq),
        }
    }

    #[inline]
    pub fn exists(&self, &(id, ch): &(usize, usize)) -> bool {
        self.tracks.get(&id).map_or(false, |track| ch < track.n_ch)
    }

    fn calc_spec_of(&self, id: usize, ch: usize, parallel: bool) -> Array2<f32> {
        let track = self.tracks.get(&id).unwrap();
        let window = Some(CowArray::from(self.windows.get(&track.sr).unwrap().view()));
        let stft = perform_stft(
            track.get_wav(ch),
            track.win_length,
            track.hop_length,
            track.n_fft,
            window,
            None,
            parallel,
        );
        let mut linspec = stft.mapv(|x| x.norm());
        match self.setting.freq_scale {
            FreqScale::Linear => {
                linspec.amp_to_db_default();
                linspec
            }
            FreqScale::Mel => {
                let mut melspec = linspec.dot(self.mel_fbs.get(&track.sr).unwrap());
                melspec.amp_to_db_default();
                melspec
            }
        }
    }

    fn update_specs(&mut self, id_list: &[usize], new_sr_set: HashSet<(u32, usize, usize)>) {
        self.windows
            .par_extend(new_sr_set.par_iter().map(|&(sr, win_length, n_fft)| {
                (sr, calc_normalized_win(WindowType::Hann, win_length, n_fft))
            }));

        if let FreqScale::Mel = self.setting.freq_scale {
            self.mel_fbs.par_extend(
                new_sr_set
                    .par_iter()
                    .map(|&(sr, _, n_fft)| (sr, mel::calc_mel_fb_default(sr, n_fft))),
            );
        }

        let specs = {
            let id_ch_tuples = self.id_ch_tuples_from(id_list);
            let len = id_ch_tuples.len();
            let mut map = IdChMap::with_capacity(len);
            map.par_extend(
                id_ch_tuples
                    .into_par_iter()
                    .map(|(id, ch)| ((id, ch), self.calc_spec_of(id, ch, len == 1))),
            );
            map
        };

        self.specs.extend(specs);
    }

    fn update_greys(&mut self, force_update_ids: Option<&[usize]>) -> bool {
        let (mut max, mut min) = self
            .specs
            .par_iter()
            .map(|(_, spec)| {
                let max = *spec.max().unwrap_or(&-f32::INFINITY);
                let min = *spec.min().unwrap_or(&f32::INFINITY);
                (max, min)
            })
            .reduce(
                || (-f32::INFINITY, f32::INFINITY),
                |(max, min): (f32, f32), (current_max, current_min)| {
                    (max.max(current_max), min.min(current_min))
                },
            );
        max = max.min(0.);
        min = min.max(max - self.setting.db_range);
        let mut changed = false;
        if abs_diff_ne!(self.max_db, max, epsilon = 1e-3) {
            self.max_db = max;
            changed = true;
        }
        if abs_diff_ne!(self.min_db, min, epsilon = 1e-3) {
            self.min_db = min;
            changed = true;
        }

        let max_sr = self
            .tracks
            .par_iter()
            .map(|(_, track)| track.sr)
            .reduce(|| 0u32, |max, x| max.max(x));
        if self.max_sr != max_sr {
            self.max_sr = max_sr;
            changed = true;
        }

        if force_update_ids.is_some() || changed {
            let force_update_ids = force_update_ids.unwrap();
            let up_ratio_map = {
                let mut map = HashMap::<usize, f32>::with_capacity(if changed {
                    self.tracks.len()
                } else {
                    force_update_ids.len()
                });
                let iter = self.tracks.par_iter().filter_map(|(id, track)| {
                    if changed || force_update_ids.contains(id) {
                        let up_ratio =
                            calc_up_ratio(track.sr, self.max_sr, self.setting.freq_scale);
                        Some((*id, up_ratio))
                    } else {
                        None
                    }
                });
                map.par_extend(iter);
                map
            };
            let new_spec_greys = {
                let mut map = IdChMap::with_capacity(self.specs.len());
                map.par_extend(self.specs.par_iter().filter_map(|(&(id, ch), spec)| {
                    if changed || force_update_ids.contains(&id) {
                        let grey = display::convert_spec_to_grey(
                            spec.view(),
                            *up_ratio_map.get(&id).unwrap(),
                            self.max_db,
                            self.min_db,
                        );
                        Some(((id, ch), grey))
                    } else {
                        None
                    }
                }));
                map
            };

            if changed {
                self.spec_greys = new_spec_greys;
            } else {
                self.spec_greys.extend(new_spec_greys)
            }
        }
        changed
    }

    fn crop_grey_of(
        &self,
        id: usize,
        ch: usize,
        sec: f64,
        target_width: u32,
        px_per_sec: f64,
    ) -> Option<Array2<f32>> {
        let track = self.tracks.get(&id).unwrap();
        let spec_grey = self.spec_greys.get(&(id, ch)).unwrap();
        let total_width = spec_grey.shape()[1] as u64;
        let wavlen = track.wavlen() as f64;
        let sr = track.sr as u64;
        let i_w = ((total_width * sr) as f64 * sec / wavlen).round() as isize;
        let width = ((total_width * target_width as u64 * sr) as f64 / wavlen / px_per_sec)
            .max(MIN_WIDTH as f64)
            .round() as usize;
        let (i_w, width) = calc_effective_w(i_w, width, total_width as usize)?;
        let im = spec_grey.slice(s![.., i_w..i_w + width]).into_owned();
        Some(im)
    }

    fn slice_wav_of(
        &self,
        id: usize,
        ch: usize,
        sec: f64,
        width: u32,
        px_per_sec: f64,
    ) -> Option<ArrayView1<f32>> {
        let track = self.tracks.get(&id).unwrap();
        let i = (sec * track.sr as f64).round() as isize;
        let length = ((track.sr as u64 * width as u64) as f64 / px_per_sec).round() as usize;
        let (i, length) = calc_effective_w(i, length, track.wavlen())?;
        Some(track.wavs.slice(s![ch, i..i + length]))
    }
}

pub fn calc_effective_w(i_w: isize, width: usize, total_width: usize) -> Option<(usize, usize)> {
    if i_w >= total_width as isize {
        None
    } else if i_w < 0 {
        let i_right = width as isize + i_w;
        if i_right <= 0 {
            None
        } else {
            Some((0, (i_right as usize).min(total_width)))
        }
    } else {
        Some((i_w as usize, width.min(total_width - i_w as usize)))
    }
}

#[cfg(test)]
mod tests {
    use image::RgbaImage;

    use super::*;

    #[test]
    fn multitrack_works() {
        let sr_strings = ["8k", "16k", "22k05", "24k", "44k1", "48k"];
        let id_list: Vec<usize> = (0..sr_strings.len()).collect();
        let path_list: Vec<String> = sr_strings
            .iter()
            .map(|x| format!("samples/sample_{}.wav", x))
            .collect();
        let mut multitrack = TrackManager::new();
        multitrack
            .add_tracks(&id_list[0..3], path_list[0..3].to_owned())
            .unwrap();
        multitrack
            .add_tracks(&id_list[3..6], path_list[3..6].to_owned())
            .unwrap();
        dbg!(multitrack.tracks.get(&0).unwrap().path_string());
        dbg!(multitrack.tracks.get(&0).unwrap().filename());
        let width: u32 = 1500;
        let height: u32 = 500;
        id_list
            .iter()
            .zip(sr_strings.iter())
            .for_each(|(&id, &sr)| {
                let imvec = multitrack.get_spec_image_of(id, 0, width, height);
                let im =
                    RgbaImage::from_vec(imvec.len() as u32 / height / 4, height, imvec).unwrap();
                im.save(format!("samples/spec_{}.png", sr)).unwrap();
                let imvec = multitrack.get_wav_image_of(id, 0, width, height, (-1., 1.));
                let im =
                    RgbaImage::from_vec(imvec.len() as u32 / height / 4, height, imvec).unwrap();
                im.save(format!("samples/wav_{}.png", sr)).unwrap();
            });

        multitrack.remove_tracks(&[0]);
    }
}
