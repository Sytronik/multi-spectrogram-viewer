use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io;
use std::iter;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use approx::abs_diff_ne;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use rayon::prelude::*;
use realfft::{RealFftPlanner, RealToComplex};

mod audio;
mod decibel;
pub mod display;
mod mel;
mod stft;
pub mod utils;
mod windows;

use decibel::DeciBelInplace;
use stft::{calc_up_ratio, perform_stft, FreqScale};
use utils::{calc_proper_n_fft, unique_filenames};
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
    pub sample_format_str: String,
    path: PathBuf,
    wavs: Array2<f32>,
    win_length: usize,
    hop_length: usize,
    fft_module: Arc<dyn RealToComplex<f32>>,
}

impl AudioTrack {
    /// new() function gets real_fft_planner
    /// because users open new tracks of the same sample rate in most cases.
    pub fn new(
        path: String,
        setting: &SpecSetting,
        real_fft_planner: &mut RealFftPlanner<f32>,
    ) -> io::Result<Self> {
        let (wavs, sr, sample_format_str) = audio::open_audio_file(path.as_str())?;
        let n_ch = wavs.shape()[0];
        let (win_length, hop_length, n_fft) = AudioTrack::calc_framing_params(sr, setting);
        Ok(AudioTrack {
            sr,
            n_ch,
            sample_format_str,
            path: PathBuf::from(path).canonicalize()?,
            wavs,
            win_length,
            hop_length,
            fft_module: real_fft_planner.plan_fft_forward(n_fft),
        })
    }

    /// reload() function doesn't get real_fft_planner
    /// because sample rates usually don't be changed by reloading.
    pub fn reload(&mut self, setting: &SpecSetting) -> io::Result<bool> {
        let (wavs, sr, sample_format_str) =
            audio::open_audio_file(self.path.to_string_lossy().as_ref())?;
        if sr == self.sr && sample_format_str == self.sample_format_str && wavs == self.wavs {
            return Ok(false);
        }
        let (win_length, hop_length, n_fft) = AudioTrack::calc_framing_params(sr, setting);
        self.sr = sr;
        self.n_ch = wavs.shape()[0];
        self.sample_format_str = sample_format_str;
        self.wavs = wavs;
        self.win_length = win_length;
        self.hop_length = hop_length;
        if self.n_fft() != n_fft {
            self.fft_module = RealFftPlanner::<f32>::new().plan_fft_forward(n_fft);
        }
        Ok(true)
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
    pub fn wavlen(&self) -> usize {
        self.wavs.shape()[1]
    }

    #[inline]
    pub fn sec(&self) -> f64 {
        self.wavs.shape()[1] as f64 / self.sr as f64
    }

    #[inline]
    pub fn n_fft(&self) -> usize {
        self.fft_module.len()
    }

    #[inline]
    pub fn calc_width(&self, px_per_sec: f64) -> u32 {
        (px_per_sec * self.wavs.shape()[1] as f64 / self.sr as f64)
            .max(MIN_WIDTH as f64)
            .round() as u32
    }

    #[inline]
    pub fn is_path_same(&self, path: &str) -> bool {
        match PathBuf::from(path).canonicalize() {
            Ok(path_buf) => path_buf == self.path,
            Err(_) => false,
        }
    }

    fn calc_framing_params(sr: u32, setting: &SpecSetting) -> (usize, usize, usize) {
        let win_length = setting.win_ms * sr as f32 / 1000.;
        let hop_length = (win_length / setting.t_overlap as f32).round() as usize;
        let win_length = hop_length * setting.t_overlap;
        let n_fft = calc_proper_n_fft(win_length) * setting.f_overlap;
        (win_length, hop_length, n_fft)
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
            self.n_fft(),
        )
    }
}

#[readonly::make]
pub struct TrackManager {
    pub tracks: HashMap<usize, AudioTrack>,
    pub filenames: HashMap<usize, String>,
    pub max_db: f32,
    pub min_db: f32,
    pub max_sec: f64,
    pub max_sr: u32,
    setting: SpecSetting,
    windows: SrMap<Array1<f32>>,
    mel_fbs: SrMap<Array2<f32>>,
    specs: IdChMap<Array2<f32>>,
    no_grey_ids: Vec<usize>,
    spec_greys: IdChMap<Array2<f32>>,
    id_max_sec: usize,
}

impl TrackManager {
    pub fn new() -> Self {
        TrackManager {
            tracks: HashMap::new(),
            filenames: HashMap::new(),
            max_db: -f32::INFINITY,
            min_db: f32::INFINITY,
            max_sec: 0.,
            max_sr: 0,
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
            no_grey_ids: Vec::new(),
            spec_greys: HashMap::new(),
            id_max_sec: 0,
        }
    }

    pub fn add_tracks(&mut self, id_list: &[usize], path_list: Vec<String>) -> Vec<usize> {
        let mut new_sr_set = HashSet::<(u32, usize, usize)>::new();
        let mut added_ids = Vec::new();
        let mut real_fft_planner = RealFftPlanner::<f32>::new();
        for (&id, path) in id_list.iter().zip(path_list.into_iter()) {
            if let Ok(track) = AudioTrack::new(path, &self.setting, &mut real_fft_planner) {
                let sec = track.sec();
                if sec > self.max_sec {
                    self.max_sec = sec;
                    self.id_max_sec = id;
                }
                if self.windows.get(&track.sr).is_none() {
                    new_sr_set.insert((track.sr, track.win_length, track.n_fft()));
                }
                self.tracks.insert(id, track);
                added_ids.push(id);
            }
        }

        self.update_filenames();
        self.update_specs(&added_ids[..], new_sr_set);
        self.no_grey_ids.extend(added_ids.iter().cloned());
        added_ids
    }

    pub fn reload_tracks(&mut self, id_list: &[usize]) -> Vec<usize> {
        let mut new_sr_set = HashSet::<(u32, usize, usize)>::new();
        let mut reloaded_ids = Vec::new();
        for &id in id_list.iter() {
            let track = self.tracks.get_mut(&id).unwrap();
            if let Ok(true) = track.reload(&self.setting) {
                let sec = track.sec();
                if sec > self.max_sec {
                    self.max_sec = sec;
                    self.id_max_sec = id;
                }
                if self.windows.get(&track.sr).is_none() {
                    new_sr_set.insert((track.sr, track.win_length, track.n_fft()));
                }
                reloaded_ids.push(id);
            }
        }

        self.update_specs(&reloaded_ids[..], new_sr_set);
        self.no_grey_ids.extend(reloaded_ids.iter().cloned());
        reloaded_ids
    }

    pub fn remove_tracks(&mut self, id_list: &[usize]) {
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

        self.update_filenames();
    }

    pub fn update_greys(&mut self) -> HashSet<usize> {
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
        let mut has_changed_all = false;
        if abs_diff_ne!(self.max_db, max, epsilon = 1e-3) {
            self.max_db = max;
            has_changed_all = true;
        }
        if abs_diff_ne!(self.min_db, min, epsilon = 1e-3) {
            self.min_db = min;
            has_changed_all = true;
        }

        let max_sr = self
            .tracks
            .par_iter()
            .map(|(_, track)| track.sr)
            .reduce(|| 0u32, |max, x| max.max(x));
        if self.max_sr != max_sr {
            self.max_sr = max_sr;
            has_changed_all = true;
        }
        let ids_need_update: HashSet<usize> = if has_changed_all {
            self.no_grey_ids.clear();
            self.tracks.keys().cloned().collect()
        } else {
            self.no_grey_ids.drain(..).collect()
        };

        if !ids_need_update.is_empty() {
            let up_ratio_map = {
                let mut map = HashMap::<usize, f32>::with_capacity(ids_need_update.len());
                let iter = self.tracks.par_iter().filter_map(|(id, track)| {
                    if ids_need_update.contains(id) {
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
                    if ids_need_update.contains(&id) {
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

            if has_changed_all {
                self.spec_greys = new_spec_greys;
            } else {
                self.spec_greys.extend(new_spec_greys)
            }
        }
        ids_need_update
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
                        None,
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
                        option_for_wav.amp_range,
                        None,
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
                    let (part_i_w, part_width) =
                        match self.calc_part_grey_info(id, ch, sec, width, px_per_sec) {
                            Some(x) => x,
                            None => return create_empty_im_entry(),
                        };
                    let vec = display::colorize_grey_with_size(
                        self.spec_greys.get(&(id, ch)).unwrap().view(),
                        drawing_width,
                        height,
                        match fast_resize_vec {
                            Some(ref vec) => vec[i],
                            None => false,
                        },
                        Some((part_i_w, part_width)),
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
                        option_for_wav.amp_range,
                        None,
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
        let mut result = vec![0u8; width as usize * height as usize * 4];
        result[i_start..i_end]
            .par_chunks_exact_mut(ch_h as usize * width as usize * 4)
            .enumerate()
            .for_each(|(ch, x)| {
                display::draw_wav_to(&mut x[..], track.get_wav(ch), width, ch_h, (-1., 1.), None)
            });
        result
    }

    pub fn get_spec_image_of(&self, id: usize, ch: usize, width: u32, height: u32) -> Vec<u8> {
        display::colorize_grey_with_size(
            self.spec_greys.get(&(id, ch)).unwrap().view(),
            width,
            height,
            false,
            None,
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
        let mut result = vec![0u8; width as usize * height as usize * 4];
        display::draw_wav_to(
            &mut result[..],
            self.tracks.get(&id).unwrap().get_wav(ch),
            width,
            height,
            amp_range,
            None,
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

    pub fn calc_hz_of(&self, y: u32, height: u32) -> f32 {
        let half_sr = self.max_sr as f32 / 2.;
        let relative_freq = 1. - y as f32 / height as f32;

        match self.setting.freq_scale {
            FreqScale::Linear => half_sr * relative_freq,
            FreqScale::Mel => mel::to_hz(mel::from_hz(half_sr) * relative_freq),
        }
    }

    pub fn get_freq_axis(&self, max_ticks: u32) -> Vec<(f64, f64)> {
        display::create_freq_axis(self.setting.freq_scale, self.max_sr, max_ticks)
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
            track.n_fft(),
            window,
            Some(Arc::clone(&track.fft_module)),
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

    fn update_filenames(&mut self) {
        let mut paths = HashMap::<usize, PathBuf>::with_capacity(self.tracks.len());
        paths.extend(
            self.tracks
                .iter()
                .map(|(&id, track)| (id, track.path.clone())),
        );
        self.filenames = unique_filenames(paths);
    }

    fn calc_part_grey_info(
        &self,
        id: usize,
        ch: usize,
        sec: f64,
        target_width: u32,
        px_per_sec: f64,
    ) -> Option<(usize, usize)> {
        let track = self.tracks.get(&id).unwrap();
        let spec_grey = self.spec_greys.get(&(id, ch)).unwrap();
        let total_width = spec_grey.shape()[1] as u64;
        let wavlen = track.wavlen() as f64;
        let sr = track.sr as u64;
        let i_w = ((total_width * sr) as f64 * sec / wavlen).round() as isize;
        let width = ((total_width * target_width as u64 * sr) as f64 / wavlen / px_per_sec)
            .max(MIN_WIDTH as f64)
            .round() as usize;
        calc_effective_w(i_w, width, total_width as usize)
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
    fn trackmanager_works() {
        let tags = ["8k", "16k", "22k05", "24k", "44k1", "48k", "stereo_48k"];
        let id_list: Vec<usize> = (0..tags.len()).collect();
        let mut path_list: Vec<String> = tags
            .iter()
            .take(6)
            .map(|x| format!("samples/sample_{}.wav", x))
            .collect();
        path_list.push(String::from("samples/stereo/sample_48k.wav"));
        let mut tm = TrackManager::new();
        let added_ids = tm.add_tracks(&id_list[0..3], path_list[0..3].to_owned());
        assert_eq!(&added_ids[..], &id_list[0..3]);
        let added_ids = tm.add_tracks(&id_list[3..], path_list[3..].to_owned());
        assert_eq!(&added_ids[..], &id_list[3..]);
        assert_eq!(tm.tracks.len(), id_list.len());

        assert_eq!(tm.spec_greys.len(), 0);
        let mut updated_ids: Vec<usize> = tm.update_greys().into_iter().collect();
        updated_ids.sort();
        assert_eq!(updated_ids, id_list);

        dbg!(tm.filenames.get(&5).unwrap());
        dbg!(tm.filenames.get(&6).unwrap());
        let width: u32 = 1500;
        let height: u32 = 500;
        id_list.iter().zip(tags.iter()).for_each(|(&id, &sr)| {
            let imvec = tm.get_spec_image_of(id, 0, width, height);
            let im = RgbaImage::from_vec(imvec.len() as u32 / height / 4, height, imvec).unwrap();
            im.save(format!("samples/spec_{}.png", sr)).unwrap();
            let imvec = tm.get_wav_image_of(id, 0, width, height, (-1., 1.));
            let im = RgbaImage::from_vec(imvec.len() as u32 / height / 4, height, imvec).unwrap();
            im.save(format!("samples/wav_{}.png", sr)).unwrap();
        });

        tm.remove_tracks(&[0]);
        let updated_ids = tm.update_greys();
        assert!(updated_ids.is_empty());
    }
}
