use std::collections::{HashMap, HashSet};
use std::error::Error as stdError;
use std::io;
use std::iter;
use std::ops::*;
use std::path::PathBuf;

use approx::abs_diff_ne;
use ndarray::{prelude::*, ScalarOperand};
use ndarray_stats::QuantileExt;
use rayon::prelude::*;
use rustfft::{num_complex::Complex, num_traits::Float, FFTnum};

pub mod audio;
pub mod decibel;
pub mod display;
pub mod mel;
pub mod realfft;
pub mod utils;
pub mod windows;
use decibel::DeciBelInplace;
use display::GreyF32Image;
use realfft::RealFFT;
use utils::{calc_proper_n_fft, pad, PadMode};

pub enum FreqScale {
    Linear,
    Mel,
}

#[readonly::make]
pub struct AudioTrack {
    path: PathBuf,
    wavs: Array2<f32>,
    pub sr: u32,
    pub n_ch: usize,
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

    pub fn get_path(&self) -> String {
        self.path.as_path().display().to_string()
    }

    pub fn get_filename(&self) -> String {
        self.path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned()
    }

    pub fn sec(&self) -> f32 {
        self.wavs.shape()[1] as f32 / self.sr as f32
    }
}

pub struct SpecSetting {
    win_ms: f32,
    t_overlap: usize,
    f_overlap: usize,
    freq_scale: FreqScale,
    db_range: f32,
}

#[readonly::make]
pub struct TrackManager {
    pub tracks: HashMap<usize, AudioTrack>,
    setting: SpecSetting,
    windows: HashMap<u32, Array1<f32>>,
    mel_fbs: HashMap<u32, Array2<f32>>,
    specs: HashMap<(usize, usize), Array2<f32>>,
    spec_greys: HashMap<(usize, usize), GreyF32Image>,
    pub max_db: f32,
    pub min_db: f32,
    max_sec: f32,
    id_max_sec: usize,
    pub max_sr: u32,
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

    fn calc_spec_of(&self, id: usize, ch: usize, parallel: bool) -> Array2<f32> {
        let track = self.tracks.get(&id).unwrap();
        let window = Some(CowArray::from(self.windows.get(&track.sr).unwrap().view()));
        let stft = perform_stft(
            track.wavs.index_axis(Axis(0), ch),
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

    fn calc_window(win_length: usize, n_fft: usize) -> Array1<f32> {
        windows::hann(win_length, false) / n_fft as f32
    }

    fn update_specs(&mut self, id_list: &[usize], new_sr_set: HashSet<(u32, usize, usize)>) {
        let new_windows = par_collect_to_hashmap!(
            new_sr_set
                .par_iter()
                .map(|&(sr, win_length, n_fft)| (sr, TrackManager::calc_window(win_length, n_fft))),
            new_sr_set.len()
        );
        self.windows.extend(new_windows);

        if let FreqScale::Mel = self.setting.freq_scale {
            let mel_fbs = par_collect_to_hashmap!(
                new_sr_set
                    .par_iter()
                    .map(|&(sr, _, n_fft)| (sr, mel::calc_mel_fb_default(sr, n_fft))),
                new_sr_set.len()
            );
            self.mel_fbs.extend(mel_fbs);
        }

        let specs = par_collect_to_hashmap!(
            id_list
                .par_iter()
                .flat_map_iter(|id| {
                    iter::repeat(id)
                        .take(self.tracks.get(id).unwrap().n_ch)
                        .enumerate()
                })
                .map(|(ch, &id)| ((id, ch), self.calc_spec_of(id, ch, id_list.len() == 1))),
            id_list.len()
        );
        self.specs.extend(specs);
    }

    pub fn add_tracks(&mut self, id_list: Vec<usize>, path_list: Vec<String>) -> io::Result<bool> {
        let mut new_sr_set = HashSet::<(u32, usize, usize)>::new();
        for (&id, path) in id_list.iter().zip(path_list.into_iter()) {
            let track = AudioTrack::new(path, &self.setting)?;
            let sec = track.sec();
            if sec > self.max_sec {
                self.max_sec = sec;
                self.id_max_sec = id;
            }
            if let None = self.windows.get(&track.sr) {
                new_sr_set.insert((track.sr, track.win_length, track.n_fft));
            }
            self.tracks.insert(id, track);
        }

        self.update_specs(id_list.as_slice(), new_sr_set);
        Ok(self.update_spec_greys(Some(id_list.as_slice())))
    }

    fn update_spec_greys(&mut self, force_update_ids: Option<&[usize]>) -> bool {
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
            let up_ratio = match self.setting.freq_scale {
                FreqScale::Linear => par_collect_to_hashmap!(
                    self.tracks.par_iter().filter_map(|(&id, track)| {
                        if changed || force_update_ids.contains(&id) {
                            Some((id, self.max_sr as f32 / track.sr as f32))
                        } else {
                            None
                        }
                    }),
                    self.tracks.len()
                ),
                FreqScale::Mel => par_collect_to_hashmap!(
                    self.tracks.par_iter().filter_map(|(&id, track)| {
                        if changed || force_update_ids.contains(&id) {
                            Some((
                                id,
                                mel::hz_to_mel(self.max_sr as f32 / 2.)
                                    / mel::hz_to_mel(track.sr as f32 / 2.),
                            ))
                        } else {
                            None
                        }
                    }),
                    self.tracks.len()
                ),
            };
            let new_spec_greys = par_collect_to_hashmap!(
                self.specs.par_iter().filter_map(|(&(id, ch), spec)| {
                    if changed || force_update_ids.contains(&id) {
                        let grey = display::spec_to_grey(
                            spec.view(),
                            *up_ratio.get(&id).unwrap(),
                            self.max_db,
                            self.min_db,
                        );
                        Some(((id, ch), grey))
                    } else {
                        None
                    }
                }),
                self.specs.len()
            );
            if changed {
                self.spec_greys = new_spec_greys;
            } else {
                self.spec_greys.extend(new_spec_greys)
            }
        }
        changed
    }

    pub fn remove_track(&mut self, id: usize) -> bool {
        let removed_track = self.tracks.remove_entry(&id).unwrap().1;
        for ch in (0..removed_track.n_ch).into_iter() {
            self.specs.remove(&(id, ch));
            self.spec_greys.remove(&(id, ch));
        }
        if self.id_max_sec == id {
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
        if self
            .tracks
            .par_iter()
            .all(|(_, track)| track.sr != removed_track.sr)
        {
            self.windows.remove(&removed_track.sr);
            self.mel_fbs.remove(&removed_track.sr);
        }
        self.update_spec_greys(None)
    }

    pub fn get_spec_wav_image(
        &self,
        output: &mut [u8],
        id: usize,
        ch: usize,
        width: u32,
        height: u32,
        blend: f64,
    ) -> Result<(), Box<dyn stdError>> {
        display::blend_spec_wav(
            output,
            self.spec_greys.get(&(id, ch)).unwrap(),
            self.tracks.get(&id).unwrap().wavs.index_axis(Axis(0), ch),
            width,
            height,
            blend,
        )
    }

    pub fn get_spec_image(&self, output: &mut [u8], id: usize, ch: usize, width: u32, height: u32) {
        display::grey_to_rgb(
            output,
            self.spec_greys.get(&(id, ch)).unwrap(),
            width,
            height,
        );
    }

    pub fn get_wav_image(
        &self,
        output: &mut [u8],
        id: usize,
        ch: usize,
        width: u32,
        height: u32,
        amp_range: (f32, f32),
    ) {
        let track = self.tracks.get(&id).unwrap();
        display::draw_wav(
            output,
            track.wavs.index_axis(Axis(0), ch),
            width,
            height,
            255,
            amp_range,
        )
    }

    pub fn get_frequency_hz(&self, id: usize, relative_freq: f32) -> f32 {
        let half_sr = self.tracks.get(&id).unwrap().sr as f32 / 2.;

        match self.setting.freq_scale {
            FreqScale::Linear => half_sr * relative_freq,
            FreqScale::Mel => mel::mel_to_hz(mel::hz_to_mel(half_sr) * relative_freq),
        }
    }
}

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
    A: FFTnum + Float + DivAssign + ScalarOperand,
{
    let n_pad_left = (n_fft - win_length) / 2;
    let n_pad_right = (((n_fft - win_length) as f32) / 2.).ceil() as usize;

    let window = if let Some(w) = window {
        assert_eq!(w.len(), win_length);
        w
    } else {
        CowArray::from(windows::hann(win_length, false) / A::from(n_fft).unwrap())
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

    let mut first_idx = front_frames.len() * hop_length - win_length / 2;
    let mut frames: Vec<Array1<A>> = to_frames_wrapper(input.slice(s![first_idx..]));

    first_idx += frames.len() * hop_length;
    let back_wav_start_idx = first_idx.min(input.len() - win_length / 2 - 1);

    let mut back_wav = pad(
        input.slice(s![back_wav_start_idx..]),
        (0, win_length / 2),
        Axis(0),
        PadMode::Reflect,
    );
    back_wav.slice_collapse(s![(first_idx - back_wav_start_idx).max(0)..]);
    let mut back_frames = to_frames_wrapper(back_wav.view());

    let n_frames = front_frames.len() + frames.len() + back_frames.len();
    let mut output = Array2::<Complex<A>>::zeros((n_frames, n_fft / 2 + 1));
    let out_frames: Vec<&mut [Complex<A>]> = output
        .axis_iter_mut(Axis(0))
        .map(|x| x.into_slice().unwrap())
        .collect();

    let mut new_module;
    let fft_module = if let Some(m) = fft_module {
        m
    } else {
        new_module = RealFFT::<A>::new(n_fft).unwrap();
        &mut new_module
    };
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

pub fn get_colormap_iter_size() -> (impl Iterator<Item = &'static u8>, usize) {
    (
        display::COLORMAP.iter().flat_map(|x| x.iter()),
        display::COLORMAP.len() * display::COLORMAP[0].len(),
    )
}

#[cfg(test)]
mod tests {
    use image::RgbaImage;
    use ndarray::{arr2, Array1};
    use rustfft::num_complex::Complex;

    use super::utils::Impulse;
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

    #[test]
    fn multitrack_works() {
        let sr_strings = ["8k", "16k", "22k05", "24k", "44k1", "48k"];
        let id_list: Vec<usize> = (0..sr_strings.len()).collect();
        let path_list: Vec<String> = sr_strings
            .iter()
            .map(|x| format!("../../samples/sample_{}.wav", x))
            .collect();
        let mut multitrack = TrackManager::new();
        multitrack
            .add_tracks(id_list[0..3].to_owned(), path_list[0..3].to_owned())
            .unwrap();
        multitrack
            .add_tracks(id_list[3..6].to_owned(), path_list[3..6].to_owned())
            .unwrap();
        dbg!(multitrack.tracks.get(&0).unwrap().get_path());
        dbg!(multitrack.tracks.get(&0).unwrap().get_filename());
        let width: u32 = 1500;
        let height: u32 = 500;
        id_list
            .iter()
            .zip(sr_strings.iter())
            .for_each(|(&id, &sr)| {
                let mut imvec = vec![0u8; (4 * width * height) as usize];
                multitrack.get_spec_image(imvec.as_mut_slice(), id, width, height);
                let im =
                    RgbaImage::from_vec(imvec.len() as u32 / height / 4, height, imvec).unwrap();
                im.save(format!("../../samples/spec_{}.png", sr)).unwrap();
                let mut imvec = vec![0u8; (4 * width * height) as usize];
                multitrack.get_wav_image(imvec.as_mut_slice(), id, width, height, (-1., 1.));
                let im =
                    RgbaImage::from_vec(imvec.len() as u32 / height / 4, height, imvec).unwrap();
                im.save(format!("../../samples/wav_{}.png", sr)).unwrap();
            });

        multitrack.remove_track(0);
    }
}
