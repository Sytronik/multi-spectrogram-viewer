#![deny(clippy::all)]

use std::convert::TryInto;
use std::sync::{RwLock, RwLockReadGuard};
use std::time::Instant;

use napi::{
    CallContext, ContextlessResult, Env, JsBuffer, JsNumber, JsObject, JsString,
    Result as JsResult, Task,
};
use napi_derive::*;

use lazy_static::{initialize, lazy_static};
use ndarray::{Array3, Axis, Slice};
use rayon::prelude::*;

mod backend;
mod napi_utils;

use backend::*;
use napi_utils::*;

#[cfg(all(unix, not(target_env = "musl"), not(target_arch = "aarch64")))]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[cfg(all(windows, target_arch = "x86_64"))]
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

lazy_static! {
    static ref TM: RwLock<TrackManager> = RwLock::new(TrackManager::new());

    static ref DRAWOPTION: RwLock<DrawOption> = RwLock::new(DrawOption {
        px_per_sec: 0.,
        height: 1,
    });
    static ref DRAWOPTION_FOR_WAV: RwLock<DrawOptionForWav> = RwLock::new(DrawOptionForWav {
        amp_range: (-1., 1.)
    });

    // to ensure only one task exsits
    static ref SPEC_TASK_EXISTS: RwLock<bool> = RwLock::new(false);
    static ref WAV_TASK_EXISTS: RwLock<bool> = RwLock::new(false);

    // image caches
    static ref SPEC_IMAGES: RwLock<IdChMap<Array3<u8>>> = RwLock::new(IdChMap::new());
    static ref WAV_IMAGES: RwLock<IdChMap<Array3<u8>>> = RwLock::new(IdChMap::new());
}

#[js_function(2)]
fn add_tracks(ctx: CallContext) -> JsResult<JsObject> {
    let new_track_ids: Vec<usize> = vec_usize_from(&ctx, 0)?;
    let new_paths: Vec<String> = vec_str_from(&ctx, 1)?;
    let mut tm = TM.write().unwrap();
    match tm.add_tracks(&new_track_ids[..], new_paths) {
        Ok(should_draw_all) => {
            let tuples = if should_draw_all {
                tm.id_ch_tuples()
            } else {
                tm.id_ch_tuples_from(&new_track_ids[..])
            };
            let task = DrawingTask {
                id_ch_tuples_spec: tuples.clone(),
                id_ch_tuples_wav: tuples,
                option: *DRAWOPTION.read().unwrap(),
                opt_for_wav: *DRAWOPTION_FOR_WAV.read().unwrap(),
            };
            ctx.env
                .spawn(task)
                .map(|async_task| async_task.promise_object())
        }
        Err(_) => Err(ctx
            .env
            .throw_error("Unsupported file type!", None)
            .err()
            .unwrap()),
    }
}

#[js_function(1)]
fn remove_tracks(ctx: CallContext) -> JsResult<JsObject> {
    let track_ids: Vec<usize> = vec_usize_from(&ctx, 0)?;
    let mut tm = TM.write().unwrap();
    {
        let mut spec_images = SPEC_IMAGES.write().unwrap();
        let mut wav_images = WAV_IMAGES.write().unwrap();
        for tup in tm.id_ch_tuples_from(&track_ids[..]).iter() {
            spec_images.remove(tup);
            wav_images.remove(tup);
        }
    }
    if tm.remove_tracks(&track_ids[..]) {
        let tuples = tm.id_ch_tuples();
        let task = DrawingTask {
            id_ch_tuples_spec: tuples.clone(),
            id_ch_tuples_wav: tuples,
            option: *DRAWOPTION.read().unwrap(),
            opt_for_wav: *DRAWOPTION_FOR_WAV.read().unwrap(),
        };
        ctx.env
            .spawn(task)
            .map(|async_task| async_task.promise_object())
    } else {
        ctx.env.create_object()
    }
}

#[js_function(5)]
fn get_spec_wav_images(ctx: CallContext) -> JsResult<JsObject> {
    // let start = Instant::now();
    let id_ch_tuples = id_ch_tuples_from(&ctx, 0)?;
    let sec: f64 = ctx.get::<JsNumber>(1)?.try_into()?;
    let width: u32 = ctx.get::<JsNumber>(2)?.try_into()?;
    assert!(width >= 1);
    let option = draw_option_from_js_obj(ctx.get::<JsObject>(3)?)?;
    assert!(option.height >= 1);
    let opt_for_wav = draw_opt_for_wav_from_js_obj(ctx.get::<JsObject>(4)?)?;
    assert!(opt_for_wav.amp_range.1 >= opt_for_wav.amp_range.0);

    let mut result = ctx.env.create_array_with_length(2)?;
    let mut result_im = ctx.env.create_object()?;

    let tm = TM.read().unwrap();

    // Categorize id_ch_tuples
    // 1. DrawingTask is needed, 2. drawing part of image is needed, 3. Cached image is used
    let id_ch_tuples: IdChVec = id_ch_tuples.into_iter().filter(|x| tm.exists(x)).collect();
    let mut total_widths = IdChMap::<u32>::with_capacity(id_ch_tuples.len());
    total_widths.extend(id_ch_tuples.iter().map(|&(id, ch)| {
        (
            (id, ch),
            tm.tracks.get(&id).unwrap().calc_width(option.px_per_sec),
        )
    }));
    let (need_spec_task, need_spec_parts, use_spec_caches) = categorize_id_ch(
        &id_ch_tuples[..],
        SPEC_TASK_EXISTS.read().unwrap(),
        SPEC_IMAGES.read().unwrap(),
        &total_widths,
        option,
        ImageKind::Spec,
    );
    let (need_wav_task, need_wav_parts, use_wav_caches) = categorize_id_ch(
        &id_ch_tuples[..],
        WAV_TASK_EXISTS.read().unwrap(),
        WAV_IMAGES.read().unwrap(),
        &total_widths,
        option,
        ImageKind::Wav(opt_for_wav),
    );

    // spawn DrawingTask
    let need_task = !need_spec_task.is_empty() || !need_wav_task.is_empty();
    // let need_task = false;
    if need_task {
        let task = DrawingTask {
            id_ch_tuples_spec: need_spec_task.clone(),
            id_ch_tuples_wav: need_wav_task.clone(),
            option,
            opt_for_wav,
        };
        result.set_element(
            1,
            ctx.env
                .spawn(task)
                .map(|async_task| async_task.promise_object())?,
        )?;
    } else {
        result.set_element(1, ctx.env.get_null()?)?;
    };

    // draw part
    if !need_spec_parts.is_empty() {
        let fast_resize_vec = if option.height <= display::MAX_SIZE {
            Some(
                need_spec_parts
                    .iter()
                    .map(|tup| *total_widths.get(tup).unwrap() <= display::MAX_SIZE)
                    .collect(),
            )
        } else {
            None
        };
        let spec_images = TM.read().unwrap().get_part_images(
            &need_spec_parts[..],
            sec,
            width,
            option,
            ImageKind::Spec,
            fast_resize_vec,
        );
        set_images_to(ctx.env, &mut result_im, spec_images, 0)?;
    }
    if !need_wav_parts.is_empty() {
        let wav_images = TM.read().unwrap().get_part_images(
            &need_wav_parts[..],
            sec,
            width,
            option,
            ImageKind::Wav(opt_for_wav),
            None,
        );
        set_images_to(ctx.env, &mut result_im, wav_images, 1)?;
    }

    // crop image cache
    if !use_spec_caches.is_empty() {
        let spec_images = crop_cached_images_(
            SPEC_IMAGES.read().unwrap(),
            &use_spec_caches[..],
            sec,
            width,
            option,
        );
        set_images_to(ctx.env, &mut result_im, spec_images, 0)?;
    }
    if !use_wav_caches.is_empty() {
        let wav_images = crop_cached_images_(
            WAV_IMAGES.read().unwrap(),
            &use_wav_caches[..],
            sec,
            width,
            option,
        );
        set_images_to(ctx.env, &mut result_im, wav_images, 1)?;
    }

    result.set_element(0, result_im)?;
    Ok(result)
}

#[js_function(1)]
fn get_overview(ctx: CallContext) -> JsResult<JsBuffer> {
    let id: usize = ctx.get::<JsNumber>(0)?.try_into_usize()?;
    let width: u32 = ctx.get::<JsNumber>(1)?.try_into()?;
    let height: u32 = ctx.get::<JsNumber>(2)?.try_into()?;
    let tm = TM.read().unwrap();
    ctx.env
        .create_buffer_with_data(tm.get_overview_of(id, width, height))
        .map(|x| x.into_raw())
}

#[js_function(2)]
fn get_hz_at(ctx: CallContext) -> JsResult<JsNumber> {
    let y: u32 = ctx.get::<JsNumber>(0)?.try_into()?;
    let height: u32 = ctx.get::<JsNumber>(1)?.try_into()?;
    let tm = TM.read().unwrap();
    ctx.env.create_double(tm.calc_hz_of(y, height) as f64)
}

#[contextless_function]
fn get_max_db(env: Env) -> ContextlessResult<JsNumber> {
    env.create_double(TM.read().unwrap().max_db as f64)
        .map(Some)
}

#[contextless_function]
fn get_min_db(env: Env) -> ContextlessResult<JsNumber> {
    env.create_double(TM.read().unwrap().min_db as f64)
        .map(Some)
}

#[js_function(1)]
fn get_n_ch(ctx: CallContext) -> JsResult<JsNumber> {
    let tm = TM.read().unwrap();
    let track = get_track!(ctx, 0, tm);
    ctx.env.create_uint32(track.n_ch as u32)
}

#[js_function(1)]
fn get_sec(ctx: CallContext) -> JsResult<JsNumber> {
    let tm = TM.read().unwrap();
    let track = get_track!(ctx, 0, tm);
    ctx.env.create_double(track.sec())
}

#[js_function(1)]
fn get_sr(ctx: CallContext) -> JsResult<JsNumber> {
    let tm = TM.read().unwrap();
    let track = get_track!(ctx, 0, tm);
    ctx.env.create_uint32(track.sr)
}

#[js_function(1)]
fn get_path(ctx: CallContext) -> JsResult<JsString> {
    let tm = TM.read().unwrap();
    let track = get_track!(ctx, 0, tm);
    ctx.env.create_string_from_std(track.path_string())
}

#[js_function(1)]
fn get_filename(ctx: CallContext) -> JsResult<JsString> {
    let tm = TM.read().unwrap();
    let track = get_track!(ctx, 0, tm);
    ctx.env.create_string_from_std(track.filename())
}

#[contextless_function]
fn get_colormap(env: Env) -> ContextlessResult<JsBuffer> {
    env.create_buffer_with_data(display::get_colormap_rgba())
        .map(|x| Some(x.into_raw()))
}

struct DrawingTask {
    id_ch_tuples_spec: IdChVec,
    id_ch_tuples_wav: IdChVec,
    option: DrawOption,
    opt_for_wav: DrawOptionForWav,
}

impl Task for DrawingTask {
    type Output = [IdChSet; 3];
    type JsValue = JsObject;

    fn compute(&mut self) -> JsResult<Self::Output> {
        let _ = TM.read().unwrap();
        if !self.id_ch_tuples_spec.is_empty() {
            *SPEC_TASK_EXISTS.write().unwrap() = true;
        }
        if !self.id_ch_tuples_spec.is_empty() {
            *WAV_TASK_EXISTS.write().unwrap() = true;
        }
        let new_spec_images = TM.read().unwrap().get_entire_images(
            &self.id_ch_tuples_spec[..],
            self.option,
            ImageKind::Spec,
        );
        let new_wav_images = TM.read().unwrap().get_entire_images(
            &self.id_ch_tuples_wav[..],
            self.option,
            ImageKind::Wav(self.opt_for_wav),
        );
        let mut for_spec = new_spec_images.keys().cloned().collect();
        let mut for_wav = new_wav_images.keys().cloned().collect();
        let for_both = extract_intersect(&mut for_spec, &mut for_wav);

        let mut spec_images = SPEC_IMAGES.write().unwrap();
        let mut wav_images = WAV_IMAGES.write().unwrap();
        let mut option = DRAWOPTION.write().unwrap();
        let mut opt_for_wav = DRAWOPTION_FOR_WAV.write().unwrap();
        if self.option != *option {
            spec_images.clear();
            wav_images.clear();
            *option = self.option;
        }
        if self.opt_for_wav != *opt_for_wav {
            wav_images.clear();
            *opt_for_wav = self.opt_for_wav;
        }
        spec_images.par_extend(new_spec_images.into_par_iter());
        wav_images.par_extend(new_wav_images.into_par_iter());
        *SPEC_TASK_EXISTS.write().unwrap() = false;
        *WAV_TASK_EXISTS.write().unwrap() = false;
        Ok([for_both, for_spec, for_wav])
    }

    fn resolve(self, env: Env, output: Self::Output) -> JsResult<Self::JsValue> {
        let mut arr = env.create_array_with_length(3)?;
        for (i, set) in output.iter().enumerate() {
            arr.set_element(
                i as u32,
                convert_id_ch_vec_to_jsarr(&env, set.iter(), set.len())?,
            )?;
        }
        Ok(arr)
    }
}

fn crop_cached_images_(
    images: RwLockReadGuard<IdChMap<Array3<u8>>>,
    id_ch_tuples: &IdChArr,
    sec: f64,
    width: u32,
    option: DrawOption,
) -> IdChMap<Vec<u8>> {
    let start = Instant::now();
    let mut result = IdChMap::new();
    let par_iter = id_ch_tuples.par_iter().map(|tup| {
        let image = images.get(&tup).unwrap();
        let total_width = image.len() / 4 / option.height as usize;
        let i_w = (sec * option.px_per_sec) as isize;
        let (i_w_eff, width_eff) = match calc_effective_w(i_w, width as usize, total_width) {
            Some((i, w)) => (i as isize, w as isize),
            None => return (*tup, vec![0u8; width as usize * option.height as usize * 4]),
        };
        let slice = Slice::from(i_w_eff..i_w_eff + width_eff);
        let im_slice = image.slice_axis(Axis(1), slice);

        let pad_left = (-i_w.min(0)) as usize;
        let pad_right = width as usize - width_eff as usize - pad_left;
        if pad_left + pad_right == 0 {
            (*tup, im_slice.to_owned().into_raw_vec())
        } else {
            let arr = utils::pad(
                im_slice,
                (pad_left, pad_right),
                Axis(1),
                utils::PadMode::Constant(0),
            );
            (*tup, arr.into_raw_vec())
        }
    });
    result.par_extend(par_iter);
    println!("crop: {:?}", start.elapsed());
    result
}

fn categorize_id_ch(
    id_ch_tuples: &IdChArr,
    task_exists: RwLockReadGuard<bool>,
    images: RwLockReadGuard<IdChMap<Array3<u8>>>,
    total_widths: &IdChMap<u32>,
    option: DrawOption,
    kind: ImageKind,
) -> (IdChVec, IdChVec, IdChVec) {
    let kind_condition = match kind {
        ImageKind::Spec => true,
        ImageKind::Wav(opt_for_wav) => *DRAWOPTION_FOR_WAV.read().unwrap() == opt_for_wav,
    };
    if !*task_exists && option.height <= display::MAX_SIZE {
        let mut need_task = IdChVec::new();
        let mut need_parts = IdChVec::new();
        let mut use_caches = IdChVec::new();
        if *DRAWOPTION.read().unwrap() == option && kind_condition {
            for tup in id_ch_tuples.iter() {
                let not_long_w = *total_widths.get(tup).unwrap() <= display::MAX_SIZE;
                match (images.contains_key(tup), not_long_w) {
                    (true, _) => use_caches.push(*tup),
                    (false, true) => {
                        need_task.push(*tup);
                        need_parts.push(*tup);
                    }
                    (false, false) => {
                        need_parts.push(*tup);
                    }
                }
            }
        } else {
            for tup in id_ch_tuples.iter() {
                need_parts.push(*tup);
                if *total_widths.get(tup).unwrap() <= display::MAX_SIZE {
                    need_task.push(*tup)
                }
            }
        }
        (need_task, need_parts, use_caches)
    } else {
        (Vec::new(), id_ch_tuples.to_owned(), Vec::new())
    }
}

#[module_exports]
fn init(mut exports: JsObject) -> JsResult<()> {
    initialize(&TM);
    initialize(&DRAWOPTION);
    initialize(&DRAWOPTION_FOR_WAV);
    initialize(&SPEC_TASK_EXISTS);
    initialize(&WAV_TASK_EXISTS);
    initialize(&SPEC_IMAGES);
    initialize(&WAV_IMAGES);

    exports.create_named_method("addTracks", add_tracks)?;
    exports.create_named_method("removeTracks", remove_tracks)?;
    exports.create_named_method("getSpecWavImages", get_spec_wav_images)?;
    exports.create_named_method("getOverview", get_overview)?;
    exports.create_named_method("getHzAt", get_hz_at)?;
    exports.create_named_method("getMaxdB", get_max_db)?;
    exports.create_named_method("getMindB", get_min_db)?;
    exports.create_named_method("getNumCh", get_n_ch)?;
    exports.create_named_method("getSec", get_sec)?;
    exports.create_named_method("getSr", get_sr)?;
    exports.create_named_method("getPath", get_path)?;
    exports.create_named_method("getFileName", get_filename)?;
    exports.create_named_method("getColormap", get_colormap)?;
    Ok(())
}
