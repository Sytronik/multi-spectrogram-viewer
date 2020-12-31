use std::sync::{RwLock, RwLockReadGuard};
use std::time::Instant;

use lazy_static::{initialize, lazy_static};
use ndarray::{Array3, ArrayViewMut3, Axis, Slice};
use neon::prelude::*;
use rayon::prelude::*;

use thesia_backend::*;

const MAX_IMAGE_SIZE: u32 = 8192;

lazy_static! {
    static ref TM: RwLock<TrackManager> = RwLock::new(TrackManager::new());
    static ref DRAWOPTION: RwLock<DrawOption> = RwLock::new(DrawOption {
        px_per_sec: 0.,
        height: 1,
        blend: 0.,
    });
    static ref IMAGES: RwLock<IdChMap<Array3<u8>>> = RwLock::new(IdChMap::new());
}

macro_rules! get_track {
    ($locked:expr, $cx:expr, $i_arg_id:expr) => {
        match $locked
            .tracks
            .get(&($cx.argument::<JsNumber>($i_arg_id)?.value() as usize))
        {
            Some(t) => t,
            None => return $cx.throw_error("Wrong track id!"),
        }
    };
}

macro_rules! get_num_arg {
    ($cx:expr, $i_arg:expr $(, $type:ty)?) => {
        $cx.argument::<JsNumber>($i_arg)?.value() $(as $type)?
    };
}

macro_rules! get_num_field {
    ($obj:expr, $cx:expr, $key:expr $(, $type:ty)?) => {
        $obj.get(&mut $cx, $key)?.downcast::<JsNumber>().unwrap().value() $(as $type)?
    };
}

macro_rules! get_arr_arg {
    ($cx:expr, $i_arg:expr, JsNumber $(, $type:ty)?) => {
        $cx.argument::<JsArray>($i_arg)?
            .to_vec(&mut $cx)?
            .into_iter()
            .map(|jsv| {
                jsv.downcast::<JsNumber>()
                    .unwrap_or($cx.number(0.))
                    .value() $(as $type)?
            })
            .collect()
    };
    ($cx:expr, $i_arg:expr, JsNumber, $default:expr $(, $type:ty)?) => {
        $cx.argument::<JsArray>($i_arg)?
            .to_vec(&mut $cx)?
            .into_iter()
            .map(|jsv| {
                jsv.downcast::<JsNumber>()
                    .unwrap_or($cx.number($default))
                    .value() $(as $type)?
            })
            .collect()
    };
    ($cx:expr, $i_arg:expr, $js_type:ident, $default:expr) => {
        $cx.argument::<JsArray>($i_arg)?
            .to_vec(&mut $cx)?
            .into_iter()
            .map(|jsv| {
                jsv.downcast::<$js_type>()
                    .unwrap_or($js_type::new(&mut $cx, $default))
                    .value()
            })
            .collect()
    };
}

fn add_tracks(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let new_track_ids: Vec<usize> = get_arr_arg!(cx, 0, JsNumber, usize);
    let new_paths: Vec<String> = get_arr_arg!(cx, 1, JsString, "");
    let callback = cx.argument::<JsFunction>(2)?;
    let mut tm = TM.write().unwrap();
    match tm.add_tracks(new_track_ids.as_slice(), new_paths) {
        Ok(should_draw_all) => {
            RenderingTask {
                id_ch_tuples: if should_draw_all {
                    tm.get_all_id_ch()
                } else {
                    tm.get_id_ch_tuples_from(new_track_ids.as_slice())
                },
                option: *DRAWOPTION.read().unwrap(),
            }
            .schedule(callback);
            Ok(cx.undefined())
        }
        Err(_) => cx.throw_error("Unsupported file type!"),
    }
}

fn remove_track(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let track_id = get_num_arg!(cx, 0, usize);
    let callback = cx.argument::<JsFunction>(1)?;
    let mut tm = TM.write().unwrap();
    let mut images = IMAGES.write().unwrap();
    let id_ch_tuples = tm.get_id_ch_tuples_from(&[track_id]);
    for tup in id_ch_tuples.iter() {
        images.remove(tup);
    }
    if tm.remove_track(track_id) {
        RenderingTask {
            id_ch_tuples: tm.get_all_id_ch(),
            option: *DRAWOPTION.read().unwrap(),
        }
        .schedule(callback);
    }
    Ok(cx.undefined())
}

fn _crop_high_q_images<'a>(
    cx: &mut FunctionContext<'a>,
    images: &RwLockReadGuard<IdChMap<Array3<u8>>>,
    sec: f64,
    width: u32,
    option: DrawOption,
) -> JsResult<'a, JsObject> {
    let obj = JsObject::new(cx);
    let id_ch_tuples: IdChVec = images.keys().cloned().collect();
    let arr = JsArray::new(cx, images.len() as u32);
    for (i, (id, ch)) in id_ch_tuples.iter().enumerate() {
        let id_ch_jsstr = cx.string(format!("{}_{}", id, ch));
        arr.set(cx, i as u32, id_ch_jsstr)?;
    }
    let mut buf = JsArrayBuffer::new(cx, images.len() as u32 * width * option.height * 4)?;
    cx.borrow_mut(&mut buf, |borrowed| {
        let chunk_iter = borrowed
            .as_mut_slice()
            .par_chunks_exact_mut((width * option.height * 4) as usize);

        id_ch_tuples
            .into_par_iter()
            .zip(chunk_iter)
            .for_each(|(id_ch, output)| {
                let image = images.get(&id_ch).unwrap();
                let total_width = image.len() / 4 / option.height as usize;
                let i_w = (sec * option.px_per_sec) as isize;
                let (i_w_eff, width_eff) = match calc_effective_w(i_w, width as usize, total_width)
                {
                    Some((i, w)) => (i as isize, w as isize),
                    None => return,
                };
                let slice = Slice::new(i_w_eff, Some(i_w_eff + width_eff), 1);

                let shape = (option.height as usize, width as usize, 4);
                let mut out_view = ArrayViewMut3::from_shape(shape, output).unwrap();
                image
                    .slice_axis(Axis(1), slice)
                    .indexed_iter()
                    .for_each(|((h, w, i), x)| {
                        out_view[[h, (w as isize - i_w.min(0)) as usize, i]] = *x;
                    });
            });
    });
    obj.set(cx, "id_ch_arr", arr)?;
    obj.set(cx, "buf", buf)?;
    Ok(obj)
}

fn _get_low_q_images<'a>(
    cx: &mut FunctionContext<'a>,
    sec: f64,
    width: u32,
    option: DrawOption,
    fast_resize: bool,
) -> JsResult<'a, JsObject> {
    let obj = JsObject::new(cx);
    let tm = TM.read().unwrap();
    let id_ch_tuples = tm.get_all_id_ch();
    let arr = JsArray::new(cx, id_ch_tuples.len() as u32);
    for (i, (id, ch)) in id_ch_tuples.iter().enumerate() {
        let id_ch_jsstr = cx.string(format!("{}_{}", id, ch));
        arr.set(cx, i as u32, id_ch_jsstr)?;
    }
    let mut buf = JsArrayBuffer::new(cx, id_ch_tuples.len() as u32 * width * option.height * 4)?;
    cx.borrow_mut(&mut buf, |borrowed| {
        tm.draw_image_parts_to(borrowed.as_mut_slice(), sec, width, option, fast_resize);
    });
    obj.set(cx, "id_ch_arr", arr)?;
    obj.set(cx, "buf", buf)?;
    Ok(obj)
}

fn get_images(mut cx: FunctionContext) -> JsResult<JsObject> {
    let sec = get_num_arg!(cx, 0);
    let width = get_num_arg!(cx, 1, u32);
    let option = {
        let object = cx.argument::<JsObject>(2)?;
        let px_per_sec = get_num_field!(object, cx, "px_per_sec");
        let height = get_num_field!(object, cx, "height", u32);
        let blend = get_num_field!(object, cx, "blend");
        DrawOption {
            px_per_sec,
            height,
            blend,
        }
    };
    let callback = cx.argument::<JsFunction>(3)?;

    // dbg!(sec, width, &option);
    let same_option = option == *DRAWOPTION.read().unwrap();
    let total_width = option.px_per_sec * TM.read().unwrap().max_sec;
    let too_large = total_width > MAX_IMAGE_SIZE as f64 || option.height > MAX_IMAGE_SIZE;
    if same_option && !too_large && IMAGES.try_read().is_ok() {
        let images = IMAGES.read().unwrap();
        let start = Instant::now();
        let obj = _crop_high_q_images(&mut cx, &images, sec, width, option);
        println!("Copy high q: {:?}", start.elapsed());
        obj
    } else {
        let start = Instant::now();
        if !same_option && !too_large {
            RenderingTask {
                id_ch_tuples: TM.read().unwrap().get_all_id_ch(),
                option,
            }
            .schedule(callback);
        }
        if let Ok(mut images) = IMAGES.try_write() {
            if too_large && !images.is_empty() {
                images.clear();
            }
        }
        let obj = _get_low_q_images(&mut cx, sec, width, option, !too_large);
        {
            let high_or_low = if too_large { "high" } else { "low" };
            println!("Draw {} q: {:?}", high_or_low, start.elapsed());
        }
        obj
    }
}
#[derive(Clone)]
struct RenderingTask {
    id_ch_tuples: IdChVec,
    option: DrawOption,
}

impl Task for RenderingTask {
    type Output = Option<IdChVec>;
    type Error = ();
    type JsEvent = JsArray;

    fn perform(&self) -> Result<Self::Output, Self::Error> {
        if *DRAWOPTION.read().unwrap() == self.option {
            Ok(None)
        } else if let Ok(mut images) = IMAGES.try_write() {
            let new = {
                let tm = TM.read().unwrap();
                tm.get_entire_images(self.id_ch_tuples.as_slice(), self.option)
            };
            let id_ch_tuples = new.keys().map(|&tup| tup).collect();
            // while (true) {}
            images.extend(new);
            *DRAWOPTION.write().unwrap() = self.option;
            Ok(Some(id_ch_tuples))
        } else {
            Ok(None)
        }
    }

    fn complete<'a>(
        self,
        mut cx: TaskContext<'a>,
        id_ch_tuples: Result<Self::Output, Self::Error>,
    ) -> JsResult<Self::JsEvent> {
        match id_ch_tuples {
            // Ok(Some(tuples)) => {
            Ok(Some(tuples)) if self.option == *DRAWOPTION.read().unwrap() => {
                let arr = JsArray::new(&mut cx, tuples.len() as u32);
                for (i, &(id, ch)) in tuples.iter().enumerate() {
                    let id_ch_jsstr = cx.string(format!("{}_{}", id, ch));
                    arr.set(&mut cx, i as u32, id_ch_jsstr)?;
                }
                Ok(arr)
            }
            Ok(Some(_)) => Ok(cx.empty_array()),
            Ok(None) => cx.throw_error("no need to refresh."),
            Err(_) => cx.throw_error("Unknown error!"),
        }
    }
}

fn get_max_db(mut cx: FunctionContext) -> JsResult<JsNumber> {
    Ok(cx.number(TM.read().unwrap().max_db))
}

fn get_min_db(mut cx: FunctionContext) -> JsResult<JsNumber> {
    Ok(cx.number(TM.read().unwrap().min_db))
}

fn get_n_ch(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let tm = TM.read().unwrap();
    let track = get_track!(tm, cx, 0);
    Ok(cx.number(track.n_ch as u32))
}

fn get_sec(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let tm = TM.read().unwrap();
    let track = get_track!(tm, cx, 0);
    Ok(cx.number(track.sec()))
}

fn get_sr(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let tm = TM.read().unwrap();
    let track = get_track!(tm, cx, 0);
    Ok(cx.number(track.sr))
}

fn get_path(mut cx: FunctionContext) -> JsResult<JsString> {
    let tm = TM.read().unwrap();
    let track = get_track!(tm, cx, 0);
    Ok(cx.string(track.get_path()))
}

fn get_filename(mut cx: FunctionContext) -> JsResult<JsString> {
    let tm = TM.read().unwrap();
    let track = get_track!(tm, cx, 0);
    Ok(cx.string(track.get_filename()))
}

fn get_colormap(mut cx: FunctionContext) -> JsResult<JsArrayBuffer> {
    let (c_iter, len) = get_colormap_iter_size();
    let mut buf = JsArrayBuffer::new(&mut cx, len as u32)?;
    cx.borrow_mut(&mut buf, |borrowed| {
        for (x, &y) in borrowed.as_mut_slice().iter_mut().zip(c_iter) {
            *x = y;
        }
    });
    Ok(buf)
}

register_module!(mut m, {
    initialize(&TM);
    initialize(&DRAWOPTION);
    initialize(&IMAGES);
    m.export_function("addTracks", add_tracks)?;
    m.export_function("removeTrack", remove_track)?;
    m.export_function("getMaxdB", get_max_db)?;
    m.export_function("getMindB", get_min_db)?;
    m.export_function("getNumCh", get_n_ch)?;
    m.export_function("getSec", get_sec)?;
    m.export_function("getSr", get_sr)?;
    m.export_function("getPath", get_path)?;
    m.export_function("getFileName", get_filename)?;
    m.export_function("getColormap", get_colormap)?;
    m.export_function("getImages", get_images)?;
    Ok(())
});
