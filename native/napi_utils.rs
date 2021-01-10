use std::collections::HashSet;
use std::convert::TryInto;
use std::hash::Hash;

use napi::{CallContext, Env, JsNumber, JsObject, JsString, Result as JsResult};

use crate::backend::{DrawOption, DrawOptionForWav, IdChMap, IdChVec};

pub trait TryIntoUsize: TryInto<u32> {
    fn try_into_usize(self) -> Result<usize, Self::Error>;
}

impl TryIntoUsize for JsNumber {
    fn try_into_usize(self) -> Result<usize, Self::Error> {
        Ok(TryInto::<u32>::try_into(self)? as usize)
    }
}

pub trait TryIntoF32: TryInto<f64> {
    fn try_into_f32(self) -> Result<f32, Self::Error>;
}

impl TryIntoF32 for JsNumber {
    fn try_into_f32(self) -> Result<f32, Self::Error> {
        Ok(TryInto::<f64>::try_into(self)? as f32)
    }
}

#[macro_export(local_inner_macros)]
macro_rules! get_track {
    ($ctx: expr, $i_id_arg: expr, $tm_read: expr) => {
        $tm_read
            .tracks
            .get(&$ctx.get::<JsNumber>(0)?.try_into_usize()?)
            .unwrap()
    };
}

#[macro_export(local_inner_macros)]
macro_rules! this_is {
    (let $tm_name:ident = $type:ident in $ctx:expr) => {
        let $tm_name: &mut $type = $ctx.env.unwrap(&mut $ctx.this_unchecked())?;
    };
}

pub fn id_ch_tuples_from(ctx: &CallContext, i_arg: usize) -> JsResult<IdChVec> {
    let arr = ctx.get::<JsObject>(i_arg)?;
    let len = arr.get_array_length()?;
    let mut vec = IdChVec::with_capacity(len as usize);
    for i in (0..len).into_iter() {
        let js_str = arr.get_element::<JsString>(i)?.into_utf8()?;
        let s = js_str.as_str()?;
        let mut iter = s.split("_").map(|x| x.parse::<usize>());
        match (iter.next(), iter.next()) {
            (Some(Ok(id)), Some(Ok(ch))) => {
                vec.push((id, ch));
            }
            _ => {
                return Err(ctx
                    .env
                    .throw_error("The array element should be \"int_int\".", None)
                    .err()
                    .unwrap())
            }
        };
    }
    Ok(vec)
}

pub fn vec_from<T: From<JsNumber>>(ctx: &CallContext, i_arg: usize) -> JsResult<Vec<T>> {
    let arr = ctx.get::<JsObject>(i_arg)?;
    let len = arr.get_array_length()?;
    let mut vec = Vec::<T>::with_capacity(len as usize);
    for i in (0..len).into_iter() {
        if let Ok(x) = arr.get_element::<JsNumber>(i)?.try_into() {
            vec.push(x);
        }
    }
    Ok(vec)
}

pub fn vec_usize_from(ctx: &CallContext, i_arg: usize) -> JsResult<Vec<usize>> {
    let arr = ctx.get::<JsObject>(i_arg)?;
    let len = arr.get_array_length()?;
    let mut vec = Vec::<usize>::with_capacity(len as usize);
    for i in (0..len).into_iter() {
        if let Ok(x) = arr.get_element::<JsNumber>(i)?.try_into_usize() {
            vec.push(x);
        }
    }
    Ok(vec)
}

pub fn vec_str_from(ctx: &CallContext, i_arg: usize) -> JsResult<Vec<String>> {
    let arr = ctx.get::<JsObject>(i_arg)?;
    let len = arr.get_array_length()?;
    let mut vec = Vec::<String>::with_capacity(len as usize);
    for i in (0..len).into_iter() {
        let jsstr = arr.get_element::<JsString>(i)?.into_utf8()?;
        vec.push(jsstr.into_owned()?);
    }
    Ok(vec)
}

pub fn convert_id_ch_vec_to_jsarr<'a>(
    env: &Env,
    arr: impl Iterator<Item = &'a (usize, usize)>,
    len: usize,
) -> JsResult<JsObject> {
    let mut obj = env.create_array_with_length(len)?;
    for (i, &(id, ch)) in arr.enumerate() {
        obj.set_element(
            i as u32,
            env.create_string_from_std(format!("{}_{}", id, ch))?,
        )?;
    }
    Ok(obj)
}

#[inline]
pub fn draw_option_from_js_obj(js_obj: JsObject) -> JsResult<DrawOption> {
    Ok(DrawOption {
        px_per_sec: js_obj
            .get_named_property::<JsNumber>("px_per_sec")?
            .try_into()?,
        height: js_obj
            .get_named_property::<JsNumber>("height")?
            .try_into()?,
    })
}

#[inline]
pub fn draw_opt_for_wav_from_js_obj(js_obj: JsObject) -> JsResult<DrawOptionForWav> {
    Ok(DrawOptionForWav {
        amp_range: (
            js_obj
                .get_named_property::<JsNumber>("min_amp")?
                .try_into_f32()?,
            js_obj
                .get_named_property::<JsNumber>("max_amp")?
                .try_into_f32()?,
        ),
    })
}

pub fn set_images_to(
    env: &Env,
    this: &mut JsObject,
    images: IdChMap<Vec<u8>>,
    index: u32,
) -> JsResult<()> {
    for ((id, ch), im) in images.into_iter() {
        let name = format!("{}_{}", id, ch);
        let buf = env.create_buffer_with_data(im)?.into_raw();
        if !this.has_named_property(name.as_str())? {
            this.set_named_property(name.as_str(), env.create_array_with_length(2)?)?;
        }
        this.get_named_property_unchecked::<JsObject>(name.as_str())?
            .set_element(index, buf)?;
    }
    Ok(())
}

#[inline]
pub fn extract_intersect<T: Clone + Eq + Hash>(
    a: &mut HashSet<T>,
    b: &mut HashSet<T>,
) -> HashSet<T> {
    let c: HashSet<T> = a.iter().filter_map(|v| b.take(v)).collect();
    a.retain(|v| !c.contains(&v));
    c
}
