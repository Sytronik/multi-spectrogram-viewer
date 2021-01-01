use neon::{prelude::*, result::Throw};

use thesia_backend::DrawOption;

#[macro_export(local_inner_macros)]
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

#[macro_export(local_inner_macros)]
macro_rules! get_num_arg {
    ($cx:expr, $i_arg:expr $(, $type:ty)?) => {
        $cx.argument::<JsNumber>($i_arg)?.value() $(as $type)?
    };
}

#[macro_export(local_inner_macros)]
macro_rules! get_num_field {
    ($obj:expr, $cx:expr, $key:expr $(, $type:ty)?) => {
        $obj.get(&mut $cx, $key)?.downcast::<JsNumber>().unwrap().value() $(as $type)?
    };
}

#[macro_export(local_inner_macros)]
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

#[macro_export(local_inner_macros)]
macro_rules! tuples_to_str_vec {
    ($id_ch_tuples: expr) => {
        $id_ch_tuples
            .iter()
            .map(|(id, ch)| std::format!("{}_{}", id, ch))
            .collect::<Vec<String>>()
    };
}

#[inline]
pub fn get_drawoption_arg_(cx: &mut FunctionContext, index: i32) -> Result<DrawOption, Throw> {
    let object = cx.argument::<JsObject>(index)?;
    let px_per_sec = get_num_field!(object, *cx, "px_per_sec");
    let height = get_num_field!(object, *cx, "height", u32);
    let blend = get_num_field!(object, *cx, "blend");
    Ok(DrawOption {
        px_per_sec,
        height,
        blend,
    })
}

#[inline]
pub fn convert_to_jsarr<'a, C: Context<'a>>(cx: &mut C, data: &[String]) -> JsResult<'a, JsArray> {
    let arr = JsArray::new(cx, data.len() as u32);
    for (i, item) in data.iter().enumerate() {
        let jsstr = cx.string(item);
        arr.set(cx, i as u32, jsstr)?;
    }
    Ok(arr)
}
