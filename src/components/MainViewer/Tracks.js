import React, { useRef, useCallback, useEffect, useState } from 'react';
import { throttle, debounce } from 'throttle-debounce';

import { SplitView } from "./SplitView";
import TrackInfo from "./TrackInfo";
import Canvas from "./Canvas";


function useRefs() {
  const refs = useRef({});

  const register = useCallback((refName) => ref => {
    refs.current[refName] = ref;
  }, []);

  return [refs, register];
}

function Tracks({ width, track_ids, getSpecWavImages }) {
  const sec = useRef(0.);
  const [height, setHeight] = useState(250);
  const draw_option = useRef({ px_per_sec: 100. });
  const [children, registerChild] = useRefs();

  const canvas_arr = track_ids.map((i) => {
    return (
      <SplitView
        key={`${i}`}
        left={<TrackInfo />}
        right={
          <Canvas ref={registerChild(`${i}_0`)} width={width} height={height} />
        }
      />
    )
  });
  const getIdChArr = () => Object.keys(children.current);
  function handleWheel(e) {
    let y_is_larger;
    let delta;
    if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
      delta = e.deltaY;
      y_is_larger = true;
    } else {
      delta = e.deltaX;
      y_is_larger = false;
    }
    if (e.altKey) {
      e.preventDefault();
      e.stopPropagation();
      if ((e.shiftKey && y_is_larger) || !y_is_larger) {
        setHeight(Math.round(Math.min(Math.max(height * (1 + delta / 1000), 10), 5000)));
      } else {
        const px_per_sec = Math.min(
          Math.max(draw_option.current.px_per_sec * (1 + e.deltaY / 1000), 0.),
          384000
        );
        if (draw_option.current.px_per_sec !== px_per_sec) {
          draw_option.current.px_per_sec = px_per_sec;
          throttled_draw([getIdChArr()]);
        }
      }
    } else if ((e.shiftKey && y_is_larger) || !y_is_larger) {
      e.preventDefault();
      e.stopPropagation();
      sec.current += delta / draw_option.current.px_per_sec;
      throttled_draw([getIdChArr()]);
    }
  }
  const draw = useCallback(
    async (id_ch_arr) => {
      const [images, promise] = getSpecWavImages(
        id_ch_arr[0],
        sec.current, width,
        { ...draw_option.current, height: height },
        { min_amp: -1., max_amp: 1. }
      );

      for (const [id_ch_str, bufs] of Object.entries(images)) {
        const ref = children.current[id_ch_str];
        // let promises = [];
        if (ref) {
          // promises.push(
          ref.draw(bufs);
          // );
        }
        // Promise.all(promises);
      }

      // cached image
      if (promise !== null) {
        const arr = await promise;
        if (arr) {
          // console.log(arr);
          debounced_draw(arr);
        }
      }
    }, [height, width]);
  const throttled_draw = useCallback(throttle(1000 / 60, draw), [draw]);
  const debounced_draw = useCallback(debounce(1000 / 60, draw), [draw]);
  // const throttled_draw = draw;
  // const debounced_draw = draw;

  useEffect(() => {
    throttled_draw([getIdChArr()]);
  }, [draw, height, width]);

  return (
    <div onWheel={handleWheel} width={width} className="tracks">
      {canvas_arr}
    </div>
  )
}

export default Tracks;