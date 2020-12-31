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

function Tracks({ width, track_ids, getImages }) {
  const sec = useRef(0.);
  const [height, setHeight] = useState(250);
  const draw_option = useRef({ px_per_sec: 100., blend: 0.3 });
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
  async function handleWheel(e) {
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
        setHeight(Math.round(Math.max(height * (1 + delta / 1000), 10)));
      } else {
        const px_per_sec = Math.min(
          Math.max(draw_option.current.px_per_sec * (1 + e.deltaY / 1000), 0.),
          384000
        );
        if (draw_option.current.px_per_sec !== px_per_sec) {
          draw_option.current.px_per_sec = px_per_sec;
          throttled_draw();
        }
      }
    } else if ((e.shiftKey && y_is_larger) || !y_is_larger) {
      e.preventDefault();
      e.stopPropagation();
      sec.current += delta / draw_option.current.px_per_sec;
      throttled_draw();
    }
  }
  const draw = useCallback(
    async () => {
      const { id_ch_arr, buf } = getImages(
        sec.current, width, { ...draw_option.current, height: height },
        (_, arr) => {
          if (arr) {
            debounced_draw();
          }
        }
      );
      const imlength = buf.byteLength / id_ch_arr.length;
      for (const [i, id_ch_str] of id_ch_arr.entries()) {
        const buf_i = buf.slice(i * imlength, (i + 1) * imlength);

        const ref = children.current[id_ch_str];
        let promises = [];
        if (ref) {
          promises.push(ref.draw(buf_i));
        }
        await Promise.all(promises);
      }
    }, [height, width]);
  const throttled_draw = useCallback(throttle(1000 / 120, draw), [draw]);
  const debounced_draw = useCallback(debounce(1000 / 120, draw), [draw]);
  useEffect(() => throttled_draw(), [draw, height, width]);

  return (
    <div onWheel={handleWheel} width={width} className="tracks">
      {canvas_arr}
    </div>
  )
}

export default Tracks;