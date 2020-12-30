import React, { forwardRef, useRef, useImperativeHandle } from 'react';

const Canvas = forwardRef(({ width, height }, ref) => {
  const canvas = useRef(null);

  useImperativeHandle(ref, () => ({
    draw: async (buf) => {
      if (!buf || buf.byteLength !== 4 * width * height) {
        return;
      }
      const ctx = canvas.current.getContext("bitmaprenderer");
      const imdata = new ImageData(new Uint8ClampedArray(buf), width, height);
      const imbmp = await createImageBitmap(imdata);
      ctx.transferFromImageBitmap(imbmp);
    }
  }));

  return (
    <>
      <canvas ref={canvas} height={height} width={width} className="Canvas" />
    </>
  );
});

export default Canvas;