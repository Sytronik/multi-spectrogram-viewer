import React, { forwardRef, useRef, useImperativeHandle } from 'react';

const Canvas = forwardRef(({ width, height }, ref) => {
  const canvas = useRef(null);
  const time = useRef(0.);

  useImperativeHandle(ref, () => ({
    draw: (bufs) => {
      const [buf_spec, buf_wav] = bufs;
      if (!buf_spec && !buf_wav) {
        return;
      }
      if (
        !(buf_spec && buf_spec.byteLength === 4 * width * height)
        && !(buf_wav && buf_wav.byteLength === 4 * width * height)
      ) {
        return;
      }
      // console.log(canvas.current);
      // const ctx = canvas.current.getContext("bitmaprenderer");
      // const imdata = new ImageData(new Uint8ClampedArray(buf_spec), width, height);
      // const imbmp = await createImageBitmap(imdata);
      // ctx.transferFromImageBitmap(imbmp);

      requestAnimationFrame(async (timestamp) => {
        if (time.current === timestamp) return;
        time.current = timestamp;
        const im_spec = new ImageData(new Uint8ClampedArray(buf_spec), width, height);
        const im_wav = new ImageData(new Uint8ClampedArray(buf_wav), width, height);
        const offscreen = new OffscreenCanvas(canvas.current.width, canvas.current.height);
        const ctx = offscreen.getContext('2d');
        ctx.clearRect(0, 0, canvas.current.width, canvas.current.height);
        const promise_bmp_spec = createImageBitmap(im_spec);
        const promise_bmp_wav = createImageBitmap(im_wav);
        const [bmp_spec, bmp_wav] = await Promise.all([promise_bmp_spec, promise_bmp_wav]);
        ctx.drawImage(bmp_spec, 0, 0);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillRect(0, 0, offscreen.width, offscreen.height);
        ctx.drawImage(bmp_wav, 0, 0);
        canvas.current
          .getContext("bitmaprenderer")
          .transferFromImageBitmap(offscreen.transferToImageBitmap());
        // im_wav.data.forEach((d, i, a) => {
        //   if ((i + 1) % 4) {
        //     a[i] = 128;
        //   }
        // });
        // ctx.putImageData(im_wav, 0, 0);
      });
    }
  }));

  return (
    <>
      <canvas ref={canvas} height={height} width={width} className="Canvas" />
    </>
  );
});

export default Canvas;