import React from 'react';
import Tracks from "./Tracks"

function MainViewer({ native }) {
  const paths = [
    "samples/sample_48k.wav",
    "samples/sample_44k1.wav",
    "samples/sample_24k.wav",
    "samples/sample_22k05.wav",
    "samples/sample_16k.wav",
    "samples/sample_8k.wav",
  ];
  const track_ids = [...paths.keys()];
  native.addTracks(track_ids, paths, (e, v) => {
    // console.log(e);
    // console.log(v);
  });
  return (
    <div className="MainViewer">
      🚩 main viewer
      {/* <TimeRuler /> */}
      <Tracks width={600} getImages={native.getImages} track_ids={track_ids} />
    </div>
  );
}

export default MainViewer;