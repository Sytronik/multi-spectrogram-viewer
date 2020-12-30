import React from "react";
import "./App.scss";
import Control from "./components/Control/Control"
import Overview from "./components/Overview/Overview"
import SlideBar from "./components/SlideBar/SlideBar"
import MainViewer from "./components/MainViewer/MainViewer"
import ColorBar from "./components/ColorBar/ColorBar"

const p = window.preload;

function App() {
  return (
    <div className="App">
      <Control />
      <div>
        <Overview />
        <SlideBar />
      </div>
      <div>
        <ColorBar />
        <MainViewer native={p.native} />
      </div>
    </div>
  );
}

export default App;
