import React, { Component } from 'react';
import Tracks from "./Tracks"

class MainViewer extends Component {
  render() {
    return (
      <div className="MainViewer">
        🚩 main viewer
        {/* <TimeRuler /> */}
        <Tracks />
        <div className="empty">🚩 empty</div>
      </div>
    );
  }
}

export default MainViewer;