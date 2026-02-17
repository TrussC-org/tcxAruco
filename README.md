# tcxAruco

ArUco marker detection addon for [TrussC](https://github.com/TrussC-org/TrussC).

Port of [ofxArucoCV4](https://github.com/tettou771/ofxArucoCV4) for TrussC, using OpenCV 4.x ArUco module.

## Features

- ArUco marker detection (DICT_4X4_50 default, all dictionaries supported)
- Individual marker pose estimation (solvePnP)
- Board detection (grid boards and custom boards)
- Threaded detection for non-blocking operation
- OpenCV → TrussC/OpenGL coordinate system conversion
- Camera calibration file support (OpenCV YAML format)

## Dependencies

- **tcxOpenCv** (included in TrussC addons, fetches OpenCV 4.10 via FetchContent)

## Setup

1. Add `tcxAruco` and `tcxOpenCv` to your project's `addons.make`:
   ```
   tcxAruco
   tcxOpenCv
   ```

2. Run projectGenerator to update your project.

3. Place a camera calibration file in `bin/data/` (OpenCV YAML format).

> **Note:** First build will take several minutes as OpenCV is downloaded and compiled via FetchContent.

## Usage

```cpp
#include <TrussC.h>
#include <tcxAruco.h>
using namespace std;
using namespace tc;
using namespace tcx;

ArucoDetector aruco;
VideoGrabber camera;

void setup() {
    camera.setup(640, 480);
    aruco.setup("calibration.yml", 640, 480);
    aruco.setMarkerSize(0.05); // 5cm markers
}

void update() {
    camera.update();
    if (camera.isFrameNew()) {
        aruco.detectMarkers(camera.getPixels(), 640, 480, 4); // RGBA
    }
}

void draw() {
    camera.draw(0, 0);
    drawBitmapString("Markers: " + to_string(aruco.getNumMarkers()), 20, 20);
}
```

### Board Detection

```cpp
// Create a 5x7 grid board
BoardHandle board = aruco.addGridBoard(5, 7, 0.04, 0.01);

// Detect boards
aruco.detectBoards(camera.getPixels(), 640, 480, 4);

if (aruco.isBoardDetected(board)) {
    Mat4 mv = aruco.getBoardModelViewMatrix(board);
    // Use mv for 3D rendering on the board
}
```

### Custom Board

```cpp
vector<ArucoMarker> markers;
markers.push_back(ArucoMarker(0, Vec3(0, 0, 0), 0.05));
markers.push_back(ArucoMarker(1, Vec3(0.1, 0, 0), 0.05));
BoardHandle custom = aruco.addCustomBoard(markers);
```

## API Reference

### ArucoDetector

| Method | Description |
|--------|-------------|
| `setup(calibFile, w, h, dict)` | Initialize with calibration and dictionary |
| `setThreaded(bool)` | Enable/disable threaded detection (call before setup) |
| `setMarkerSize(meters)` | Set marker size for pose estimation |
| `detectMarkers(data, w, h, ch)` | Detect markers from pixel data |
| `detectMarkers(pixels)` | Detect markers from Pixels object |
| `detectBoards(data, w, h, ch)` | Detect markers + estimate board poses |
| `getNumMarkers()` | Number of detected markers |
| `getMarkerIds()` | List of detected marker IDs |
| `getModelViewMatrix(index)` | Get marker's model-view matrix (TrussC coords) |
| `getProjectionMatrix()` | Get projection matrix from calibration |
| `addGridBoard(mx, my, len, sep)` | Add a grid board |
| `addCustomBoard(markers)` | Add a custom board |
| `isBoardDetected(handle)` | Check if board was detected |
| `getBoardModelViewMatrix(handle)` | Get board's model-view matrix |

### Coordinate System

OpenCV uses Y-down, Z-forward. TrussC/OpenGL uses Y-up, Z-backward.
The conversion is handled automatically — all returned matrices are in TrussC coordinate space.

## Camera Calibration

The calibration file should be in OpenCV YAML format with `camera_matrix` and `distortion_coefficients`:

```yaml
%YAML:1.0
camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ fx, 0, cx, 0, fy, cy, 0, 0, 1 ]
distortion_coefficients: !!opencv-matrix
   rows: 1
   cols: 5
   dt: d
   data: [ k1, k2, p1, p2, k3 ]
```

You can generate this file using OpenCV's camera calibration tools or the `calibrateCamera()` function.

## Example

See `example-detect/` for a complete working example with webcam input.

## License

MIT
