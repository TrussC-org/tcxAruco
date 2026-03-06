#include "tcApp.h"

void tcApp::setup() {
    grabber.setup(640, 480);

    aruco.setThreaded(true);
    aruco.setup("calibration.yml", 640, 480);
    aruco.setMarkerSize(0.012f);

    // Register a single board with all 50 markers in a 10x5 grid
    float markerLen = 0.012f;
    float markerSep = 0.003f;
    BoardHandle h = aruco.addGridBoard(10, 5, markerLen, markerSep);
    boardHandles.push_back(h);

    logNotice() << "Registered 1 board (10x5 grid, 50 markers)";
}

void tcApp::update() {
    grabber.update();
    if (grabber.isFrameNew()) {
        auto* pixels = grabber.getPixels();
        if (pixels) {
            aruco.detectBoards(pixels, grabber.getWidth(), grabber.getHeight(), 4);
        }
    }
    numDetected = aruco.getNumMarkers();
}

void tcApp::draw() {
    clear(0.12f);

    // Draw camera image
    grabber.draw(0, 0, getWindowWidth(), getWindowHeight());

    // AR overlay: draw 3D boxes on markers (white) and boards (red)
    aruco.beginAR();
    for (int i = 0; i < aruco.getNumMarkers(); i++) {
        aruco.drawMarkerOverlay(i);
    }
    for (auto h : boardHandles) {
        aruco.drawBoardOverlay(h);
    }
    aruco.endAR();

    // Overlay
    setColor(1);
    drawBitmapString("tcxAruco - Board Detection (threaded)", 20, 20);
    drawBitmapString("Markers detected: " + to_string(numDetected), 20, 40);

    if (numDetected > 0) {
        auto ids = aruco.getMarkerIds();
        string idStr = "IDs: ";
        for (int i = 0; i < (int)ids.size(); i++) {
            if (i > 0) idStr += ", ";
            idStr += to_string(ids[i]);
        }
        drawBitmapString(idStr, 20, 60);

        // Show board detection status
        BoardHandle bh = boardHandles[0];
        int boardMarkers = aruco.getBoardMarkersDetected(bh);
        bool boardOk = aruco.isBoardDetected(bh);
        drawBitmapString("Board: " + string(boardOk ? "DETECTED" : "not detected") +
                         " (" + to_string(boardMarkers) + "/50 markers)", 20, 80);
    }

    drawBitmapString("Threaded: " + string(aruco.isThreaded() ? "ON" : "OFF"), 20, getWindowHeight() - 20);
}
