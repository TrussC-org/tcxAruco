#include "tcApp.h"

void tcApp::setup() {
    // List cameras and select device
    auto devices = grabber.listDevices();
    for (auto& d : devices) {
        logNotice() << "Camera [" << d.deviceId << "]: " << d.deviceName;
    }
    grabber.setDeviceID(2);
    grabber.setup(3840, 2160);

    aruco.setThreaded(true);
    aruco.setup("calibration.yml", 3840, 2160);
    aruco.setMarkerSize(0.012f);

    // Register boards: one per marker ID (0-49), simulating PartsTracker usage
    float size = 0.012f;
    for (int i = 0; i < 50; i++) {
        vector<ArucoMarker> markers;
        markers.push_back(ArucoMarker(i, Vec3(0, 0, 0), size));
        BoardHandle h = aruco.addCustomBoard(markers);
        boardHandles.push_back(h);
    }

    logNotice() << "Registered " << boardHandles.size() << " boards";
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

    // Overlay
    setColor(1);
    drawBitmapString("tcxAruco - Board Detection (4K, threaded)", 20, 20);
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
        int boardsDetected = 0;
        string boardInfo;
        for (auto h : boardHandles) {
            int m = aruco.getBoardMarkersDetected(h);
            if (m > 0) {
                boardInfo += to_string(h) + "(" + to_string(m) + ") ";
            }
            if (aruco.isBoardDetected(h)) boardsDetected++;
        }
        drawBitmapString("Boards detected: " + to_string(boardsDetected), 20, 80);
        if (!boardInfo.empty()) {
            drawBitmapString("Board markers: " + boardInfo, 20, 100);
        }
    }

    drawBitmapString("Threaded: " + string(aruco.isThreaded() ? "ON" : "OFF"), 20, getWindowHeight() - 20);
}
