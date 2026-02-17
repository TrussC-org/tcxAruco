#include "tcApp.h"

void tcApp::setup() {
    grabber.setup(640, 480);
    aruco.setup("calibration.yml", 640, 480);
    aruco.setMarkerSize(0.05f);
}

void tcApp::update() {
    grabber.update();
    if (grabber.isFrameNew()) {
        auto* pixels = grabber.getPixels();
        if (pixels) {
            aruco.detectMarkers(pixels, grabber.getWidth(), grabber.getHeight(), 4);
        }
    }
    numDetected = aruco.getNumMarkers();
}

void tcApp::draw() {
    clear(0.12f);

    // Draw camera image
    grabber.draw(0, 0, getWindowWidth(), getWindowHeight());

    // Draw marker info overlay
    setColor(1);
    drawBitmapString("tcxAruco - Marker Detection", 20, 20);
    drawBitmapString("Markers detected: " + to_string(numDetected), 20, 40);

    if (numDetected > 0) {
        auto ids = aruco.getMarkerIds();
        string idStr = "IDs: ";
        for (int i = 0; i < (int)ids.size(); i++) {
            if (i > 0) idStr += ", ";
            idStr += to_string(ids[i]);
        }
        drawBitmapString(idStr, 20, 60);
    }

    drawBitmapString("Press 't' to toggle threaded mode", 20, getWindowHeight() - 40);
    drawBitmapString("Threaded: " + string(aruco.isThreaded() ? "ON" : "OFF"), 20, getWindowHeight() - 20);
}

void tcApp::keyPressed(int key) {
    if (key == 't') {
        // Note: threading can only be changed before setup
        logNotice() << "Threading mode can only be changed before setup()";
    }
}
