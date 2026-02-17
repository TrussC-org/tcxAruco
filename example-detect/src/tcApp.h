#pragma once

#include <TrussC.h>
#include <tcxAruco.h>
using namespace std;
using namespace tc;
using namespace tcx;

class tcApp : public App {
public:
    void setup() override;
    void update() override;
    void draw() override;
    void keyPressed(int key) override;

private:
    VideoGrabber grabber;
    ArucoDetector aruco;
    int numDetected = 0;
};
