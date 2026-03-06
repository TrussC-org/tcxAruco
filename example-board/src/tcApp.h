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

private:
    VideoGrabber grabber;
    ArucoDetector aruco;
    int numDetected = 0;
    vector<BoardHandle> boardHandles;
};
