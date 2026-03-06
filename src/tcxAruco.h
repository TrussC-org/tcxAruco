#pragma once

// =============================================================================
// tcxAruco - ArUco marker detection for TrussC
// =============================================================================
// Header-only ArUco marker detector using OpenCV 4.x.
// Port of ofxArucoCV4 for TrussC.
//
// Usage:
//   #include <TrussC.h>
//   #include <tcxAruco.h>
//   using namespace tcx;
//
//   ArucoDetector aruco;
//   aruco.setup("calibration.yml", 640, 480);
//   aruco.setMarkerSize(0.05);
//   aruco.detectMarkers(pixelData, 640, 480, 4);
//   cout << "Found: " << aruco.getNumMarkers() << endl;
// =============================================================================

#include <TrussC.h>
#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstring>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <opencv2/objdetect/aruco_board.hpp>

namespace tcx {

// =============================================================================
// ArucoMarker - Marker definition for custom boards
// =============================================================================

struct ArucoMarker {
    int id;
    tc::Vec3 corners[4];

    // Constructor 1: center + size (XY plane, preserves center Z)
    // ArUco corner order: 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left (image coords, Y down)
    ArucoMarker(int _id, tc::Vec3 center, float size) : id(_id) {
        float half = size / 2.0f;
        corners[0] = tc::Vec3(center.x - half, center.y - half, center.z);
        corners[1] = tc::Vec3(center.x + half, center.y - half, center.z);
        corners[2] = tc::Vec3(center.x + half, center.y + half, center.z);
        corners[3] = tc::Vec3(center.x - half, center.y + half, center.z);
    }

    // Constructor 2: 3 corners (4th corner auto-computed as parallelogram)
    ArucoMarker(int _id, tc::Vec3 p0, tc::Vec3 p1, tc::Vec3 p2) : id(_id) {
        corners[0] = p0;
        corners[1] = p1;
        corners[2] = p2;
        corners[3] = p0 + (p2 - p1);
    }
};

// Board handle (opaque, internally an index)
using BoardHandle = int;
constexpr BoardHandle INVALID_BOARD_HANDLE = -1;

// =============================================================================
// ArucoDetector
// =============================================================================

class ArucoDetector {
public:
    ArucoDetector() : markerSize_(0.15f) {}

    ~ArucoDetector() {
        if (threaded_ && running_) {
            running_ = false;
            if (workerThread_.joinable()) {
                workerThread_.join();
            }
        }
    }

    // =========================================================================
    // Setup
    // =========================================================================

    void setup(const std::string& calibrationFile, float w, float h,
               cv::aruco::PredefinedDictionaryType dictType = cv::aruco::DICT_4X4_50) {
        imageSize_ = cv::Size((int)w, (int)h);

        dictionary_ = cv::aruco::getPredefinedDictionary(dictType);
        // Preserve previously set detectorParams_ (e.g. from setMinMaxMarkerDetectionSize)
        detector_ = cv::aruco::ArucoDetector(dictionary_, detectorParams_);

        loadCameraParameters(calibrationFile);
        generateProjectionMatrix(0.05f, 100.0f);

        if (threaded_) {
            running_ = true;
            workerThread_ = std::thread(&ArucoDetector::workerFunction, this);
        }
    }

    // Threading (must call before setup)
    void setThreaded(bool threaded) {
        if (!cameraMatrix_.empty()) {
            tc::logWarning("tcxAruco") << "setThreaded() must be called before setup()";
            return;
        }
        threaded_ = threaded;
    }
    bool isThreaded() const { return threaded_; }

    // Marker size in meters
    void setMarkerSize(float meters) { markerSize_ = meters; }
    float getMarkerSize() const { return markerSize_; }

    // Detection size range (safe to call before or after setup)
    void setMinMaxMarkerDetectionSize(float minSize, float maxSize) {
        detectorParams_.minMarkerPerimeterRate = minSize;
        detectorParams_.maxMarkerPerimeterRate = maxSize;
        detector_ = cv::aruco::ArucoDetector(dictionary_, detectorParams_);
        workerDetectorDirty_.store(true);
    }

    // =========================================================================
    // Board management
    // =========================================================================

    BoardHandle addGridBoard(int markersX, int markersY,
                             float markerLength, float markerSeparation) {
        int numMarkers = markersX * markersY;
        std::vector<int> ids(numMarkers);

        // Find max used ID across existing boards
        int maxUsedId = -1;
        for (const auto& entry : boards_) {
            if (!entry.active) continue;
            const auto& boardIds = entry.board.getIds();
            for (int id : boardIds) {
                if (id > maxUsedId) maxUsedId = id;
            }
        }

        for (int i = 0; i < numMarkers; i++) {
            ids[i] = maxUsedId + 1 + i;
        }

        cv::Size boardSize(markersX, markersY);
        cv::aruco::GridBoard gridBoard(boardSize, markerLength, markerSeparation, dictionary_, ids);

        BoardEntry entry;
        entry.board = gridBoard;
        entry.active = true;

        BoardHandle handle = (BoardHandle)boards_.size();
        boards_.push_back(std::move(entry));
        return handle;
    }

    BoardHandle addCustomBoard(const std::vector<ArucoMarker>& markers) {
        if (markers.empty()) {
            tc::logError("tcxAruco") << "addCustomBoard: empty marker list";
            return INVALID_BOARD_HANDLE;
        }

        std::vector<std::vector<cv::Point3f>> objPoints;
        std::vector<int> ids;

        for (const auto& marker : markers) {
            std::vector<cv::Point3f> corners(4);
            for (int i = 0; i < 4; i++) {
                corners[i] = cv::Point3f(marker.corners[i].x, marker.corners[i].y, marker.corners[i].z);
            }
            objPoints.push_back(corners);
            ids.push_back(marker.id);
        }

        cv::aruco::Board customBoard(objPoints, dictionary_, ids);

        BoardEntry entry;
        entry.board = customBoard;
        entry.active = true;

        BoardHandle handle = (BoardHandle)boards_.size();
        boards_.push_back(std::move(entry));
        return handle;
    }

    void removeBoard(BoardHandle handle) {
        if (handle < 0 || handle >= (int)boards_.size()) return;
        boards_[handle].active = false;
        boards_[handle].detected = false;
        boards_[handle].markersDetected = 0;
    }

    // =========================================================================
    // Detection - accepts raw pixel data
    // =========================================================================

    // Detect markers from raw pixel data (RGBA or RGB)
    void detectMarkers(const unsigned char* data, int width, int height, int channels) {
        if (!threaded_) {
            DetectResult result;
            findMarkers(detector_, data, width, height, channels, result);
            markerIds_ = std::move(result.markerIds);
            markerCorners_ = std::move(result.markerCorners);
            rvecs_ = std::move(result.rvecs);
            tvecs_ = std::move(result.tvecs);
        } else {
            fetchResult();
            submitRequest(data, width, height, channels, false);
        }
    }

    // Detect markers from Pixels object
    void detectMarkers(const tc::Pixels& pixels) {
        detectMarkers(pixels.getData(), pixels.getWidth(), pixels.getHeight(), pixels.getChannels());
    }

    // Detect markers + estimate board poses
    void detectBoards(const unsigned char* data, int width, int height, int channels) {
        if (!threaded_) {
            DetectResult result;
            findMarkers(detector_, data, width, height, channels, result);
            estimateBoardPoses(result);
            markerIds_ = std::move(result.markerIds);
            markerCorners_ = std::move(result.markerCorners);
            rvecs_ = std::move(result.rvecs);
            tvecs_ = std::move(result.tvecs);
            applyBoardResults(result);
        } else {
            fetchResult();
            submitRequest(data, width, height, channels, true);
        }
    }

    void detectBoards(const tc::Pixels& pixels) {
        detectBoards(pixels.getData(), pixels.getWidth(), pixels.getHeight(), pixels.getChannels());
    }

    // =========================================================================
    // Results
    // =========================================================================

    int getNumMarkers() const { return (int)markerIds_.size(); }
    std::vector<int> getMarkerIds() const { return markerIds_; }
    std::vector<std::vector<cv::Point2f>> getMarkerCorners() const { return markerCorners_; }

    // =========================================================================
    // Board queries
    // =========================================================================

    bool isBoardDetected(BoardHandle handle) const {
        if (handle < 0 || handle >= (int)boards_.size()) return false;
        return boards_[handle].active && boards_[handle].detected;
    }

    int getBoardMarkersDetected(BoardHandle handle) const {
        if (handle < 0 || handle >= (int)boards_.size()) return 0;
        return boards_[handle].markersDetected;
    }

    int getBoardMarkerCount(BoardHandle handle) const {
        if (handle < 0 || handle >= (int)boards_.size()) return 0;
        return (int)boards_[handle].board.getIds().size();
    }

    // =========================================================================
    // 3D Matrices (TrussC coordinate system: Y-up, Z-back)
    // =========================================================================

    tc::Mat4 getModelViewMatrix(int markerIndex) const {
        if (markerIndex < 0 || markerIndex >= (int)rvecs_.size()) {
            return tc::Mat4();
        }
        return cvToTcMatrix(rvecs_[markerIndex], tvecs_[markerIndex]);
    }

    tc::Mat4 getProjectionMatrix() const {
        return projMatrix_;
    }

    tc::Mat4 getBoardModelViewMatrix(BoardHandle handle) const {
        if (handle < 0 || handle >= (int)boards_.size() || !boards_[handle].detected) {
            return tc::Mat4();
        }
        return cvToTcMatrix(boards_[handle].rvec, boards_[handle].tvec);
    }

    // =========================================================================
    // AR overlay drawing
    // =========================================================================

    // Begin AR drawing mode (sets projection matrix from calibration)
    void beginAR() const {
        // Save current projection and modelview
        sgl_matrix_mode_projection();
        sgl_push_matrix();
        tc::Mat4 proj = projMatrix_.transposed();
        sgl_load_matrix(proj.m);
        sgl_matrix_mode_modelview();
        sgl_push_matrix();
    }

    // End AR drawing mode (restores previous matrices)
    void endAR() const {
        // Restore modelview
        sgl_pop_matrix();
        // Restore projection
        sgl_matrix_mode_projection();
        sgl_pop_matrix();
        sgl_matrix_mode_modelview();
        // Sync RenderContext with restored sgl state
        tc::getDefaultContext().resetMatrix();
    }

    // Draw marker overlay: white wireframe box sitting on marker + XYZ gizmo
    void drawMarkerOverlay(int markerIndex, float gizmoScale = 0.5f) const {
        if (markerIndex < 0 || markerIndex >= (int)rvecs_.size()) return;
        tc::Mat4 mv = cvToTcMatrix(rvecs_[markerIndex], tvecs_[markerIndex]);
        drawPoseBox(mv, markerSize_, tc::Color(1, 1, 1, 0.8f), gizmoScale);
    }

    // Draw board overlay: red wireframe box + XYZ gizmo, centered on board
    void drawBoardOverlay(BoardHandle handle, float sizeScale = 3.0f, float gizmoScale = 0.5f) const {
        if (handle < 0 || handle >= (int)boards_.size() || !boards_[handle].detected) return;
        tc::Mat4 mv = cvToTcMatrix(boards_[handle].rvec, boards_[handle].tvec);
        float size = estimateBoardSize(handle) * sizeScale;
        tc::Vec3 center = computeBoardCenter(handle);
        drawPoseBox(mv, size, tc::Color(1, 0.3f, 0.2f, 0.8f), gizmoScale, center);
    }

    // Draw all detected markers
    void drawAllMarkerOverlays(float gizmoScale = 0.5f) const {
        beginAR();
        for (int i = 0; i < getNumMarkers(); i++) {
            drawMarkerOverlay(i, gizmoScale);
        }
        endAR();
    }

    // Draw all detected boards
    void drawAllBoardOverlays(const std::vector<BoardHandle>& handles, float gizmoScale = 0.5f) const {
        beginAR();
        for (auto h : handles) {
            drawBoardOverlay(h, gizmoScale);
        }
        endAR();
    }

private:
    // =========================================================================
    // AR drawing helpers
    // =========================================================================

    void drawPoseBox(const tc::Mat4& mv, float size, const tc::Color& color,
                     float gizmoScale, tc::Vec3 offset = tc::Vec3(0, 0, 0)) const {
        // Use RenderContext::loadMatrix to keep sgl and RenderContext in sync
        tc::getDefaultContext().loadMatrix(mv);

        float half = size / 2.0f;

        // Wireframe box sitting on the marker face
        // After cvToTcMatrix: XY = marker face, Z = toward camera
        // Offset is in OpenCV board coords, flip Y/Z for TrussC
        tc::noFill();
        tc::setColor(color);
        tc::pushMatrix();
        tc::translate(offset.x, -offset.y, -offset.z);
        tc::translate(0, 0, half);  // Raise box along Z (toward camera)
        tc::drawBox(size, size, size);
        tc::popMatrix();

        // XYZ gizmo at offset position
        tc::pushMatrix();
        tc::translate(offset.x, -offset.y, -offset.z);
        float len = size * gizmoScale;
        tc::setColor(1, 0, 0);  // X = red
        tc::drawLine(0, 0, 0, len, 0, 0);
        tc::setColor(0, 1, 0);  // Y = green
        tc::drawLine(0, 0, 0, 0, len, 0);
        tc::setColor(0, 0, 1);  // Z = blue
        tc::drawLine(0, 0, 0, 0, 0, len);
        tc::popMatrix();
    }

    float estimateBoardSize(BoardHandle handle) const {
        auto& entry = boards_[handle];
        auto objPoints = entry.board.getObjPoints();
        if (objPoints.empty() || objPoints[0].size() < 4) return markerSize_;
        auto& pts = objPoints[0];
        float dx = pts[1].x - pts[0].x;
        float dy = pts[1].y - pts[0].y;
        return std::sqrt(dx * dx + dy * dy);
    }

    // Compute board center in board's local coordinate system (OpenCV coords)
    tc::Vec3 computeBoardCenter(BoardHandle handle) const {
        auto& entry = boards_[handle];
        auto objPoints = entry.board.getObjPoints();
        if (objPoints.empty()) return tc::Vec3(0, 0, 0);

        float minX = 1e9f, maxX = -1e9f;
        float minY = 1e9f, maxY = -1e9f;
        for (auto& corners : objPoints) {
            for (auto& p : corners) {
                if (p.x < minX) minX = p.x;
                if (p.x > maxX) maxX = p.x;
                if (p.y < minY) minY = p.y;
                if (p.y > maxY) maxY = p.y;
            }
        }
        return tc::Vec3((minX + maxX) / 2.0f, (minY + maxY) / 2.0f, 0);
    }

    // =========================================================================
    // Internal types
    // =========================================================================

    struct BoardEntry {
        cv::aruco::Board board = cv::aruco::Board(
            std::vector<std::vector<cv::Point3f>>(),
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50),
            std::vector<int>()
        );
        cv::Vec3d rvec, tvec;
        bool detected = false;
        int markersDetected = 0;
        bool active = true;
    };

    struct DetectRequest {
        std::vector<unsigned char> pixelData;
        int width = 0, height = 0, channels = 0;
        bool detectBoards = false;
    };

    struct DetectResult {
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners;
        std::vector<cv::Vec3d> rvecs, tvecs;

        struct BoardResult {
            cv::Vec3d rvec, tvec;
            bool detected = false;
            int markersDetected = 0;
        };
        std::vector<BoardResult> boardResults;
    };

    // =========================================================================
    // Thread communication helpers (main thread only)
    // =========================================================================

    void fetchResult() {
        if (!hasResult_.load()) return;

        std::lock_guard<std::mutex> lock(resultMutex_);
        markerIds_ = std::move(latestResult_.markerIds);
        markerCorners_ = std::move(latestResult_.markerCorners);
        rvecs_ = std::move(latestResult_.rvecs);
        tvecs_ = std::move(latestResult_.tvecs);
        applyBoardResults(latestResult_);
        hasResult_.store(false);
    }

    void submitRequest(const unsigned char* data, int width, int height, int channels, bool boards) {
        std::lock_guard<std::mutex> lock(requestMutex_);
        latestRequest_.width = width;
        latestRequest_.height = height;
        latestRequest_.channels = channels;
        latestRequest_.detectBoards = boards;
        size_t bytes = (size_t)width * height * channels;
        latestRequest_.pixelData.resize(bytes);
        std::memcpy(latestRequest_.pixelData.data(), data, bytes);
        hasRequest_.store(true);
    }

    void applyBoardResults(const DetectResult& result) {
        std::lock_guard<std::mutex> lock(boardsMutex_);
        for (size_t i = 0; i < boards_.size() && i < result.boardResults.size(); i++) {
            boards_[i].rvec = result.boardResults[i].rvec;
            boards_[i].tvec = result.boardResults[i].tvec;
            boards_[i].detected = result.boardResults[i].detected;
            boards_[i].markersDetected = result.boardResults[i].markersDetected;
        }
    }

    // =========================================================================
    // Camera parameters
    // =========================================================================

    void loadCameraParameters(const std::string& filename) {
        std::string fullPath = tc::getDataPath(filename);
        cv::FileStorage fs(fullPath, cv::FileStorage::READ);

        if (!fs.isOpened()) {
            tc::logError("tcxAruco") << "Cannot open calibration file: " << filename;
            return;
        }

        fs["camera_matrix"] >> cameraMatrix_;
        fs["distortion_coefficients"] >> distCoeffs_;

        if (cameraMatrix_.type() != CV_64F) cameraMatrix_.convertTo(cameraMatrix_, CV_64F);
        if (distCoeffs_.type() != CV_64F) distCoeffs_.convertTo(distCoeffs_, CV_64F);

        fs.release();
    }

    // =========================================================================
    // Coordinate conversion: OpenCV -> TrussC
    // =========================================================================
    // OpenCV: X-right, Y-down, Z-forward
    // TrussC/OpenGL: X-right, Y-up, Z-backward
    // Conversion: diag(1,-1,-1) applied to both rotation and translation

    static tc::Mat4 cvToTcMatrix(const cv::Vec3d& rvec, const cv::Vec3d& tvec) {
        cv::Mat R;
        cv::Rodrigues(rvec, R);

        // R_gl = diag(1,-1,-1) * R_cv * diag(1,-1,-1)
        // T_gl = diag(1,-1,-1) * T_cv
        // TrussC Mat4 is row-major: Mat4(row0..., row1..., row2..., row3...)
        return tc::Mat4(
            (float) R.at<double>(0,0), (float)-R.at<double>(0,1), (float)-R.at<double>(0,2), (float) tvec[0],
            (float)-R.at<double>(1,0), (float) R.at<double>(1,1), (float) R.at<double>(1,2), (float)-tvec[1],
            (float)-R.at<double>(2,0), (float) R.at<double>(2,1), (float) R.at<double>(2,2), (float)-tvec[2],
            0.0f,                      0.0f,                      0.0f,                       1.0f
        );
    }

    void generateProjectionMatrix(float nearPlane, float farPlane) {
        if (cameraMatrix_.empty()) return;

        double fx = cameraMatrix_.at<double>(0, 0);
        double fy = cameraMatrix_.at<double>(1, 1);
        double cx = cameraMatrix_.at<double>(0, 2);
        double cy = cameraMatrix_.at<double>(1, 2);

        double w = imageSize_.width;
        double h = imageSize_.height;

        // Row-major projection matrix for TrussC
        // Note: This matches the standard OpenGL projection but stored row-major
        projMatrix_ = tc::Mat4(
            (float)(2.0 * fx / w), 0.0f,                       (float)(1.0 - 2.0 * cx / w),                          0.0f,
            0.0f,                  (float)(2.0 * fy / h),       (float)(2.0 * cy / h - 1.0),                          0.0f,
            0.0f,                  0.0f,                        (float)(-(farPlane + nearPlane) / (farPlane - nearPlane)), (float)(-2.0 * farPlane * nearPlane / (farPlane - nearPlane)),
            0.0f,                  0.0f,                        -1.0f,                                                 0.0f
        );
    }

    // =========================================================================
    // Marker detection
    // =========================================================================

    void findMarkers(cv::aruco::ArucoDetector& det,
                     const unsigned char* data, int width, int height, int channels,
                     DetectResult& result) {
        // Convert to grayscale
        cv::Mat image;
        if (channels == 4) {
            cv::Mat rgba(height, width, CV_8UC4, const_cast<unsigned char*>(data));
            cv::cvtColor(rgba, image, cv::COLOR_RGBA2GRAY);
        } else if (channels == 3) {
            cv::Mat rgb(height, width, CV_8UC3, const_cast<unsigned char*>(data));
            cv::cvtColor(rgb, image, cv::COLOR_RGB2GRAY);
        } else {
            image = cv::Mat(height, width, CV_8UC1, const_cast<unsigned char*>(data)).clone();
        }

        // Detect markers
        result.markerCorners.clear();
        result.markerIds.clear();
        det.detectMarkers(image, result.markerCorners, result.markerIds);

        // Pose estimation for individual markers
        result.rvecs.clear();
        result.tvecs.clear();

        if (!result.markerIds.empty() && markerSize_ > 0 && !cameraMatrix_.empty()) {
            for (size_t i = 0; i < result.markerCorners.size(); i++) {
                std::vector<cv::Point3f> objPoints;
                float half = markerSize_ / 2.0f;
                objPoints.push_back(cv::Point3f(-half, -half, 0));
                objPoints.push_back(cv::Point3f( half, -half, 0));
                objPoints.push_back(cv::Point3f( half,  half, 0));
                objPoints.push_back(cv::Point3f(-half,  half, 0));

                cv::Vec3d rvec, tvec;
                try {
                    cv::solvePnP(objPoints, result.markerCorners[i], cameraMatrix_, distCoeffs_, rvec, tvec);
                    result.rvecs.push_back(rvec);
                    result.tvecs.push_back(tvec);
                } catch (cv::Exception&) {
                    result.rvecs.push_back(cv::Vec3d(0, 0, 0));
                    result.tvecs.push_back(cv::Vec3d(0, 0, 0));
                }
            }
        }
    }

    // =========================================================================
    // Board pose estimation
    // =========================================================================

    void estimateBoardPoses(DetectResult& result) {
        std::lock_guard<std::mutex> lock(boardsMutex_);
        result.boardResults.resize(boards_.size());

        for (size_t i = 0; i < boards_.size(); i++) {
            auto& entry = boards_[i];
            auto& boardResult = result.boardResults[i];
            boardResult.detected = false;
            boardResult.markersDetected = 0;

            if (!entry.active || result.markerIds.empty()) continue;

            std::vector<cv::Point3f> objPoints;
            std::vector<cv::Point2f> imgPoints;
            entry.board.matchImagePoints(result.markerCorners, result.markerIds, objPoints, imgPoints);

            boardResult.markersDetected = (int)objPoints.size() / 4;

            if (objPoints.size() >= 4) {
                bool useGuess = entry.detected;
                cv::Vec3d rvec = entry.rvec;
                cv::Vec3d tvec = entry.tvec;

                try {
                    bool isTwoMarker = (objPoints.size() == 8);
                    int solveMethod = cv::SOLVEPNP_SQPNP;

                    if (isTwoMarker) useGuess = false;

                    bool success = cv::solvePnP(objPoints, imgPoints, cameraMatrix_, distCoeffs_,
                                                rvec, tvec, useGuess, solveMethod);

                    if (success) {
                        // Reprojection error check
                        std::vector<cv::Point2f> projPoints;
                        cv::projectPoints(objPoints, rvec, tvec, cameraMatrix_, distCoeffs_, projPoints);
                        double totalError = 0;
                        for (size_t j = 0; j < imgPoints.size(); j++) {
                            double dx = imgPoints[j].x - projPoints[j].x;
                            double dy = imgPoints[j].y - projPoints[j].y;
                            totalError += std::sqrt(dx*dx + dy*dy);
                        }
                        double avgError = totalError / imgPoints.size();

                        // Jump detection and rejection
                        bool shouldReject = false;
                        if (entry.detected) {
                            double jumpDist = cv::norm(tvec - entry.tvec);
                            if (isTwoMarker) {
                                if (avgError > 15.0) shouldReject = true;
                            } else {
                                if (jumpDist > 0.1 && avgError > 10.0) shouldReject = true;
                            }
                        }

                        if (shouldReject) {
                            boardResult.detected = true;
                            boardResult.rvec = entry.rvec;
                            boardResult.tvec = entry.tvec;
                        } else {
                            boardResult.detected = true;
                            boardResult.rvec = rvec;
                            boardResult.tvec = tvec;
                        }
                    }
                } catch (cv::Exception&) {
                    // solvePnP failed
                }
            }
        }
    }

    // =========================================================================
    // Worker thread
    // =========================================================================

    void workerFunction() {
        // Disable OpenCV internal parallelism (GCD on macOS) in the worker
        // thread to avoid heap corruption from nested thread pool conflicts.
        cv::setNumThreads(1);

        // Worker's own detector instance — avoids sharing detector_ with main thread
        cv::aruco::ArucoDetector workerDetector(dictionary_, detectorParams_);

        // Persistent request buffer — avoids per-frame 33MB alloc/free via swap
        DetectRequest request;

        while (running_) {
            if (!hasRequest_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            // Rebuild worker detector if params changed (e.g. setMinMaxMarkerDetectionSize)
            if (workerDetectorDirty_.exchange(false)) {
                workerDetector = cv::aruco::ArucoDetector(dictionary_, detectorParams_);
            }

            {
                std::lock_guard<std::mutex> lock(requestMutex_);
                // Swap pixel buffers — O(1), both vectors keep their capacity
                request.pixelData.swap(latestRequest_.pixelData);
                request.width = latestRequest_.width;
                request.height = latestRequest_.height;
                request.channels = latestRequest_.channels;
                request.detectBoards = latestRequest_.detectBoards;
                hasRequest_.store(false);
            }

            DetectResult result;
            findMarkers(workerDetector, request.pixelData.data(),
                        request.width, request.height, request.channels, result);

            if (request.detectBoards) {
                estimateBoardPoses(result);
            }

            {
                std::lock_guard<std::mutex> lock(resultMutex_);
                latestResult_ = std::move(result);
                hasResult_.store(true);
            }
        }
    }

    // =========================================================================
    // Member variables
    // =========================================================================

    // OpenCV ArUco
    cv::aruco::Dictionary dictionary_;
    cv::aruco::DetectorParameters detectorParams_;
    cv::aruco::ArucoDetector detector_;

    // Camera parameters
    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;
    cv::Size imageSize_;

    // Settings
    float markerSize_;
    bool threaded_ = true;

    // Detection results (main thread)
    std::vector<int> markerIds_;
    std::vector<std::vector<cv::Point2f>> markerCorners_;
    std::vector<cv::Vec3d> rvecs_, tvecs_;

    // Projection matrix
    tc::Mat4 projMatrix_;

    // Board management
    std::vector<BoardEntry> boards_;
    std::mutex boardsMutex_;

    // Thread communication
    DetectRequest latestRequest_;
    std::mutex requestMutex_;
    std::atomic<bool> hasRequest_{false};

    DetectResult latestResult_;
    std::mutex resultMutex_;
    std::atomic<bool> hasResult_{false};

    // Worker detector rebuild flag
    std::atomic<bool> workerDetectorDirty_{false};

    // Worker thread
    std::thread workerThread_;
    std::atomic<bool> running_{false};
};

} // namespace tcx
