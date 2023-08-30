#include "cam_calibration.h"

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <string>
#include <iostream>

using namespace cv;




/**
 * estimate calibration result
 */
static double computeReprojectionErrors(const std::vector<std::vector<Point3f> >& objectPoints,
                                        const std::vector<std::vector<Point2f> >& imagePoints,
                                        const std::vector<Mat>& rvecs, const std::vector<Mat>& tvecs,
                                        const Mat& cameraMatrix, const Mat& distCoeffs,
                                        std::vector<float>& perViewErrors )
{
    std::vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for( i = 0; i < (int)objectPoints.size(); i++ )
    {
        projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
                      cameraMatrix, distCoeffs, imagePoints2);
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);
        int n = (int)objectPoints[i].size();
        perViewErrors[i] = (float)std::sqrt(err*err/n);
        totalErr += err*err;
        totalPoints += n;
    }

    return std::sqrt(totalErr/totalPoints);
}





/**
 * init params for calibration
 */

void params_setting(Ptr<SimpleBlobDetector>& detector)
{
    SimpleBlobDetector::Params blobParam;

    // blobParam.minThreshold = 5;
    blobParam.maxThreshold = 255;


    blobParam.filterByArea = true;
    blobParam.minArea = 10;
    // blobParam.maxArea = 25000;
    blobParam.minDistBetweenBlobs = 5;

    blobParam.filterByCircularity = false;
    blobParam.minCircularity = 0.9;

    blobParam.filterByConvexity = true;
    blobParam.minConvexity = 0.87;

    blobParam.filterByInertia = true;
    blobParam.minInertiaRatio = 0.01;

    detector  = SimpleBlobDetector::create(blobParam);
}



/**
 * write camera internal parameters in files
 */
static void save_cam_params(const std::string& filenames,
                            const Mat& cameraMatrix,
                            const Mat& distCoeffs,
                            const std::vector<Mat>& rvecs,
                            const std::vector<Mat>& tvecs,
                            double totalAvgErr)
{
    FileStorage fs(filenames, FileStorage::WRITE);

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "avg rms err" << totalAvgErr;

}




bool spc_cam_calib(std::vector<cv::String>& images, const std::string& filename)
{
    Ptr<SimpleBlobDetector> Detector;
    params_setting(Detector);

    TermCriteria criteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 50, 0.001);

    std::vector<std::vector<Point3f> > objpoints;
    std::vector<std::vector<Point2f> > imgpoints;
    std::vector<Point3f> objp;

    for (int i = 0; i < 26; i++)
        for (int j = 0; j < 3; j++)
            objp.push_back(Point3f(i * 17.5, j * 17.5, 0));

    Size pattern(3, 26);

    Mat result;

    for (int i = 0; i < images.size(); i++)
    {
        Mat img = imread(images[i]);
        cvtColor(img, result, COLOR_BGR2GRAY);

        // // 如果标定失败，可以考虑改善图像质量，以便于检测到圆。
        Mat enhanceImg;
        equalizeHist(result, enhanceImg);


        std::vector<KeyPoint> keypoints;
        Mat keyPoints, keyPoints_gray;

        Detector -> detect(enhanceImg, keypoints);
        drawKeypoints(img, keypoints, keyPoints);

        std::vector<Point2f> corners;
        cvtColor(keyPoints, keyPoints_gray, COLOR_BGR2GRAY);

        bool ret = findCirclesGrid(keyPoints_gray, pattern, corners, CALIB_CB_SYMMETRIC_GRID, Detector);
        std::cout << "detect result " << ret << std::endl;

        if (ret)
        {
            cornerSubPix(keyPoints_gray, corners, Size(3,3), Size(-1,-1), criteria);
            drawChessboardCorners(img, pattern, corners, ret);

            objpoints.push_back(objp);
            imgpoints.push_back(corners);
        }

        //imshow("result", img);
        //waitKey(1000);
    }

    if (objpoints.empty())
        return false;

    Mat cameraMatrix, distCoeffs;
    std::vector<Mat> rvecs;
    std::vector<Mat> tvecs;
    calibrateCamera(objpoints, imgpoints, Size(768, 1280), cameraMatrix, distCoeffs, rvecs, tvecs, 0, criteria);

    std::vector<float> perErr;
    double errs = computeReprojectionErrors(objpoints, imgpoints, rvecs, tvecs, cameraMatrix, distCoeffs, perErr);
    save_cam_params(filename, cameraMatrix, distCoeffs, rvecs, tvecs, errs);

    return true;
}
