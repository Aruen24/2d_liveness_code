#include "cam_calibration.h"
#include "opencv2/core.hpp"

#include <iostream>
#include <vector>
int main( int argc, char** argv )
{

    std::vector<cv::String> images;
    const std::string filename = "cam_param.yaml";
    cv::glob("/home/tao/workspace/zhongjiarunWork/download/cam_calibration/tpp/2/*.png", images);


    bool res = spc_cam_calib(images, filename);

    return 0;
}
