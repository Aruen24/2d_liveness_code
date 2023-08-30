//create by zjr
#ifndef __CAM_CALIBRATION_H
#define __CAM_CALIBRATION_H

#include "opencv2/core.hpp"
#include <string>
#include <vector>

/**
 * @param images           containers, including all images
 * @param filename         output file's name
 * @retun [out]            camera parameters will be saved in output file
 */
bool spc_cam_calib(std::vector<cv::String>& images,
                   const std::string& filename);
#endif
