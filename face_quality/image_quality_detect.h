#ifndef __IMAGE_QUALITY_DETECT__
#define __IMAGE_QUALITY_DETECT__

#include "iot_io_api.h"
#include "ALG_SDK.h"



/**
 * @param image                 红外图(适用于没加玻璃镜片)
 * @param size                  图像长度
 * @return                      0, 图像曝光度正常
 * @return                      SPC_ERR_UNDER_EXPOSURE 201, 图像欠曝
 * @return                      SPC_ERR_OVER_EXPOSURE  202, 图像过曝
 **/
SPC_ERROR_MSG image_exposure_check(uint8_t *image, int size);

/**
 * @param image                 红外图
 * @return                      1, 图像不模糊
 * @return                      0, 图像模糊
 **/
bool_t blur_detect(uint8_t *image);

/**
 * HWT 哈尔小波变换，用于图像模糊检测
 * 
 * @param image                 原始红外图
 * @param transformed           小波变换后图像
 **/
void Haar_wavelet_transform(uint8_t *image, uint8_t *transformed);

/**
 * 计算图像x-轴方向梯度
 * @param src                   原红外图
 * @param dst                   x-轴方向梯度图
 **/
void x_gradient(uint8_t *src, int8_t *dst);

/**
 * 计算图像y-轴方向梯度
 * @param src                   原红外图
 * @param dst                   y-轴方向梯度图
 **/
void y_gradietn(uint8_t *src, int8_t *dst);

/**
 * 图像直方图均衡化，增强原始图像对比度，同时会增加图像背景噪点
 * 加玻璃镜片的情况下，原红外图中有较亮的像素，增强后容易出现过曝
 * 
 * @param src                   红外图
 * @param dst                   增强后红外图
 **/
void equalizeHist(uint8_t *src, uint8_t *dst);

#endif
