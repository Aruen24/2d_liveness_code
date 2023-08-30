#include "image_quality_detect.h"
#include "iot_simd_api.h"
#include "iot_cam_spro_module.h"
#include "iot_uart_common.h"
#include <math.h>


#define PIXEL_REGION 16


SPC_ERROR_MSG image_exposure_check(uint8_t *image, int size)
{
    uint16_t dark_count = 0;
    uint16_t brightness_count = 0;
    uint16_t pixel_count[256] = {0};

    float dark_prop = 0;
    float brightness_prop = 0;

    //calc image's histogram
    for (uint32_t i = 0; i < size; i++)
        pixel_count[image[i]]++;

    for (uint32_t low = 0; low < PIXEL_REGION * 4; low++)
        dark_count += pixel_count[low];

    for (uint32_t high = PIXEL_REGION * 12; high < 256; high++)
        brightness_count += pixel_count[high];

    
    dark_prop = dark_count / size;
    brightness_prop = brightness_count / size;

    IOT_LOG_INFO(" exposure test %f %f \n", dark_prop, brightness_prop);

    if (dark_prop > 0.6)
        return SPC_ERR_UNDER_EXPOSURE;
    else if (brightness_prop > 0.15)
        return SPC_ERR_OVER_EXPOSURE;
    else
        return SPC_ERR_NONE;

}


void x_gradient(uint8_t *src, int8_t *dst)
{
    for (uint32_t i = 0; i < 480; i++)
    {
        for (uint32_t j = 1; j < 288 - 1; j++)
        {
            dst[i * 288 + j] = (int8_t)((src[i * 288 + j-1] - src[i * 288 +j + 1]) / 2);
        }
    }
}


void y_gradient(uint8_t *src, int8_t *dst)
{
    for (uint32_t i = 1; i < 480 - 1; i++)
    {
        for (uint32_t j = 0; j < 288; j++)
        {
            dst[i * 288 + j] = (int8_t)((src[(i - 1)* 288 + j] - src[(i + 1) * 288 + j]) / 2);
        }
    }
}



bool_t blur_detect(uint8_t *image)
{
    float *grad = (float *)os_mem_malloc(0, sizeof(float) * 480 * 288);
    float sums = 0;   
    float size = 478*286;

    //calc laplacian grad
    for (uint32_t i = 1; i < 480 - 1; i++)
    {
        for (uint32_t j = 1; j < 288 - 1; j++)
        {
            grad[i * 288 + j] = ((float)image[i * 288 + j-1] + (float)image[i * 288 + j+1] + (float)image[(i - 1)* 288 + j] + (float)image[(i + 1)* 288 + j]) - 
                                      (4 * (float)image[i * 288 + j]);
            sums += grad[i * 288 + j];
        }
    }

    float means = sums / size;
    IOT_LOG_INFO(" grad %f  %f  %f  %f means is %f sums is %f \n", grad[0], grad[1], grad[769], grad[770], means, sums);
    float var = 0;
    for (uint32_t i = 1; i < 480 - 1; i++)
    {
        for (uint32_t j = 1; j < 288 - 1; j++)
        {
            float temp = grad[i * 288 + j] - means;
            float diff = temp * temp;
            var += diff;
            //IOT_LOG_INFO(" grad %f temp %f diff %f var %f \n", grad[i * PIC_WIDTH + j], temp, diff, var);
        }
    }

    float threshold = var / size;

    IOT_LOG_INFO(" image blur threshold is %f  means is %f  var is %f \n", threshold, means, var);

    os_mem_free(grad);

    if (threshold > 40) return true;
    else                return false;
    
}



void Haar_wavelet_transform(uint8_t *src, uint8_t *dst)
{
	int height = 480;
	int width = 288;

    float *horizontal = (float *)os_mem_malloc(0, sizeof(float)*480*288);

	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width/2; j++)
		{
			float meanPixel = (src[i*288 + 2*j] + src[i*288 + 2*j+1]) / 2;
            horizontal[i*288 + j] = meanPixel;
            horizontal[i*288 + j+144] = src[i*288 + 2*j] - meanPixel;

		}
	}
	for(int i = 0; i < height/2; i++)
	{
		for(int j = 0; j < width; j++)
		{
			float meanPixel = (horizontal[2*i*288 + j] + horizontal[(2*i + 1)*288 + j]) / 2;
            dst[i*width + j] = (uint8_t)meanPixel;
            dst[(i+240)*width + j] = (uint8_t)(horizontal[2*i*width + j] - meanPixel);
		}
	}

    os_mem_free(horizontal);
}


void minMaxLoc(uint8_t *image, int x, int y, int width, int height, float *maxValue)
{
    uint8_t *scratch = (uint8_t *)os_mem_malloc(0, 12 * width * height);
    uint8_t *out     = (uint8_t *)os_mem_malloc(0, width * height);

    iot_simd_crop_resize_uint8(image, out, 1, 480, 288, height, width,
                              y, x, height, width, scratch);

    *maxValue = image[0];
    for (int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
            if (out[i*width + j] > (*maxValue))
                *maxValue = out[i*width + j];
    }

    os_mem_free(scratch);
    os_mem_free(out);
}

void getEmax(uint8_t *src, uint8_t *dst, int scale)
{
	int height = 480;
	int width = 288;
	int h_scaled = height / scale;
	int w_scaled = width / scale;

	for(int i = 0; i<h_scaled; i++)
	{
		for(int j = 0; j<w_scaled; j++)
		{
			float maxValue;
            minMaxLoc(src, scale*j, scale*i, scale, scale, &maxValue);
            dst[i*w_scaled + j] = (uint8_t)maxValue;
		}
	}
}



void equalizeHist(uint8_t *src, uint8_t *dst)
{
    uint32_t pixel_count[256] = {0};
    uint32_t cdf[256] = {0};

    //calc image's histogram
    for (uint32_t i = 0; i < 480*288; i++)
    {
        pixel_count[src[i]]++;
    }

    uint32_t mmin = pixel_count[0];
    uint32_t sums = 0;

    //calc image's cumulative distribution function
    for (uint32_t j = 0; j < 256; j++)
    {
        sums += pixel_count[j];
        cdf[j] = sums ;

        if (pixel_count[j] < mmin)  mmin = pixel_count[j];
    }

    float demi = (480*288) - mmin;

    for(uint32_t k = 0; k < 480*288; k++)
    {
        float temp = cdf[src[k]] - mmin;
        float value = (temp / demi) * 255;
        dst[k] = (uint8_t)value; 
        //IOT_LOG_INFO(" dst %d src %d ", dst[k], src[k]);
        
    }

}

