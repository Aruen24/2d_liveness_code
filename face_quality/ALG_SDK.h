//created by zjr

#ifndef __ALG_SDK__H
#define __ALG_SDK__H


#ifdef __cplusplus
extern "C" {
#endif



typedef enum
{
		SPC_ERR_NONE 						= 0,             	// 识别成功
		SPC_ERR_MOUTH_OPEN 					= 200,     			// 张嘴，被劫持

		SPC_ERR_UNDER_EXPOSURE 				= 201,     			// 人脸图像欠曝(过暗)
		SPC_ERR_OVER_EXPOSURE 				= 202,     			// 人脸图像过曝
		
		SPC_ERR_ANTIFAKE_2D 				= 300,    			// 2D假体
		SPC_ERR_ANTIFAKE_3D 				= 301,    			// 3D假体

		SPC_ERR_NO_FACE 					= 600,        		// 未检测到人脸
		SPC_ERR_MULTI_PERSONS 				= 601,  			// 检测到多人脸

		SPC_ERR_TOO_SAMLL 					= 602,      		// 人脸过小
		SPC_ERR_TOO_BIG 					= 603,       		// 人脸过大
		SPC_ERR_NO_ATTN 					= 604,        		// 未正视摄像头
		
		SPC_ERR_TOO_LEFT 					= 605,       		// 人脸太靠左
		SPC_ERR_TOO_RIGHT 					= 606,      		// 人脸太靠右
		SPC_ERR_TOO_UP 						= 607,         		// 人脸太靠上
		SPC_ERR_TOO_DOWN 					= 608, 	  			// 人脸太靠下

		SPC_ERR_LEFT_ROLL					= 630,		  		// 翻滚角度，表示左侧歪头
		SPC_ERR_RIGHT_ROLL 					= 631,	  			// 翻滚角度，表示右侧歪头

		SPC_ERR_UP_PITCH 					= 640,       		// 俯仰角度，表示上仰头
		SPC_ERR_DOWN_PITCH 					= 641,     			// 俯仰角度，表示俯视头

		SPC_ERR_LEFT_YAW 					= 650,       		// 偏航角度，表示左侧摇头
		SPC_ERR_RIGHT_YAW 					= 651,      		// 偏航角度，表示右侧摇头

		SPC_ERR_EXSIT 						= 700,          	// 重复注册
		SPC_ERR_NOT_EXSIT 					= 701,      		// 人脸不存在录入库中

		SPC_ERR_FAILED 						= 800          		// 识别失败

}SPC_ERROR_MSG;



/**
 * @param ir_img            寰呰瘑鍒殑绾㈠鍥�
 * @param speckle_img       寰呰瘑鍒殑鏁ｆ枒鍥�
 * @param depth_img         寰呰瘑鍒殑娣卞害鍥�
 * @param face_feature      瀛樻斁寰呭尮閰嶈瘑鍒殑浜鸿劯鐗瑰緛(澶у皬涓�512)
 * @return 		            0 浜鸿劯璇嗗埆鎴愬姛
 * @return 		            SPC_ERR_NO_FACE 琛ㄧず鏈娴嬪埌浜鸿劯
 * @return 		            SPC_ERR_ANTIFAKE_2D 琛ㄧず2D鍋囦綋鏀诲嚮
 * @return 		            SPC_ERR_ANTIFAKE_3D 琛ㄧず3D鍋囦綋鏀诲嚮
 * @return                  SPC_ERR_FAILED 琛ㄧず鍖归厤澶辫触
 */
SPC_ERROR_MSG spc_face_recognition(uint8_t *ir_img, uint8_t *speckle_img, uint8_t *depth_img, int *uid);



/**
 * @param ir_img            寰呯敤浜庢敞鍐岀殑绾㈠鍥�
 * @param uid               娉ㄥ唽id鍙�
 * @return 		            0 浜鸿劯娉ㄥ唽鎴愬姛
 * @return 		            SPC_ERR_NO_FACE 琛ㄧず鏈娴嬪埌浜鸿劯
 * @return 		            SPC_ERR_EXIST   uid宸茶娉ㄥ唽
 */
SPC_ERROR_MSG spc_face_register(uint8_t *ir_img, uint8_t *speckle_img, uint8_t *depth_img, uint16_t uid);



/**
 * @param uid               寰呭垹闄ょ殑id鍙�
 * @return 		            0 浜鸿劯鍒犻櫎鎴愬姛
 */
SPC_ERROR_MSG spc_face_remove(uint16_t uid);

int spc_query_face(int *counts, uint16_t *ids);

//鏌ヨ杩斿洖flash涓繚瀛樼殑浜鸿劯鏁�
static int fatfs_query_Ids_Num(int *counts);

void spc_empty_face();

void enchance_contrast(uint8_t *src, int h,int w,float ratio, uint8_t *dst);

float *fd_res;

#ifdef __cplusplus
extern
#endif
#endif
