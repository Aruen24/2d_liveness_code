1. 3DFace_server
2D活体检测代码，数据路径在dataPath.py里面。

2. cam_calibration
产线标定，每组需要5张图像，每张检测结果 detect result = 0/1 ，
其中1表示检测成功。大于3张检测成功则表示标定成功。
把相机参数保存在cam_param.yaml中。


3. create_datasets
处理数据集的脚本，里面用到的人脸检测算法，可以替换。
16bit转8bit需要除以4  但是物奇的深度值单位是0.5mm所以除以2转化成mm，  所以整体是除以8

4. 2dliveness_quant
2D活体检测模型量化训练
替换NDK目录下的两个文件：
./ndk/examples/example_lenet5_with_basic_sim_quant_layer.py
./ndk/examples/data_generator_imagenet_partial.py
深度图是16bit的  这个进行普通定点化到16bit就行  不需要量化训练。  量化后和pb的准确率应该是保持一致的 。 所以保证pb性能就行 

5.face_quality
当前是评估整张图的图像质量

6. demosV14
姿态检测
姿态代码就是代码仓库上最新V14版本。
看HeadAngle就行。

