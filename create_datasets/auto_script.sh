#!/bin/bash


path= /home/tao/workspace/zhongjiarunWork/datas/fake0913/A4100g
save_path= /home/tao/workspace/zhongjiarunWork/datas/train_false_2d_0913
cd $path
for file in `ls`
do
	if [ -d $file ]
	then
		cd $file
		for subdir in `ls`
		do
			if [ -d $subdir ]
			then
				cd $subdir
				python3 /home/tao/workspace/zhongjiarunWork/download/create_datasets/read_raw.py
				dep="_depth"
				ir="_ir"
				images=$subdir$ir

				if [ ! -d $save_path/$file ];then
					mkdir $save_path/$file
				fi

				mkdir $subdir$dep && mkdir $subdir$ir
				mv *-depth.png  $subdir$dep && mv *-ir.png  $subdir$ir
				python3 /home/tao/workspace/zhongjiarunWork/download/create_datasets/DBFace/main.py $images
				python3 /home/tao/workspace/zhongjiarunWork/download/create_datasets/cropFaceDepth.py $subdir $images
				mv *_face.png $subdir$dep && mv *raw.png $subdir$dep
				mv $subdir$dep $save_path/$file
				cd ..
			fi
		done

		cd ..
		#rename 's/_depth.png' *.png
	fi
done
