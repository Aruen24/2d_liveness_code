#-*- coding:utf-8 -*-

import os
import sys
import glob
import cv2
import shutil
import numpy as np


np.set_printoptions(threshold=sys.maxsize)

#dirs = glob.glob("/home/kazyun/xianImages/xianDatas/*/*/")
#print(dirs, len(dirs))

#for dr in dirs:
#    direct = dr + "_png"
#    if os.path.exists(direct):
#        pass
#    else:
#        os.makedirs(direct)
path = "/home/tao/workspace/zhongjiarunWork/datas/fake0913/A4100g"
imgs = glob.glob(path+"/*.bin")
print(imgs)


save_dir_ir = path + "_ir"
save_dir_depth = path + "_depth"
if not os.path.exists(save_dir_ir):
    os.makedirs(save_dir_ir)
if not os.path.exists(save_dir_depth):
    os.makedirs(save_dir_depth)

for ig in imgs:

    if "ir" not in ig:
        img = np.fromfile(ig, dtype=np.uint16)
        img.shape = (1280, 768)

        img = img / 8
        img = img.astype(np.uint8)
        new_img = img
        name = ig.strip().split(".")[0]
        name = name.replace(path,save_dir_depth)
        cv2.imwrite(name + ".png", new_img)

    else:
        img = np.fromfile(ig, dtype=np.uint8)
        img.shape = (1280, 768)
        name = ig.strip().split(".")[0]
        name = name.replace(path, save_dir_ir)
        cv2.imwrite(name + ".png", img)


"""
ref_imgs = glob.glob("/home/kazyun/Downloads/6_7产测参考图/*.raw")
for ref in ref_imgs:
    img = np.fromfile(ref, dtype=np.uint8)
    temp = []

    for i in range(len(img)):
        temp_8bits = "{:08b}".format(img[i])
        temp.append(list(temp_8bits))

    new_img = np.array(temp)
    new_img = np.reshape(new_img.astype(int), (1280, 768))
    new_img[new_img > 0] = 255

    name = ref.strip().split(".")[0]
    cv2.imwrite(name + ".png", new_img)

"""
