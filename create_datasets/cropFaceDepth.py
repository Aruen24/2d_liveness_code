#-*- coding:utf-8 -*-

import glob
import cv2
import sys
import numpy as np
# import tkinter
import matplotlib.pyplot as plt


# if len(sys.argv) <= 1:
#         print("parameter error!")
#         exit()
# imgs = glob.glob(sys.argv[1] + "/*.bin")
# files = sys.argv[2] + ".txt"

binFiles_path = "/home/tao/workspace/zhongjiarunWork/datas/fake0913/A4100g"
imgs = glob.glob(binFiles_path + "/*.bin")
files = "/home/tao/workspace/zhongjiarunWork/datas/fake0913/A4100g_ir.txt"


print(files)
roi = np.genfromtxt(files, dtype='str')

for ig in roi:
        names = ig[4].strip().split("/")[-1].split("-ir")[0]
        names = binFiles_path +"/"+ names
        print(names)
        if int(ig[0]) < 0 or int(ig[1]) < 0 or int(ig[2]) < 0 or int(ig[3]) < 0 :
                continue
        else:
                img = np.fromfile(names + "-depth.bin", dtype=np.uint16)
                img2 = np.fromfile(names + "-depth.bin", dtype=np.uint16)

                img.shape=(1280, 768)
                img2.shape=(1280, 768)

                image = img[int(ig[1]): int(ig[1]) + int(ig[3]), int(ig[0]): int(ig[0]) + int(ig[2])]
                image2 = img2[int(ig[1]): int(ig[1]) + int(ig[3]), int(ig[0]): int(ig[0]) + int(ig[2])]
                image = cv2.resize(image, (96,112))
                image2 = image2 / 8
                new_img = image2.astype(np.uint8)

                cv2.imwrite(names + "_face.png", image)
                cv2.imwrite(names + "_face_8bit_raw.png", new_img)

"""
def face_depth(scripts):

        all_files = glob.glob(scripts + "*.txt")
        for txt in all_files:
                labels = np.genfromtxt(txt, dtype='str')
                files = labels[:, 4]
                bbox = labels[:, :4]
                for i, img in enumerate(files):
                #print(i, img.strip().split(".")[0] + "_depth_depth.png")
                        file_name = img.strip().split(".")[0] + "_depth_full.png"
                        image = cv2.imread(file_name, -1)
                        print(int(bbox[i, 0]), int(bbox[i, 1]), int(bbox[i, 2]), int(bbox[i, 3]))
                        image = image[int(bbox[i, 1]): int(bbox[i, 1]) + int(bbox[i, 3]), int(bbox[i, 0]): int(bbox[i, 0]) + int(bbox[i, 2]) ]
                        image = cv2.resize(image, (96, 112))
                        cv2.imwrite(img.strip().split(".")[0] + "_face_112x96.png", image)


if __name__ == '__main__':

        file_path = "/home/kazyun/compare_spc/true_decode/zjr/"
        face_depth(file_path)
"""
