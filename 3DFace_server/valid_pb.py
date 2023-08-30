from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import time

import h5py
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import pickle
# from queue import Queue
import random,threading,time
import utli
import train_function
import dataPath
from multiprocessing import Process,Queue
from tensorflow.python.platform import gfile

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def main(args):

    a = np.zeros((1,2,3))
    b = a.shape
    c = b[0]

    true_data_dirs = dataPath.trueValidFilePaths
    false_data_dirs = dataPath.falseValidFilePaths

    N = len(true_data_dirs)
    true_paths_raw = []
    for i in range(N):
        true_paths_raw += utli.get_dataset_common(true_data_dirs[i])
    num_true = len(true_paths_raw)
    N = len(false_data_dirs)
    false_paths_raw = []
    for i in range(N):
        false_paths_raw += utli.get_dataset_common(false_data_dirs[i])
    num_false = len(false_paths_raw)
    all_paths = true_paths_raw+false_paths_raw
    print("true num:%d  false num:%d"%(num_true,num_false))


    args.batch_size = num_true + num_false
    pb_files = args.pb_path
    with gfile.FastGFile(pb_files, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        img_placeholder,Predictions = tf.import_graph_def(graph_def, return_elements=["Placeholder:0","mobilenet/out:0"])
        print(img_placeholder, Predictions)



    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())

        label = np.zeros((num_false+num_true))
        label[0:num_true] = 1
        data = np.zeros((num_false+num_true,args.image_h_size,args.image_w_size,1))
        print("start read true data..")
        data[0:num_true, :, :, :] = utli.get_batch_data_process(true_paths_raw,num_process=25)
        print("start read false data..")
        data[num_true:, :, :, :] = utli.get_batch_data_process(false_paths_raw,num_process=25)

        print("start cal data..")
        feed_dict = {img_placeholder: data}

        predection= sess.run(Predictions, feed_dict=feed_dict)
        #print(predection)

        trainAcc = utli.get_roc(label,predection)
        print("Far:%.6f   FRR=%.6f  ACC=%.6f"%(trainAcc[0],trainAcc[1],trainAcc[2]))
        err_index = utli.get_predection_err_index(label,predection)
        total_acc = 1 - len(err_index)/(num_false+num_true)
        print("err_num=%d"%(len(err_index)))
        print("total_acc=%.6f"%(total_acc))
        for i in err_index:
            print(all_paths[i])







def parse_arguments(argv):
    parser = argparse.ArgumentParser()


    parser.add_argument('--pb_path', type=str,
                        default='/home/wyw/3DFace_server/log/faceFlat3/pb/spc_2d_0713_v2.pb')

    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=32)
    parser.add_argument('--image_w_size', type=int,
                        help='Image size (height, width) in pixels.', default=96)
    parser.add_argument('--image_h_size', type=int,
                        help='Image size (height, width) in pixels.', default=112)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
