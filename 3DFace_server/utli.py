import os
import sys
import random,threading,time
import numpy as np
import cv2
from multiprocessing import Process,Queue
from absl import logging
import tensorflow as tf
from six import iteritems


imgW = 96
imgH = 112

class Producer(threading.Thread):
    def __init__(self, name,queue,data_dirs, nrof_epochs,batch_size):
        threading.Thread.__init__(self, name=name)
        self.img=queue
        self.nrof_epochs = nrof_epochs
        self.batch_size = batch_size
        N = len(data_dirs)
        self.paths_raw =[]
        for i in range(N):
            self.paths_raw += get_dataset_common(data_dirs[i])

        self.nrof_images = len(self.paths_raw)
        self.index = np.linspace(0,self.nrof_images-1,self.nrof_images).astype(np.int)
        self.nrof_batchs = self.nrof_images//batch_size

        print("%s img num: %d"%(name, self.nrof_images))

    def run(self):
        for k in range(self.nrof_epochs):
            random.shuffle(self.index)
            for i in range(self.nrof_batchs):
                start_index = i * self.batch_size
                end_index = min((i + 1) * self.batch_size, self.nrof_images)
                temp_index = self.index[start_index:end_index]
                paths_batch = [self.paths_raw[i] for i in temp_index]
                images = get_batch_data(paths_batch, do_augmentation=True, do_flip=False)
                self.img.put(images)

        print("%s finished!" % self.getName())


def get_data_process(name,queue,data_dirs, nrof_epochs,batch_size):
    N = len(data_dirs)
    paths_raw = []
    for i in range(N):
        paths_raw += get_dataset_common(data_dirs[i])

    nrof_images = len(paths_raw)

    index = [i for i in range(nrof_images)]
    nrof_batchs = nrof_images // batch_size
    print("%s img num: %d" % (name, nrof_images))
    for k in range(nrof_epochs):
        random.shuffle(index)
        for i in range(nrof_batchs):
            start_index = (int)(i *batch_size)
            end_index = min((i + 1) * batch_size, nrof_images)
            temp_index = index[start_index:end_index]
            paths_batch = [paths_raw[i] for i in temp_index]


            images = get_batch_data_process(paths_batch, do_augmentation=True, do_flip=False)
            queue.put(images)




def get_batch_data_process(filePaths, do_augmentation=False, do_flip=False, num_process = 5):
    N = len(filePaths)
    if N%num_process==0:
        batch_size = N//num_process
    else:
        batch_size = N//num_process+1
    imgs = np.zeros((N,imgH,imgW,1)).astype(np.float32)
    img_queue = Queue(maxsize=num_process)
    for i in range(num_process):
        start_index = (int)(i * batch_size)
        end_index = min((i + 1) * batch_size, N)
        filePath = filePaths[start_index:end_index]
        time.sleep(max(0,(i-20)*0.15))
        img_process = Process(target=get_batch_data, args=(filePath,False,do_flip,img_queue))
        img_process.start()

    end_index = 0
    for i in range(num_process):
        img = img_queue.get()
        #print(img)
        shape = img.shape
        start_index = end_index
        end_index = end_index + shape[0]
        imgs[start_index:end_index,:,:,:] = img
    return imgs
np.set_printoptions(threshold=sys.maxsize)
def get_batch_data(filePaths, do_augmentation=False, do_flip=False, img_quene=None):
    N = len(filePaths)
    imgs = np.zeros((N,imgH,imgW,1)).astype(np.float32)
    for i in range(N):
        filePath = filePaths[i]
        img = cv2.imread(filePath, -1)
        if len(img.shape) == 2:
            img = cv2.merge((img,img, img))
        img = img[:,:,1].astype(np.float32)
        if do_flip:
            img=cv2.flip(img,1)
        #process_img(img)

        if np.max(img) - np.min(img) == 0:
            img = 0
        elif 'light' in filePath or 'sznormal' in filePath or "GMdatas" in filePath or "gm0707" in filePath or "true_valid" in filePath or "office" in filePath:
            #if np.min(img) == 0:
            #    continue
            if np.max(img) - np.min(img) != 0:
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
            else:
                img = 0
        else:
            img = img * 2
            img[img == 0] = 405
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
        imgs[i,:,:,0] = img
    if img_quene != None:
        img_quene.put(imgs)

    return imgs

def get_dataset_path_from_list(dir_list):
    N = len(dir_list)
    paths_raw = []
    for i in range(N):
        paths_raw += get_dataset_common(dir_list[i])
    return paths_raw

def get_dataset_common(input_dir, min_images = 1):
  ret = []
  person_names = []
  for person_name in os.listdir(input_dir):
    person_names.append(person_name)
  person_names = sorted(person_names)
  for person_name in person_names:
    _subdir = os.path.join(input_dir, person_name)
    if not os.path.isdir(_subdir):
      continue
    for img in os.listdir(_subdir):
        filePath = os.path.join(_subdir, img)
        ret.append(filePath)
  return ret

def cal_mean(img):
    sum = 0
    cont = 0
    for i in range(imgH):
        for j in range(imgW):
            if(img[i,j]>200 and img[i,j]<900):
            # if (img[i, j] > 1):
                sum += img[i,j]
                cont +=1
    if cont==0:
        mean = 0
    else:
        mean = (float)(sum/cont)
    return mean

def process_img(img):

    mean = cal_mean(img)
    offset = mean-100
    for i in range(imgH):
        for j in range(imgW):
            if (img[i, j] > 1):
                img[i, j] -= offset
                # img[i, j] /= 100.0


def learnrate(filename,turn):
    file=open(filename)
    lines=file.readlines()
    for line in lines:
        line=line.split("\t")
        if turn>=int(line[0]):
            learn_rate=float(line[1])
        else:
            learn_rate =learn_rate
            break
    return learn_rate


def get_vars_to_restore(ckpt=None):
  """Returns list of variables that should be saved/restored.

  Args:
    ckpt: Path to existing checkpoint.  If present, returns only the subset of
        variables that exist in given checkpoint.

  Returns:
    List of all variables that need to be saved/restored.
  """
  model_vars = tf.trainable_variables()
  # Add batchnorm variables.
  bn_vars = [v for v in tf.global_variables()
             if 'moving_mean' in v.op.name or 'moving_variance' in v.op.name]
  model_vars.extend(bn_vars)
  model_vars = sorted(model_vars, key=lambda x: x.op.name)
  if ckpt is not None:
    ckpt_var_names = tf.contrib.framework.list_variables(ckpt)
    ckpt_var_names = [name for (name, unused_shape) in ckpt_var_names]
    for v in model_vars:
      if v.op.name not in ckpt_var_names:
        logging.warn('Missing var %s in checkpoint: %s', v.op.name,
                     os.path.basename(ckpt))
    model_vars = [v for v in model_vars if v.op.name in ckpt_var_names]
  return model_vars

def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))

def get_roc(label_batches,predictions_soft):
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    # true_score=[]
    # fasle_score=[]

    predictions=np.argmax(predictions_soft)
    #print(predictions_soft)
    for i in range(len(label_batches)):
        predictions = np.argmax(predictions_soft[i])
        #print(predictions)
        if np.argmax(label_batches[i]) == 1 and predictions == 1:
            TP += 1
            # true_score.append(predictions_soft[i,1])
        elif np.argmax(label_batches[i]) == 1 and predictions == 0:
            FN += 1
            # true_score.append(predictions_soft[i,1])
        elif np.argmax(label_batches[i]) == 0 and predictions == 1:
            FP += 1
            # fasle_score.append(predictions_soft[i,0])
        elif np.argmax(label_batches[i]) == 0 and predictions== 0:
            TN += 1
            # fasle_score.append(predictions_soft[i,0])

    FAR = 1
    if (TN + FP) > 0.00001:
        FAR = 1.0 * FP / (TN + FP)

    FRR = 1
    if (TP + FN) > 0.00001:
        FRR = 1.0 * FN / (TP + FN)
    acc = (TN+TP)/(len(label_batches))
    print(len(label_batches))
    return FRR,FAR, acc

def get_predection_err_index(label_batches,predictions_soft):
    err_index=[]
    for i in range(len(label_batches)):
        predictions = np.argmax(predictions_soft[i])
        if label_batches[i] != predictions:
            err_index.append(i)
    return err_index
