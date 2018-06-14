import u_net
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from libtiff import TIFF
import scipy
import random
import os
import shutil

def read_tif(file_path, resize=None, print_log=True):
    """

    参数
        file_path tif文件路径
        resize 对加载进来的图片进行resize操作，参数值为(w, h)或(w, h, c)格式的数组。该值默认为None, 表示不进行此项操作。
        print_log 是否打印图片信息 默认True
    """
    tif = TIFF.open(file_path, mode='r')
    image = tif.read_image()
    if not (resize is None):
        image = cv2.resize(image, resize)
    if print_log:
        print(image.shape)
    return image
def load_labelx(file_path, resize, max_value=1.0):
    labelx = read_tif(file_path, resize=resize)
    labelx[labelx < 128] = 0
    labelx[labelx > 128] = 255
    labelx = labelx / (255 / max_value)
    return labelx
def load_images(fn_list,batch_size):
    f = np.load(fn_list)
    images = np.reshape(cv2.resize(f['image'], (572, 572)), (batch_size, 572, 572, 3))
    label = f['label'] / 256
    labels = np.reshape(cv2.resize(label, (572, 572)), (572, 572, 1))
    return np.array(images), np.array(labels)
def tresult(label, p, t, f,copy=False):
    logits = p[0, :, :, 0]
    #logits = cv2.blur(logits, (5, 5))
    logits = cv2.blur(logits, (5, 5))
    logits = cv2.blur(logits, (3, 3))
    name_root = f.split('_')[-1].split('.')[0]
    scipy.misc.imsave("/usr/lhw/data/U-net/AOI_2_Vegas_Roads_Train/result/logits_new6m/%s" % name_root + '.png', logits)
    # logits = cv2.resize(logits, (w, h))
    logits[logits < t] = 0.0
    logits[logits > t] = 1.0
    #TP = logits * label, TP + FP = logits, TP + FN = label, FP = logits - logits * label, TN + FP = 1 - label
    l1 = np.sum(logits * label) / np.sum(label) # Recall
    l2 = np.sum(logits * label) / np.sum(logits) # Precision
    l3 = np.sum(logits - logits * label) / np.sum(1 - label) # FPR
    F1 = 2 * l1 * l2 / (l1 + l2) # TPR
    if copy:
        logits = logits.copy()
    return logits, l1, l2, l3, F1
def tresult_withoutlabel(p, t=0.5, copy=False):
    logits = p[0, :, :, 0]
    # logits = cv2.resize(logits, (w, h))
    logits[logits < t] = 0.0
    logits[logits > t] = 1.0

    if copy:
        logits = logits.copy()
    return logits

u = u_net.u_net("cpu:0", 0, batch_size=1, output_dim=1)
logit = tf.nn.sigmoid(u.outputs, name="pred")
saver = tf.train.Saver()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver.restore(sess,tf.train.latest_checkpoint('models/'))
count = 0
R = []
P = []
F1 = 0
TEST_PATH = '/usr/lhw/data/U-net/AOI_2_Vegas_Roads_Train/test_npz'
fn_list = [os.path.join(TEST_PATH, fn) for fn in os.listdir(TEST_PATH) if "npz" in fn]
for f in fn_list:
    count += 1
    image, label = load_images(f,1)
    p = sess.run(logit, feed_dict={u.inputs: image})
    logits, l1, l2, l3, f1= tresult(label, p, 0.3, f)
    F1 += f1
    R.append(l1)
    P.append(l2)
    print("%d: testing: %s" % (count, f))
    print("Recall: %.6f, Precision: %.6f, F1: %.6f" % (l1, l2, f1))
    name_root = f.split('_')[-1].split('.')[0]
    scipy.misc.imsave("/usr/lhw/data/U-net/AOI_2_Vegas_Roads_Train/result/new_result6m/%s" % name_root + '.png', logits)

print("F1 = %f" % (F1 / count))

