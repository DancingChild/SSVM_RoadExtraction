import matplotlib.pyplot as plt
import numpy as np
import cv2
from libtiff import TIFF
import random
import os

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
def get_wh(image):
    h, w = image.shape[:2]
    return w, h

def load_labelx(file_path, resize=None):
    labelx = read_tif(file_path, resize=resize)
    labelx[labelx < 128] = 0
    labelx[labelx > 128] = 255
    return labelx
def load_pnglabel(file_path):
    label = cv2.imread(file_path)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    label[label < 128] = 0
    label[label > 128] = 255
    return label

def rand_range(a, b, size=None):
    return (b - a) * np.random.random(size) + a

def random_cut(w, h):
    a = rand_range(0, 0.62)
    b = rand_range(0, 0.62)
    c = rand_range(0.38, 0.99 - a)
    d = rand_range(0.38, 0.99 - b)
    leftup = (int(a * w), int(b * h))
    rightdown = (int((a + c) * w), int((b + d) * h))
    return leftup, rightdown

def cut_image(image, leftup, rightdown):
    image_copy = image.copy()
    sub_image = image_copy[leftup[1]:rightdown[1], leftup[0]:rightdown[0]]
    return sub_image

def display_subimage(image, leftup, rightdown, color=(255, 0, 0)):
    plt.figure(figsize=(10, 10))
    image_copy = image.copy()
    x = cv2.rectangle(image_copy, leftup, rightdown, color, 5)
    plt.imshow(image_copy)

def pair_cut(c, l):
    cw, ch = get_wh(c)
    lw, lh = get_wh(l)
    assert cw == lw and ch == lh
    leftup, rightdown = random_cut(cw, ch)
    sub_clip = cut_image(c, leftup, rightdown)
    sub_label = cut_image(l, leftup, rightdown)
    return sub_clip, sub_label, leftup, rightdown

path_images = '/home/ly/data/road_data/image/train'
path_labels = '/home/ly/data/road_data/image/train_label'
path_out = '/home/ly/data/road_data/train2m_npz'
im_files = os.listdir(path_images)
j=1
for im_file in im_files:
    if not im_file.endswith('.tif'):
        continue
    name_root = im_file.split('_')[-1].split('.')[0]

    im_file_images = os.path.join(path_images, im_file)
    #im_file_labels = os.path.join(path_labels, 'RGB-PanSharpen_AOI_2_Vegas_' + name_root + '_mask' + '.tif')
    im_file_labels = os.path.join(path_labels, 'mask_' + name_root  + '.png')
    im_file_out = os.path.join(path_out, 'AOI_2_Vegas_' + name_root + '.npz')
    clipx = read_tif(im_file_images)
    labelx = load_pnglabel(im_file_labels)
    np.savez_compressed(file=im_file_out, image=clipx, label=labelx)
    print("SAVE %d%% | %s" % (j, im_file_out))
    j = j + 1

