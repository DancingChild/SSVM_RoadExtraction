# -*- coding: utf-8 -*-


"""
MIT License

Copyright (c) 2017 sli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import numbers
import collections
import numpy as np
import tensorflow as tf
import h5py

name_change={'u-net/conv_1_1/W':'block1_conv1_W_1:0',
'u-net/conv_1_1/b':'block1_conv1_b_1:0',
'u-net/conv_1_2/W':'block1_conv2_W_1:0',
'u-net/conv_1_2/b':'block1_conv2_b_1:0',
'u-net/conv_2_1/W':'block2_conv1_W_1:0',
'u-net/conv_2_1/b':'block2_conv1_b_1:0',
'u-net/conv_2_2/W':'block2_conv2_W_1:0',
'u-net/conv_2_2/b':'block2_conv2_b_1:0',
'u-net/conv_3_1/W':'block3_conv1_W_1:0',
'u-net/conv_3_1/b':'block3_conv1_b_1:0',
'u-net/conv_3_2/W':'block3_conv2_W_1:0',
'u-net/conv_3_2/b':'block3_conv2_b_1:0',
'u-net/conv_3_3/W':'block3_conv3_W_1:0',
'u-net/conv_3_3/b':'block3_conv3_b_1:0',
'u-net/conv_4_1/W':'block4_conv1_W_1:0',
'u-net/conv_4_1/b':'block4_conv1_b_1:0',
'u-net/conv_4_2/W':'block4_conv2_W_1:0',
'u-net/conv_4_2/b':'block4_conv2_b_1:0',
'u-net/conv_4_3/W':'block4_conv3_W_1:0',
'u-net/conv_4_3/b':'block4_conv3_b_1:0',
'#u-net/conv_5_1/W':'block5_conv1_W_1:0',
'#u-net/conv_5_1/b':'block5_conv1_b_1:0',
'#u-net/conv_5_2/W':'block5_conv2_W_1:0',
'#u-net/conv_5_2/b':'block5_conv2_b_1:0',
'#u-net/conv_5_3/W':'block5_conv3_W_1:0',
'#u-net/conv_5_3/b':'block5_conv3_b_1:0'}
def get_var(name, reuse=False, regularizer_scale=0, *args, **kwargs):
    """
    构建一个变量节点，变量名为name/var

    该实现通过控制变量空间tf.variable_scope(name, reuse)来控制变量的重用性

    参数
        name 操作名
        reuse 是否重用
        regularizer_scale 正则化率，当后面的参数不存在regularizer，且该值为正实数时有效
        *args **kwargs 将作为tf.get_variable的参数
    """
    if not isinstance(regularizer_scale, numbers.Number) or regularizer_scale < 0:
        regularizer_scale = 0
    if "regularizer" not in kwargs:
        print("\033[1;33m L2 REGULARIZER SCALE = \033[1;31m%.5f\033[0m"%regularizer_scale)
        kwargs["regularizer"] = tf.contrib.layers.l2_regularizer(float(regularizer_scale))
    with tf.variable_scope(name, reuse=reuse):
        ret = tf.get_variable("var", *args, **kwargs)
    print("\033[1;35m %s ->\n\033[1;33m%s\033[0m"%(ret.name, kwargs["regularizer"]))
    return ret

def lrelu(x, leak=0.1, name="lrelu"):
    """
    Leaky-Relu
    """
    return tf.maximum(x, leak*x, name)

def batch_norm(x, is_training, reuse=False, bn_decay=0.5, epsilon=1e-3, name="batch_norm"):
    """
    Batch Normalization

    参数
        x 输入
        is_training 是否为训练阶段

    ------------------------------
    # Old version:
    return tf.contrib.layers.batch_norm(
            x, 
            decay=bn_decay, 
            scale=True, 
            center=True,
            epsilon=epsilon, 
            is_training=is_training, 
            updates_collections=None, 
            reuse=reuse,
            scope=name)
    """
    with tf.variable_scope(name, reuse=reuse):
        x_mean, x_variance = tf.nn.moments(x, list(range(len(x.get_shape()) - 1)), name="moments")

        beta = tf.Variable(tf.constant(0.0, shape=x_mean.get_shape()), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=x_mean.get_shape()), name='gamma', trainable=True)
        moving_mean = tf.Variable(tf.constant(0.0, shape=x_mean.get_shape()), name='moving_mean', trainable=False)
        moving_variance = tf.Variable(tf.constant(1.0, shape=x_mean.get_shape()), name='moving_variance', trainable=False)

        tf.add_to_collection("batch_norm", beta)
        tf.add_to_collection("batch_norm", gamma)
        tf.add_to_collection("batch_norm", moving_mean)
        tf.add_to_collection("batch_norm", moving_variance)
        
        moving_mean_op = moving_mean * bn_decay + x_mean * (1 - bn_decay)
        moving_variance_op = moving_variance * bn_decay + x_variance * (1 - bn_decay)

        update_moving_mean = tf.assign(moving_mean, moving_mean_op)
        update_moving_variance = tf.assign(moving_variance, moving_variance_op)
        
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

        if isinstance(is_training, bool):
            mean, variance = (x_mean, x_variance) if is_training else (moving_mean_op, moving_variance_op)
        else:
            mean, variance = tf.cond(is_training, lambda: (x_mean, x_variance), lambda: (moving_mean_op, moving_variance_op))
        return tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon, name="normalization")

def get_collection(scope=None, key=tf.GraphKeys.GLOBAL_VARIABLES):
    """
    获取tensorflow变量列表var_list

    参数
        scope 一个或一组tensorflow变量空间名
        key tensorflow Graph键
    返回
        变量列表
    """
    if not isinstance(scope, collections.Iterable):
        scope = [scope]
    var_list = []
    for s in scope:
        var_list += tf.get_collection(key=key, scope=s)
    #print(var_list)
    return var_list

_check_dict = {}

def set_check(key, value):
    """
    将变量值value添加到检查点
    """
    if isinstance(value, tf.Tensor):
        if key not in _check_dict:
            _check_dict[key] = []
        _check_dict[key].append(value)
    else:
        raise Exception("参数value: %s不是一个Tensor"%value)

def run_check(sess, feed_dict=[], pause=True):
    fmt = "\033[1;33mCHECK \033[1;35m%s ->%d\n\033[1;36m%s\033[0m"
    for k, v in _check_dict.items():
        value = sess.run(v, feed_dict=feed_dict)
        print(fmt%(k, len(value), value))
    if pause:
        input("任意键继续......")

def _var_dict(var_list, var_dict):
    for var in var_list:
        var_name = var.name.split(":")[0]
        if var_name in var_dict:
            raise Exception("重复存储<%s>"%var_name)
        var_dict[var_name] = var
    return var_dict

def merge_list2dict(var_list, var_dict):
    return _var_dict(var_list, var_dict)

def _save_weights(sess, *var_list, **var_dict):
    """
    Save weights use numpy.savez_compressed
    """
    var_dict = _var_dict(var_list, var_dict)
    for key in var_dict:
        var_dict[key] = var_dict[key].eval(session=sess)
    return var_dict

def save_weights(sess, file_name, *var_list, **var_dict):
    """
    使用numpy.savez_compressed保存网络权重到npz文件

    参数
        sess tensorflow 会话
        file_name 保存文件名
        var_list 待保存变量列表
        var_dict 待保存变量字典，键为保存该变量时的命名
    """
    np.savez_compressed(file_name, **_save_weights(sess, *var_list, **var_dict))

def asyn_save_weights(sess, file_name, th_pool, *var_list, **var_dict):
    """
    异步保存网络权重

    参数
        参见save_weights
        th_pool 线程池 basic_util.thread_pool
    """
    th_pool.add_task(
        np.savez_compressed, 
        file_name, 
        **_save_weights(sess, *var_list, **var_dict))

def load_weights(file_name):
    """
    使用numpy.load加载权重

    参数
        file_name 权重文件名
    返回
        numpy.load返回值
    """
    weights = np.load(file_name)
    # print("\n\033[1;34mRELOAD\033[0m ".join(weights.files))
    return weights

def load_h5weight(file_name, *var_list, **var_dict):
    f = h5py.File(file_name)
    dict_weight= { }
    for layer, g in f.items():
        for name ,d in g.items():
            #print(name)
            #print(type(d.value))
            dict_weight[name] = d.value
    var_dict = _var_dict(var_list, var_dict)
    for key in var_dict:
        new_key = 'a'
        if key in name_change:
            new_key = name_change[key]
        if new_key not in dict_weight:
            # raise Exception("没有发现%s的对应数据"%key)
            print("\033[1;31m没有发现%s的对应数据\033[0m"%key)
            continue
        print("\033[1;34mRELOAD\033[0m %s"%key)
        var_dict[key] = var_dict[key].assign(dict_weight[new_key])
    return list(var_dict.values())


def load_weights_op(file_name, *var_list, **var_dict):
    """
    加载网络权重，并赋值

    参数
        file_name 权重文件名
        var_list 待赋值变量列表
        var_dict 待赋值变量字典，键为保存该变量时的命名
    返回
        赋值tensorflow操作列表
    """
    weights = load_weights(file_name)
    print(" !!!!!weights:%s"%type(weights))
    var_dict = _var_dict(var_list, var_dict)
    for key in var_dict:
        if key not in weights:
            # raise Exception("没有发现%s的对应数据"%key)
            print("\033[1;31m没有发现%s的对应数据\033[0m"%key)
            continue
        print("\033[1;34mRELOAD\033[0m %s"%key)
        var_dict[key] = var_dict[key].assign(weights[key])
    return list(var_dict.values())

def get_sess():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    # tf_config.log_device_placement = True
    return tf.Session(config=tf_config)

def deconv2d(x, output_shape, k_h = 5, k_w = 5, strides =[1, 2, 2, 1], biases_init_value=None, regularizer_scale=0.0001, name='deconv2d'):
    """
    解卷积层
    """
    with tf.variable_scope(name):
        weights = get_var(
            name='w', 
            regularizer_scale=regularizer_scale,
            shape=[k_h, k_w, output_shape[-1], x.get_shape()[-1]], 
            initializer = tf.contrib.layers.xavier_initializer())
        deconv = tf.nn.conv2d_transpose(x, weights, output_shape, strides = strides)
        if isinstance(biases_init_value, numbers.Number):
            biases = get_var(
                name='b', 
                regularizer_scale=regularizer_scale,
                shape=[output_shape[-1]], 
                initializer = tf.constant_initializer(value=biases_init_value))
            deconv = tf.nn.bias_add(deconv, biases) 
        return deconv
            
def conv2d(x, output_dim, k_h=5, k_w=5, strides=[1, 2, 2, 1], biases_init_value=None, padding="SAME", regularizer_scale=0.0001, name='conv2d'):
    """
    卷积层
    """
    with tf.variable_scope(name):
        weights = get_var(
            name='w', 
            regularizer_scale=regularizer_scale,
            shape=[k_h, k_w, x.get_shape()[-1], output_dim], 
            initializer = tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(x, weights, strides = strides, padding = padding)
        if isinstance(biases_init_value, numbers.Number):
            biases = get_var(
                name='b', 
                regularizer_scale=regularizer_scale,
                shape=[output_dim], 
                initializer = tf.constant_initializer(value=biases_init_value))
            conv = tf.nn.bias_add(conv, biases)
        return conv

def linear(x, output_dim, biases_init_value=None, regularizer_scale=0.0001, name="linear"):
    """
    线性单元
    """
    input_dim = x.get_shape().as_list()[-1]
    y = tf.reshape(x, shape=[-1, input_dim], name="x")
    with tf.variable_scope(name):
        weights = get_var(
            name='w', 
            regularizer_scale=regularizer_scale, 
            shape=[input_dim, output_dim], 
            initializer = tf.contrib.layers.xavier_initializer())
        y = tf.matmul(y, weights)
        if isinstance(biases_init_value, numbers.Number):
            biases = get_var(
                name='b', 
                regularizer_scale=regularizer_scale,
                shape=[output_dim], 
                initializer = tf.constant_initializer(value=biases_init_value))
            y = tf.nn.bias_add(y, biases)
        return y

# GAN判别器损失
def discriminator_loss(labels, logits, D, name="discriminator_loss"):
    with tf.variable_scope(name):
        dr = D(labels, name="D")
        dp = D(logits, name="D", reuse=True)
        lossR = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=dr,
                labels=tf.ones_like(dr)
            )
        )
        lossP = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=dp,
                labels=tf.zeros_like(dp)
            )
        )
        lossG = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=dp,
                labels=tf.ones_like(dp)
            )
        )
        return lossR + lossP, lossG, lossR, lossP

def get_expdecay_lr(init_lr, decay_steps, global_step, decay_factor=0.5, staircase=True, name="learning_rate"):
    """
    指数衰减的学习率

    参数
        init_lr 初始学习率
        decay_steps 更新学习率的间隔步数
        global_step 全局迭代次数
        decay_factor 学习率的衰减率
        staircase 是否为离散的更新学习率
        name 命名

    返回
        学习率
    """
    lr = tf.train.exponential_decay(
        learning_rate=init_lr,
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=decay_factor,
        staircase=staircase,
        name=name
    )
    return lr

def add_scalar_summary(*var_list, **var_dict):
    var_dict = _var_dict(var_list, var_dict)
    with tf.device("cpu:0"):
        for k, v in var_dict.items():
            print("\033[1;31mSCALAR SUMMARY <%s>\033[0m"%k)
            tf.summary.scalar(k, tf.reduce_mean(v))


def optimizer_op(opt, grads, global_step, name):
    """
    执行梯度更新

    参数
        opt 优化器
        grads 执行更新的梯度
        global_step
        name

    返回
        更新操作
    """
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt_op = opt.apply_gradients(grads, global_step=global_step, name=name)
    return opt_op

def avg_grads(tower_grads):
    """
    用于一机多卡时，计算梯度均值
    参数
        tower_grads 形如
            [
                [(gpu0_g1, v1), (gpu0_g2, v2), ... , (gpu0_gn, vn)],
                [(gpu1_g1, v1), (gpu1_g2, v2), ... , (gpu1_gn, vn)],
                ...
            ]
    返回
        [(avg_g1, v1), (avg_g2, v2), ... , (avg_gn, vn)]
    """
    avg_grads = []
    for gv in zip(*tower_grads):
        """
        此时gv形如
            [(gpu0_g1, v1), (gpu1_g1, v1), ... , (gpuk_g1, v1)]
        """
        grads = []
        for g, v in gv:
            if g is None:
                print("\033[1;36mNONE GRADS -> %s\033[0m"%v.name)
            else:
                exp_g = tf.expand_dims(g, axis=0)
                grads.append(exp_g)
        if len(grads) > 0:
            grad = tf.reduce_mean(tf.concat(grads, axis=0), axis=0)
            var = gv[0][1]
            avg_grads.append((grad, var))
    print(len(avg_grads))
    return avg_grads

def _shortcut(x, out_shape, stride, regularizer_scale=0.0001, version="v2"):
    """
    残差网络(resnet)shortcut分支
    如果输入x与out_shape的shape不相等，则通过stride步长的1*1卷积操作使x的shape满足预期。

    参数
        regularizer_scale 在v1版本中有效
    """
    if x.get_shape() != out_shape:
        print("\033[1;35mUSE SHORTCUT-\033[1;31m%s\033[0m"%version)
        print("SHORTCUT: %s -> %s"%(x.get_shape(), out_shape))
        if version == "v1":
            x = conv2d(x, out_shape[-1], k_h = 1, k_w = 1, strides =[1, stride, stride, 1], regularizer_scale=regularizer_scale, name="short_cut_conv")
            print(x.get_shape(), out_shape)
        elif version == "v2":
            n_pad = int((out_shape[-1] - x.get_shape()[-1])//2)
            x = tf.nn.avg_pool(x, [1, stride, stride, 1], [1, stride, stride, 1], 'SAME')
            x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [n_pad, n_pad]])
            print(x.get_shape(), out_shape, n_pad, type(n_pad))
        else:
            raise Exception("未实现的版本%s！"%version)
    print("SHORTCUT %s->%s"%(x.get_shape(), out_shape))
    assert x.get_shape()[1] == out_shape[1] and x.get_shape()[2] == out_shape[2]
    return x

def _bnrelu(x, is_training, name, leak=0):
    with tf.variable_scope(name):
        return lrelu(batch_norm(x, is_training), leak=leak)


def _preact(x, mode, is_training):
    y = _bnrelu(x, is_training, "preact")
    if mode == "both_preact":
        return y, y
    elif mode == "no_preact":
        return x, y
    else:
        return x, x

def residual_block(x, ch_out, stride, preact_mode, block_mode, is_training, regularizer_scale, name):
    """
    参差块
    
    参数
        x 输入张量
        ch_out 输出通道数
        stride 卷积步长
        preact_mode 预激活模式选择<str> both_preact, no_preact, default
        block_mode 参差块模式选择<str> bottleneck default
        regularizer_scale 正则化率
        is_training 是否为训练阶段
        name 命名
    返回
        参差块输出
    """
    with tf.variable_scope(name):
        x, y = _preact(x, preact_mode, is_training)
        if block_mode == "bottleneck":
            y = conv2d(y, 
                ch_out, k_h=1, k_w=1, 
                strides=[1, 1, 1, 1], 
                regularizer_scale=regularizer_scale,
                name="conv_0")
            y = _bnrelu(y, is_training, "bnrelu_0")
            y = conv2d(y, 
                ch_out, k_h=3, k_w=3, 
                strides=[1, stride, stride, 1], 
                regularizer_scale=regularizer_scale,
                name="conv_1")
            y = _bnrelu(y, is_training, "bnrelu_1")
            y = conv2d(y, 
                ch_out*4, k_h=1, k_w=1, 
                strides=[1, 1, 1, 1], 
                regularizer_scale=regularizer_scale,
                name="conv_2")
        else:
            y = conv2d(y, 
                ch_out, k_h=3, k_w=3, 
                strides=[1, stride, stride, 1], 
                regularizer_scale=regularizer_scale,
                name="conv_0")
            y = _bnrelu(y, is_training, "bnrelu_0")
            y = conv2d(y, 
                ch_out, k_h=3, k_w=3, 
                strides=[1, 1, 1, 1], 
                regularizer_scale=regularizer_scale,
                name="conv_1")
        return y + _shortcut(x, y.get_shape(), stride)
    
class default_value:
    pass

def dict_v(dict_obj, key, default=default_value()):
    """
    获取字典值
    
    参数
        dict_obj 字典对象
        key 键
        default 键对应值不存在时，使用此值作为默认值
    返回
        dict_obj[key]
    """
    assert isinstance(dict_obj, dict)
    if key not in dict_obj:
        if isinstance(default, default_value):
            raise Exception("对象%s中不存在键%s。"%(dict_obj, key))
        else:
            dict_obj[key] = default
    return dict_obj[key]

class net:
    """
    自定义网络基类
    """

    def __init__(self, **kwargs):
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def __call__(self, x, *args, **kwargs):
        raise Exception("未实现的根类引用")


class resnet(net):
    """
    残差网络
    
    引用
    He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 770-778.
    [https://arxiv.org/pdf/1512.03385.pdf]
    """
    def __init__(self, resnet_config):
        net.__init__(**locals())

    def __call__(self, x, num_outputs, name="resnet", regularizer_scale=0.0001, is_training=False, reuse=False, first_k_hw=(7, 7)):
        """
        参数
            self
            x 输入
            num_outputs 当该值为int类型的实例时，将激活最后两层global_avg_pool & fully_connected，并输出shape=[batch_size, num_outputs]的向量，其他情况下，输出最后一个残差块经过bnrelu操作后的输出
            name 网络名称，将作为整个网络最高级的变量空间名
            regularizer_scale 正则化率
            is_training 指定是否为训练阶段，python-bool类型，或者为tf.bool类型的计算节点
            reuse 网络权重是否重用 python-bool类型
            first_k_hw 第一个卷积层的卷积核大小 k_h, k_w=first_k_hw
        返回
            构建完成的网络输出的引用
        """
        with tf.variable_scope(name, reuse=reuse) as scope:
            return resnet._build(**locals())

    def _build(self, x, **kwargs):
        regularizer_scale = dict_v(kwargs, "regularizer_scale", 0.0001)
        y = batch_norm(x, kwargs["is_training"], name="first_bn")
        first_k_h, first_k_w = kwargs["first_k_hw"]
        y = conv2d(y, 
            64, k_h=first_k_h, k_w=first_k_w,
            strides=[1, 2, 2, 1], 
            regularizer_scale=regularizer_scale,
            name="conv_0")
        y = _bnrelu(y, dict_v(kwargs, "is_training"), "bnrelu_0")
        y = tf.nn.max_pool(y, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", data_format='NHWC', name="pool_0")
        for i, c in enumerate(self.resnet_config):
            with tf.variable_scope("group_%d"%i):
                for j in range(dict_v(c, "n_blocks")):
                    if j==0:
                        if i==0:
                            preact_mode = "no_preact"
                        else:
                            preact_mode = "both_preact"
                    else:
                        preact_mode = "default"
                    y = residual_block(
                        y, dict_v(c, "ch_out"), 
                        stride=dict_v(c, "stride", 1) if j==0 else 1, 
                        preact_mode=preact_mode, 
                        block_mode=dict_v(c, "block_mode", "bottleneck"), 
                        is_training=dict_v(kwargs, "is_training"), 
                        regularizer_scale=regularizer_scale,
                        name="block_%d"%j)
        y = _bnrelu(y, dict_v(kwargs, "is_training"), "bnrelu_last")
        if isinstance(kwargs["num_outputs"], int):
            y = tf.reduce_mean(y, axis=[1, 2], name="global_avg_pool")
            print("\033[1;33mGLOBAL_AVG_POOL FEATURES SIZE: \033[1;35m%s\033[0m"%y.get_shape())
            y = linear(y, 
                output_dim=dict_v(kwargs, "num_outputs"), 
                biases_init_value=0.0, 
                regularizer_scale=regularizer_scale, 
                name="fully_connected")
            """
            y = tf.contrib.layers.fully_connected(
                inputs=y, 
                num_outputs=dict_v(kwargs, "num_outputs"), 
                activation_fn=tf.identity,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                reuse=dict_v(kwargs, 'reuse'),
                scope="fully_connected"
            )
            """
        return y

def resnetiny():
    """
    conv_0           |             = 1
    residual_blocks  | 2+2+2       = 6
    [fully_connected]|             = 1
    -----------------+-------------|---
    +                              = 8
    """
    _resnetiny = resnet([
        {"n_blocks":1,"ch_out":64,"stride":1,"block_mode":"basic"},
        {"n_blocks":1,"ch_out":128,"stride":2,"block_mode":"basic"},
        {"n_blocks":1,"ch_out":512,"stride":2,"block_mode":"basic"}
    ])
    return _resnetiny

def resnet18():
    """
    resnet-18
    conv_0           |             =  1
    residual_blocks  |(2+2+2+2)*2  = 16
    [fully_connected]|             =  1
    -----------------+-------------|---
    +                              = 18
    """
    _resnet18 = resnet([
        {"n_blocks":2,"ch_out":64,"stride":1,"block_mode":"basic"},
        {"n_blocks":2,"ch_out":128,"stride":2,"block_mode":"basic"},
        {"n_blocks":2,"ch_out":256,"stride":2,"block_mode":"basic"},
        {"n_blocks":2,"ch_out":512,"stride":2,"block_mode":"basic"}
    ])
    return _resnet18

def resnet34():
    """
    resnet-34
    conv_0           |             =  1
    residual_blocks  |(3+4+6+3)*2  = 32
    [fully_connected]|             =  1
    -----------------+-------------|---
    +                              = 34
    """
    _resnet34 = resnet([
        {"n_blocks":3,"ch_out":64,"stride":1,"block_mode":"basic"},
        {"n_blocks":4,"ch_out":128,"stride":2,"block_mode":"basic"},
        {"n_blocks":6,"ch_out":256,"stride":2,"block_mode":"basic"},
        {"n_blocks":3,"ch_out":512,"stride":2,"block_mode":"basic"}
    ])
    return _resnet34

def resnet50():
    """
    resnet-50
    conv_0           |             =  1
    residual_blocks  |(3+4+6+3)*3  = 48
    [fully_connected]|             =  1
    -----------------+-------------|---
    +                              = 50
    """
    _resnet50 = resnet([
        {"n_blocks":3,"ch_out":64,"stride":1,"block_mode":"bottleneck"},
        {"n_blocks":4,"ch_out":128,"stride":2,"block_mode":"bottleneck"},
        {"n_blocks":6,"ch_out":256,"stride":2,"block_mode":"bottleneck"},
        {"n_blocks":3,"ch_out":512,"stride":2,"block_mode":"bottleneck"}
    ])
    return _resnet50

def resnet101():
    """
    resnet-101
    conv_0           |             =  1
    residual_blocks  |(3+4+23+3)*3 = 99
    [fully_connected]|             =  1
    -----------------+-------------|---
    +                              =101
    """
    _resnet101 = resnet([
        {"n_blocks":3,"ch_out":64,"stride":1,"block_mode":"bottleneck"},
        {"n_blocks":4,"ch_out":128,"stride":2,"block_mode":"bottleneck"},
        {"n_blocks":23,"ch_out":256,"stride":2,"block_mode":"bottleneck"},
        {"n_blocks":3,"ch_out":512,"stride":2,"block_mode":"bottleneck"}
    ])
    return _resnet101

def top_k_error(predictions, labels, k, summary_name=None):
    with tf.device("cpu:0"):
        in_topk = tf.to_float(tf.nn.in_top_k(predictions, labels, k=k))
        topkerr = 1 - tf.reduce_mean(in_topk)
        if summary_name is not None:
            tf.summary.scalar("error/top_%d_error/%s"%(k, summary_name), topkerr)
        else:
            tf.summary.scalar("error/top_%d_error"%k, topkerr)
        return topkerr

class AE(net):
    """
    自动编码机
    """

    def __init__(self, encoder, decoder):
        net.__init__(**locals())
        self.c = None

    def __call__(self, x, name="AE", train=False, reuse=False):
        with tf.variable_scope(name, reuse=reuse) as scope:
            return AE._build(**locals())

    def _build(self, x, **kwargs):
        self.c = self.encoder(x, name="encoder", 
            train=kwargs['train'], reuse=kwargs['reuse'])
        _y = self.decoder(self.c, name="decoder",
            train=kwargs['train'], reuse=kwargs['reuse'])
        return _y

    def get_code(self):
        """
        获取编码结果
        """
        if self.c is None:
            raise Exception("网络未建立！")
        return self.c
        
