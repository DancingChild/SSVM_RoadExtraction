# -*- coding: utf-8 -*-
import numbers
import tensorflow as tf

def variable(name, regularizer_scale=0, **kwargs):
    """
    构建一个变量节点，变量名为name

    参数
        name 操作名
        regularizer_scale 正则化率，当后面的参数不存在regularizer，且该值为正实数时有效
        **kwargs 将作为tf.get_variable的参数

    返回
        变量的引用
    """
    if not isinstance(regularizer_scale, numbers.Number) or regularizer_scale < 0:
        regularizer_scale = 0
    if "regularizer" not in kwargs:
        print("\033[1;33m L2 REGULARIZER SCALE = \033[1;31m%.5f\033[0m"%regularizer_scale)
        kwargs["regularizer"] = tf.contrib.layers.l2_regularizer(float(regularizer_scale))
    ret = tf.get_variable(name, **kwargs)
    return ret

def conv2d(name_or_scope, x, regularizer_scale, kernel_shape, strides, act_func=tf.nn.relu, padding="SAME", biases_initial_value=None):
    """
    2d卷积操作

    参数
        name_or_scope 变量命名空间实例或空间名
        x 输入
        regularizer_scale 正则化率
        kernel_shape 卷积核形状 [h, w, in_channels, out_channels]
        strides 卷积步长
        padding 字符串: "SAME"或"VALID"，默认为"SAME"
        biases_initial_value 偏移量的初始值，默认为None代表不使用偏移量

    返回
        卷积操作后节点的引用
    """
    # print("ASSERT %d == %d"%(kernel_shape[-2], x.get_shape().as_list()[-1]))
    assert kernel_shape[-2] == x.get_shape().as_list()[-1]
    with tf.variable_scope(name_or_scope):
        W = variable("W", regularizer_scale, shape=kernel_shape, initializer = tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(x, W, strides=strides, padding=padding)
        if isinstance(biases_initial_value, numbers.Number):
            b = variable("b", regularizer_scale, shape=kernel_shape[-1], initializer = tf.constant_initializer(value=biases_initial_value))
            conv = tf.nn.bias_add(conv, b)
    print("%s \033[1;33m%s\033[0m"%(name_or_scope, conv.get_shape().as_list()))
    if not (act_func is None):
        conv = act_func(conv)
    return conv

def conv2d_transpose(name_or_scope, x, regularizer_scale, kernel_shape, output_shape, strides, act_func=tf.nn.relu, padding="VALID", biases_initial_value=None):
    """
    2d反卷积操作

    参数
        name_or_scope 变量命名空间实例或空间名
        x 输入
        regularizer_scale 正则化率
        kernel_shape 卷积核形状 [h, w, in_channels, out_channels]
        out_put_shape 反卷积输出形状
        strides 卷积步长
        padding 字符串: "SAME"或"VALID"，默认为"VALID"
        biases_initial_value 偏移量的初始值，默认为None代表不使用偏移量

    返回
        反卷积操作后的节点的引用
    """
    assert kernel_shape[-1] == x.get_shape().as_list()[-1]
    assert kernel_shape[-2] == output_shape[-1]
    with tf.variable_scope(name_or_scope):
        W = variable("W", regularizer_scale, shape=kernel_shape, initializer = tf.contrib.layers.xavier_initializer())
        deconv = tf.nn.conv2d_transpose(x, W, output_shape, strides=strides, padding=padding)
        if isinstance(biases_initial_value, numbers.Number):
            b = variable("b", regularizer_scale, shape=kernel_shape[-1], initializer = tf.constant_initializer(value=biases_initial_value))
            deconv = tf.nn.bias_add(deconv, b)
    print("%s \033[1;33m%s\033[0m"%(name_or_scope, deconv.get_shape().as_list()))
    if not (act_func is None):
        deconv = act_func(deconv)
    return deconv

def merge(name, layers):
    return tf.concat(layers, axis=-1, name=name)

def total_loss(name, loss):
    """
    加入正则化项后的损失

    返回
        total_loss[T], loss[L], regularization_losses[R]
        总损失         原始损失 正则化项损失
        T = L + R
    """
    with tf.variable_scope(name):
        regularization_losses = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = tf.reduce_mean(loss) + regularization_losses
    return total_loss, loss, regularization_losses

def weight_variable(shape, initial):

    init = initial(shape)
    return tf.Variable(init)

def bias_variable(shape, initial):

    init = initial(shape)
    return tf.Variable(init)
def deconv_layer(x, upscale, name, padding='SAME', w_init=None):

    x_shape = tf.shape(x)
    in_shape = x.shape.as_list()

    w_shape = [upscale * 2, upscale * 2, in_shape[-1], 1]
    strides = [1, upscale, upscale, 1]

    W = weight_variable(w_shape, w_init)
    tf.summary.histogram('weights_{}'.format(name), W)

    out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], w_shape[2]]) * tf.constant(strides, tf.int32)
    deconv = tf.nn.conv2d_transpose(x, W, out_shape, strides=strides, padding=padding)

    return deconv

def conv_layer(x, W_shape, b_shape=None, name=None,
               padding='SAME', use_bias=True, w_init=None, b_init=None):

    W =weight_variable(W_shape, w_init)
    tf.summary.histogram('weights_{}'.format(name), W)

    if use_bias:
        b = bias_variable([b_shape], b_init)
        tf.summary.histogram('biases_{}'.format(name), b)

    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

    return conv + b if use_bias else conv

def side_layer(inputs, name, upscale):
    """
        https://github.com/s9xie/hed/blob/9e74dd710773d8d8a469ad905c76f4a7fa08f945/examples/hed/train_val.prototxt#L122
        1x1 conv followed with Deconvoltion layer to upscale the size of input image sans color
    """
    with tf.variable_scope(name):
        in_shape = inputs.shape.as_list()
        w_shape = [1, 1, in_shape[-1], 1]

        classifier = conv_layer(inputs, w_shape, b_shape=1,
                                     w_init=tf.constant_initializer(),
                                     b_init=tf.constant_initializer(),
                                     name=name + '_reduction')

        classifier = deconv_layer(classifier, upscale=upscale,
                                       name='{}_deconv_{}'.format(name, upscale),
                                       w_init=tf.truncated_normal_initializer(stddev=0.1))

        return classifier
class u_net:

    def __init__(self, device_str, regularizer, batch_size, output_dim, reuse=False, name_or_scope="u-net"):
        self.graph = {}
        with tf.variable_scope(name_or_scope), tf.device(device_str):
            self.graph["inputs"] = tf.placeholder(dtype=tf.float32, shape=(batch_size,480, 480, 3))

            conv_config = {
                "name_or_scope": "conv_1_1", 
                "x": self.graph["inputs"], 
                "regularizer_scale": regularizer, 
                "kernel_shape": [3, 3, 3, 64], 
                "strides": [1]*4, 
                "padding": "SAME", 
                "biases_initial_value": 0.0
            }
            self.graph["conv_1_1"] = conv2d(**conv_config)

            conv_config["name_or_scope"] = "conv_1_2"
            conv_config["x"] = self.graph["conv_1_1"]
            conv_config["kernel_shape"] = [3, 3, 64, 64]
            self.graph["conv_1_2"] = conv2d(**conv_config)

            self.graph["side_1"] = side_layer(self.graph["conv_1_2"], "side_1", 1)

            self.graph["pool_1"] = tf.nn.max_pool(self.graph["conv_1_2"], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

            conv_config["name_or_scope"] = "conv_2_1"
            conv_config["x"] = self.graph["pool_1"]
            conv_config["kernel_shape"] = [3, 3, 64, 128]
            self.graph["conv_2_1"] = conv2d(**conv_config)

            conv_config["name_or_scope"] = "conv_2_2"
            conv_config["x"] = self.graph["conv_2_1"]
            conv_config["kernel_shape"] = [3, 3, 128, 128]
            self.graph["conv_2_2"] = conv2d(**conv_config)

            self.graph["side_2"] = side_layer(self.graph["conv_2_2"], "side_2", 2)

            self.graph["pool_2"] = tf.nn.max_pool(self.graph["conv_2_2"], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

            conv_config["name_or_scope"] = "conv_3_1"
            conv_config["x"] = self.graph["pool_2"]
            conv_config["kernel_shape"] = [3, 3, 128, 256]
            self.graph["conv_3_1"] = conv2d(**conv_config)

            conv_config["name_or_scope"] = "conv_3_2"
            conv_config["x"] = self.graph["conv_3_1"]
            conv_config["kernel_shape"] = [3, 3, 256, 256]
            self.graph["conv_3_2"] = conv2d(**conv_config)

            self.graph["side_3"] = side_layer(self.graph["conv_3_2"], "side_3", 4)

            self.graph["pool_3"] = tf.nn.max_pool(self.graph["conv_3_2"], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

            conv_config["name_or_scope"] = "conv_4_1"
            conv_config["x"] = self.graph["pool_3"]
            conv_config["kernel_shape"] = [3, 3, 256, 512]
            self.graph["conv_4_1"] = conv2d(**conv_config)

            conv_config["name_or_scope"] = "conv_4_2"
            conv_config["x"] = self.graph["conv_4_1"]
            conv_config["kernel_shape"] = [3, 3, 512, 512]
            self.graph["conv_4_2"] = conv2d(**conv_config)

            self.graph["side_4"] = side_layer(self.graph["conv_4_2"], "side_4", 8)
            self.graph["pool_4"] = tf.nn.max_pool(self.graph["conv_4_2"], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

            conv_config["name_or_scope"] = "conv_5_1"
            conv_config["x"] = self.graph["pool_4"]
            conv_config["kernel_shape"] = [3, 3, 512, 1024]
            self.graph["conv_5_1"] = conv2d(**conv_config)

            conv_config["name_or_scope"] = "conv_5_2"
            conv_config["x"] = self.graph["conv_5_1"]
            conv_config["kernel_shape"] = [3, 3, 1024, 1024]
            self.graph["conv_5_2"] = conv2d(**conv_config)

            self.graph["side_5"] = side_layer(self.graph["conv_5_2"], "side_5", 16)

           # 合并side_output layers
            self.graph["side_outputs"] = [self.graph["side_1"], self.graph["side_2"], self.graph["side_3"], self.graph["side_4"], self.graph["side_5"]]
            w_shape = [1, 1, len(self.graph["side_outputs"]), 1]
            self.fuse = conv_layer(tf.concat(self.graph["side_outputs"], axis=3),
                                        w_shape, name='fuse_1', use_bias=False,
                                        w_init=tf.constant_initializer(0.2))

            self.graph["up_1"] = conv2d_transpose("up_1", self.graph["conv_5_2"], regularizer, [2, 2, 512, 1024], self.graph["conv_4_2"].get_shape().as_list(), [1, 2, 2, 1])

            self.graph["merge_1"] = merge("merge_1", [self.graph["up_1"], self.graph["conv_4_2"]])

            conv_config["name_or_scope"] = "conv_6_1"
            conv_config["x"] = self.graph["merge_1"]
            conv_config["kernel_shape"] = [3, 3, 1024, 512]
            self.graph["conv_6_1"] = conv2d(**conv_config)
            
            conv_config["name_or_scope"] = "conv_6_2"
            conv_config["x"] = self.graph["conv_6_1"]
            conv_config["kernel_shape"] = [3, 3, 512, 512]
            self.graph["conv_6_2"] = conv2d(**conv_config)

            self.graph["up_2"] = conv2d_transpose("up_2", self.graph["conv_6_2"], regularizer, [2, 2, 256, 512], self.graph["conv_3_2"].get_shape().as_list(), [1, 2, 2, 1])

            self.graph["merge_2"] = merge("merge_2", [self.graph["up_2"], self.graph["conv_3_2"]])

            conv_config["name_or_scope"] = "conv_7_1"
            conv_config["x"] = self.graph["merge_2"]
            conv_config["kernel_shape"] = [3, 3, 512, 256]
            self.graph["conv_7_1"] = conv2d(**conv_config)

            conv_config["name_or_scope"] = "conv_7_2"
            conv_config["x"] = self.graph["conv_7_1"]
            conv_config["kernel_shape"] = [3, 3, 256, 256]
            self.graph["conv_7_2"] = conv2d(**conv_config)

            self.graph["up_3"] = conv2d_transpose("up_3", self.graph["conv_7_2"], regularizer, [2, 2, 128, 256], self.graph["conv_2_2"].get_shape().as_list(), [1, 2, 2, 1])

            self.graph["merge_3"] = merge("merge_3", [self.graph["up_3"], self.graph["conv_2_2"]])

            conv_config["name_or_scope"] = "conv_8_1"
            conv_config["x"] = self.graph["merge_3"]
            conv_config["kernel_shape"] = [3, 3, 256, 128]
            self.graph["conv_8_1"] = conv2d(**conv_config)

            conv_config["name_or_scope"] = "conv_8_2"
            conv_config["x"] = self.graph["conv_8_1"]
            conv_config["kernel_shape"] = [3, 3, 128, 128]
            self.graph["conv_8_2"] = conv2d(**conv_config)

            self.graph["up_4"] = conv2d_transpose("up_4", self.graph["conv_8_2"], regularizer, [2, 2, 64, 128], self.graph["conv_1_2"].get_shape().as_list(), [1, 2, 2, 1])

            self.graph["merge_4"] = merge("merge_4", [self.graph["up_4"], self.graph["conv_1_2"]])

            conv_config["name_or_scope"] = "conv_9_1"
            conv_config["x"] = self.graph["merge_4"]
            conv_config["kernel_shape"] = [3, 3, 128, 64]
            self.graph["conv_9_1"] = conv2d(**conv_config)

            conv_config["name_or_scope"] = "conv_9_2"
            conv_config["x"] = self.graph["conv_9_1"]
            conv_config["kernel_shape"] = [3, 3, 64, 64]
            self.graph["conv_9_2"] = conv2d(**conv_config)

            conv_config["name_or_scope"] = "outputs"
            conv_config["x"] = self.graph["conv_9_2"]
            conv_config["kernel_shape"] = [3, 3, 64, output_dim]
            conv_config["act_func"] = None
            self.graph["outputs"] = conv2d(**conv_config)
        self.__dict__.update(self.graph)


