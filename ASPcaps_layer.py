# -*- coding: utf-8 -*-

'''
__author__ = 'Cimy'
__mtime__  = '2021/01/20'
 If you have any question, please contact me fell free.
 e-mail: wangjp85@mail2.sysu.edu.cn
'''
 
from keras import backend as K
import tensorflow as tf
import numpy as np
from keras import layers, initializers, regularizers, constraints
from keras.utils import conv_utils
from keras.layers import InputSpec
from keras.utils.conv_utils import conv_output_length
from batchdot import own_batch_dot
cf = K.image_data_format() == '..'
useGPU = True


def squeeze(s):
    sq = K.sum(K.square(s), axis=-1, keepdims=True)
    return (sq / (1 + sq)) * (s / K.sqrt(sq + K.epsilon()))


class ConvertToCaps(layers.Layer):

    def __init__(self, **kwargs):
        super(ConvertToCaps, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=2)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape.insert(1 if cf else len(output_shape), 1)
        return tuple(output_shape)

    def call(self, inputs):
        return K.expand_dims(inputs, 1 if cf else -1)

    def get_config(self):
        config = {
            'input_spec': 5
        }
        base_config = super(ConvertToCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FlattenCaps(layers.Layer):

    def __init__(self, **kwargs):
        super(FlattenCaps, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=4)

    def compute_output_shape(self, input_shape):
        if not all(input_shape[1:]):
            raise ValueError('The shape of the input to "FlattenCaps" '
                             'is not fully defined '
                             '(got ' + str(input_shape[1:]) + '. '
                             'Make sure to pass a complete "input_shape" '
                             'or "batch_input_shape" argument to the first '
                             'layer in your model.')
        return (input_shape[0], np.prod(input_shape[1:-1]), input_shape[-1])

    def call(self, inputs):
        shape = K.int_shape(inputs)
        return K.reshape(inputs, (-1, np.prod(shape[1:-1]), shape[-1]))


class CapsToScalars(layers.Layer):

    def __init__(self, **kwargs):
        super(CapsToScalars, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=3)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])

    def call(self, inputs):
        return K.sqrt(K.sum(K.square(inputs + K.epsilon()), axis=-1))

class ASPCaps_layer(layers.Layer):
    def __init__(self, ch_j = 32, n_j = 4,
                 kernel_size=(3, 3),
                 strides=(1, 1),                 
                 dilation_rate=(1, 1),
                 padding='same',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super(ASPCaps_layer, self).__init__(**kwargs)
        self.ch_j = ch_j  # Number of capsules in layer J
        self.n_j = n_j  # Number of neurons in a capsule in J
        self.filters = self.ch_j * self.n_j
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = (1, 1)
        self.deformable_groups = 1
        self.use_bias = use_bias
        self.kernel_initializer=kernel_initializer
        self.bias_initializer=bias_initializer
        self.kernel_regularizer=kernel_regularizer
        self.bias_regularizer=bias_regularizer

    def build(self, input_shape):

        self.h_i, self.w_i, self.ch_i, self.n_i = input_shape[1:5]

        self.h_j, self.w_j = [conv_utils.conv_output_length(input_shape[i + 1],
                                                            self.kernel_size[i],
                                                            padding=self.padding,
                                                            stride=self.strides[i],
                                                            dilation=self.dilation_rate[i]) for i in (0, 1)]
                                                            
        self.kernel_size = (self.kernel_size[0], self.kernel_size[1],
                                 self.ch_i * self.n_i, self.ch_j * self.n_j)
        self.kernel = self.add_weight(
            name = 'kernel',
            shape = self.kernel_size,
            initializer = self.kernel_initializer,
            regularizer = self.kernel_regularizer,
            trainable = True,
            dtype = 'float32',
            )
        if self.use_bias:
            self.bias = self.add_weight(
                name = 'bias',
                shape = (self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                dtype='float32',
                )
                
        self.offset_kernel = self.add_weight(
            name = 'offset_kernel',
            shape = (self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], 
                    3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]),
            initializer = 'zeros',
            trainable = True,
            dtype = 'float32')
        
        self.offset_bias = self.add_weight(
            name = 'offset_bias',
            shape = (3 * self.kernel_size[0] * self.kernel_size[1] * self.deformable_groups,),
            initializer='zeros',
            trainable = True,
            dtype = 'float32',
            )
        self.ks = self.kernel_size[0] * self.kernel_size[1]
        self.ph, self.pw = (self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2
        self.phw = tf.constant([self.ph, self.pw], dtype = 'int32')
        self.patch_yx = tf.stack(tf.meshgrid(tf.range(-self.phw[1], self.phw[1] + 1), tf.range(-self.phw[0], self.phw[0] + 1))[::-1], axis = -1)
        self.patch_yx = tf.reshape(self.patch_yx, [-1, 2])
        
        self.built = True
        super(ASPCaps_layer, self).build(input_shape)

    def call(self, inputs):
        x = K.reshape(inputs, (-1, self.h_i, self.w_i, self.ch_i * self.n_i))
        offset = tf.nn.conv2d(x, self.offset_kernel, strides = [1, self.strides[0], self.strides[1], 1], padding = 'SAME', dilations=[1, 1, 1, 1])
        offset += self.offset_bias
        bs, ih, iw, ic = [v.value for v in offset.shape]
        bs = tf.shape(x)[0]
        oyox, mask = offset[..., :2*self.ks], offset[..., 2*self.ks:]
        mask = tf.nn.sigmoid(mask)
        grid_yx = tf.stack(tf.meshgrid(tf.range(iw), tf.range(ih))[::-1], axis = -1)
        grid_yx = tf.reshape(grid_yx, [1, ih, iw, 1, 2]) + self.phw + self.patch_yx
        grid_yx = tf.cast(grid_yx, 'float32') + tf.reshape(oyox, [bs, ih, iw, -1, 2])
        grid_iy0ix0 = tf.floor(grid_yx)
        grid_iy1ix1 = tf.clip_by_value(grid_iy0ix0 + 1, 0, tf.constant([ih+1, iw+1], dtype = 'float32'))
        grid_iy1, grid_ix1 = tf.split(grid_iy1ix1, 2, axis = 4)
        grid_iy0ix0 = tf.clip_by_value(grid_iy0ix0, 0, tf.constant([ih+1, iw+1], dtype = 'float32'))
        grid_iy0, grid_ix0 = tf.split(grid_iy0ix0, 2, axis = 4)
        grid_yx = tf.clip_by_value(grid_yx, 0, tf.constant([ih+1, iw+1], dtype = 'float32'))
        batch_index = tf.tile(tf.reshape(tf.range(bs), [bs, 1, 1, 1, 1, 1]), [1, ih, iw, self.ks, 4, 1])
        grid = tf.reshape(tf.concat([grid_iy1ix1, grid_iy1, grid_ix0, grid_iy0, grid_ix1, grid_iy0ix0], axis = -1), [bs, ih, iw, self.ks, 4, 2])
        grid = tf.concat([batch_index, tf.cast(grid, 'int32')], axis = -1)
        delta = tf.reshape(tf.concat([grid_yx - grid_iy0ix0, grid_iy1ix1 - grid_yx], axis = -1), [bs, ih, iw, self.ks, 2, 2])
        w = tf.expand_dims(delta[..., 0], axis = -1) * tf.expand_dims(delta[..., 1], axis = -2)
        x = tf.pad(x, [[0, 0], [int(self.ph), int(self.ph)], [int(self.pw), int(self.pw)], [0, 0]])
        map_sample = tf.gather_nd(x, grid)
        map_bilinear = tf.reduce_sum(tf.reshape(w, [bs, ih, iw, self.ks, 4, 1]) * map_sample, axis = -2) * tf.expand_dims(mask, axis = -1)
        map_all = tf.reshape(map_bilinear, [bs, ih, iw, map_bilinear.shape[3]*map_bilinear.shape[4]])
        output = tf.nn.conv2d(map_all, tf.reshape(self.kernel, [1, 1, -1, self.filters]), strides = [1, 1 , 1, 1], padding = 'SAME')
        if self.use_bias:
            output += self.bias

        outputs = squeeze(K.reshape(output, ((-1, self.h_j, self.w_j, self.ch_j, self.n_j))))

        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.h_j, self.w_j, self.ch_j, self.n_j)


class CapsuleLayer(layers.Layer):

    def __init__(self, num_capsule = 16, dim_capsule = 16, channels = 0, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.channels = channels
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        if(self.channels != 0):
            assert int(self.input_num_capsule / self.channels) / (self.input_num_capsule / self.channels) == 1, "error"
            self.W = self.add_weight(shape=[self.num_capsule, self.channels,
                                            self.dim_capsule, self.input_dim_capsule],
                                     initializer=self.kernel_initializer,
                                     name='W')

            self.B = self.add_weight(shape=[self.num_capsule, self.dim_capsule],
                                     initializer=self.kernel_initializer,
                                     name='B')
        else:
            self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                            self.dim_capsule, self.input_dim_capsule],
                                     initializer=self.kernel_initializer,
                                     name='W')
            self.B = self.add_weight(shape=[self.num_capsule, self.dim_capsule],
                                     initializer=self.kernel_initializer,
                                     name='B')

        self.built = True

    def call(self, inputs, training=None):
        inputs_expand = K.expand_dims(inputs, 1)

        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        if(self.channels != 0):
            W2 = K.repeat_elements(self.W, int(self.input_num_capsule / self.channels), 1)
        else:
            W2 = self.W
        inputs_hat = K.map_fn(lambda x: own_batch_dot(x, W2, [2, 3]), elems=inputs_tiled)

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.input_num_capsule].
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = tf.nn.softmax(b, dim=1)

            # c.shape =  [batch_size, num_capsule, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
            # outputs.shape=[None, num_capsule, dim_capsule]
            outputs = squash(own_batch_dot(c, inputs_hat, [2, 2]) + self.B)  # [None, 10, 16]

            if i < self.routings - 1:
                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                b += own_batch_dot(outputs, inputs_hat, [2, 3])
        # End: Routing algorithm -----------------------------------------------------------------------#

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])


def _squash(input_tensor):
    norm = tf.norm(input_tensor, axis=-1, keep_dims=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))


def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm)
    return scale * vectors


