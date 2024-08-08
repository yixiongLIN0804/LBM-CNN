import warnings

warnings.filterwarnings("ignore")
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1，2"  #指定第2，3，4张显卡处理



from tensorflow.keras.layers import (Input, Conv3D, UpSampling3D,Conv3DTranspose,
                                     Concatenate, MaxPooling3D,
                                     )
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import keras
import keras.backend as K
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
import pickle
import numpy as np


from mpl_toolkits.mplot3d import proj3d



# 重点是rb和r的区别，rb是打开2进制文件，文本文件用r
# f = open(r'D:\深度学习\ 3D_XT.pkl','rb')
data1 = np.load('3D_XTtrain.npz')
datax = data1['dataX']
data2 = np.load('3D_YTtrain.npz')
daty = data2['dataY']
# g = open(r'D:\深度学习\ 3D_YT.pkl','rb')
# daty = pickle.load(g)
# data = np.load('D:\深度学习\LBM_3D_train4060.npz')
# daty = data['y']
# datax = data['x']
# data = np.load('D:\深度学习\LBM_3D_test4060.npz')
# datyte = (data['y'])*10000000
# dataxte = data['x']
print(datax.shape)
print(daty.shape)
datax = tf.cast(datax,dtype=tf.float32)
# datax = datax[:,]
daty = tf.cast(daty,dtype=tf.float32)


# daty = tf.expand_dims(daty,axis=4)
# datax1 = tf.cast(datax1,dtype=tf.float32)
# daty1 = tf.cast(daty1,dtype=tf.float32)
# daty1 = tf.expand_dims(daty1,axis=4)
# print(datax.shape,daty.shape)
def resize_3D(data_3D,x,y,z):
    import numpy as np
    import tensorflow as tf
    # 读取3D数据并且检测类型
    if data_3D.ndim != 3:
        print("输入数组不为3维度数组")
    # x_data = np.size(data_3D, 0)
    # y_data = np.size(data_3D, 1)
    # z_data = np.size(data_3D, 2)
    # print(np.shape(data_3D))
    data_3D = tf.image.resize(data_3D[:, :, :], [x, y])
    # print(np.shape(data_3D))
    # print(np.shape(np.transpose(data_3D, (1, 2, 0))))
    data_3D = tf.image.resize(np.transpose(data_3D, (1, 2, 0)), [y, z])
    # print(np.shape(data_3D))
    data_3D = np.transpose(data_3D, (2, 0, 1))
    # print(np.shape(data_3D))
    return data_3D

x = np.zeros([124,256,128,128,3],dtype=np.float32)
y = np.zeros([124,256,128,128,1],dtype=np.float32)
for I in range(0, 124):
    datax1 = np.squeeze(resize_3D(datax[I, :, :, :, 0], 256, 128, 128))
    datax2 = np.squeeze(resize_3D(datax[I, :, :, :, 1], 256, 128, 128))
    datax3 = np.squeeze(resize_3D(datax[I, :, :, :, 2], 256, 128, 128))
    # # datax = np.squeeze(resize_3D(datax[I, :, :, :, :], 64, 32, 32))
    x[I, :, :, :, 0] = datax1
    x[I, :, :, :, 1] = datax2
    x[I, :, :, :, 2] = datax3
    # x[I, :, :, :, :] = datax

    daty1 = np.squeeze(resize_3D(daty[I, :, :, :,], 256, 128, 128))
    y[I, :, :, :, 0] = daty1




y = y/1000
x = tf.cast(x,dtype=tf.float32)

y = tf.cast(y,dtype=tf.float32)


x_train = x
y_train = y
#
print(x_train.shape,y_train.shape)

# 显示并对比CFD和CNN结果
def cut_3D(y_test,out,error):

    T_1 = y_test
    T_2 = out
    T_3 = error
    if y_test.ndim != 3:
        print("输入数组不为3维度数组")
    if out.ndim != 3:
            print("输入数组不为3维度数组")
    if error.ndim != 3:
        print("输入数组不为3维度数组")
    # print(T.shape)
    (x1, y1, z1) = T_1.shape
    (x2, y2, z2) = T_2.shape
    (x3, y3, z3) = T_3.shape
    # print(x)
    # print(y)
    # print(z)
    # 下刀前准备
    x1_cut = round(x1/2)
    y1_cut = round(y1/2)
    z1_cut = round(z1/2)
    x2_cut = round(x2/2)
    y2_cut = round(y2/2)
    z2_cut = round(z2/2)
    x3_cut = round(x3/2)
    y3_cut = round(y3/2)
    z3_cut = round(z3/2)
    # print(x_cut)
    # print(y_cut)
    # print(z_cut)
    # 准备好了 下刀!!
    T1_x = np.squeeze(T_1[x1_cut, :, ])
    T1_y = np.squeeze(T_1[:, y1_cut, ])
    T1_z = np.squeeze(T_1[:, :, z1_cut])
    T2_x = np.squeeze(T_2[x2_cut, :, ])
    T2_y = np.squeeze(T_2[:, y2_cut, ])
    T2_z = np.squeeze(T_2[:, :, z2_cut])
    T3_x = np.squeeze(T_3[x3_cut, :, ])
    T3_y = np.squeeze(T_3[:, y3_cut, ])
    T3_z = np.squeeze(T_3[:, :, z3_cut])
    # print(T_x.shape)
    # print(T_y.shape)
    # print(T_z.shape)
    # 构建二维网格数据
    zx1, zy1 = np.meshgrid(np.linspace(0, x1-1, x1), np.linspace(0, y1-1, y1))
    yx1, yz1 = np.meshgrid(np.linspace(0, x1-1, x1), np.linspace(0, z1-1, z1))
    xz1, xy1 = np.meshgrid(np.linspace(0, z1-1, z1), np.linspace(0, y1-1, y1))
    zx2, zy2 = np.meshgrid(np.linspace(0, x2-1, x2), np.linspace(0, y2-1, y2))
    yx2, yz2 = np.meshgrid(np.linspace(0, x2-1, x2), np.linspace(0, z2-1, z2))
    xz2, xy2 = np.meshgrid(np.linspace(0, z2-1, z2), np.linspace(0, y2-1, y2))
    zx3, zy3 = np.meshgrid(np.linspace(0, x3-1, x3), np.linspace(0, y3-1, y3))
    yx3, yz3 = np.meshgrid(np.linspace(0, x3-1, x3), np.linspace(0, z3-1, z3))
    xz3, xy3 = np.meshgrid(np.linspace(0, z3-1, z3), np.linspace(0, y3-1, y3))
    # print(zx.shape)
    # print(zy.shape)
    # print(yx.shape)
    # print(yz.shape)
    # print(xz.shape)
    # print(xy.shape)
    # create the figure
    fig = plt.figure()
    fig.suptitle('3D_cut')

    # plt.subplot(1, 3, 1)

    # ax1 = fig.add_subplot(221)
    # ax1.imshow(np.transpose(T1_x), cmap='rainbow', interpolation='nearest', origin='lower', extent=[0, y1, 0, z1])
    # ax1.set(aspect='equal')
    # ax1.set_title('x_to')
    # ax2 = fig.add_subplot(222)
    # ax2.imshow(np.transpose(T1_y), cmap='rainbow', interpolation='nearest', origin='lower', extent=[0, x1, 0, z1])
    # ax2.set(aspect='equal')
    # ax2.set_title('y_to')
    # ax3 = fig.add_subplot(223)
    # ax3.imshow(np.transpose(T1_z), cmap='rainbow', interpolation='nearest', origin='lower', extent=[0, x1, 0, y1])
    # ax3.set(aspect='equal')
    # ax3.set_title('z_to')
    norm = matplotlib.colors.Normalize(vmin=0.4, vmax=0.7)  # 设置colorbar显示的最大最小值
    ax = fig.add_subplot(131, projection='3d')
    cset = ax.contourf(np.transpose(T1_x), xy1, xz1, 100, zdir='x', offset=x1_cut, cmap='rainbow', alpha=1, norm=norm)
    cset = ax.contourf(yx1, np.transpose(T1_y), yz1, 100, zdir='y', offset=y1_cut, cmap='rainbow', alpha=1, norm=norm)
    cset = ax.contourf(zx1, zy1, np.transpose(T1_z), 100, zdir='z', offset=z1_cut, cmap='rainbow', alpha=1, norm=norm)
    ax.set_title('3D')
    ax.set_zlim((0, z1))
    ax.set_xlim((0, x1))
    ax.set_ylim((0, y1))
    ax.set(aspect='equal')
    plt.colorbar(cset)


    # plt.subplot(1, 3, 2)
    # fig.suptitle('3D_cut')
    # ax1 = fig.add_subplot(221)
    # ax1.imshow(np.transpose(T2_x), cmap='rainbow', interpolation='nearest', origin='lower', extent=[0, y2, 0, z2])
    # ax1.set(aspect='equal')
    # ax1.set_title('x_to')
    # ax2 = fig.add_subplot(222)
    # ax2.imshow(np.transpose(T2_y), cmap='rainbow', interpolation='nearest', origin='lower', extent=[0, x2, 0, z2])
    # ax2.set(aspect='equal')
    # ax2.set_title('y_to')
    # ax3 = fig.add_subplot(223)
    # ax3.imshow(np.transpose(T2_z), cmap='rainbow', interpolation='nearest', origin='lower', extent=[0, x2, 0, y2])
    # ax3.set(aspect='equal')
    # ax3.set_title('z_to')
    ax1 = fig.add_subplot(132, projection='3d')
    cset = ax1.contourf(np.transpose(T2_x), xy2, xz2, 100, zdir='x', offset=x2_cut, cmap='rainbow', alpha=1, norm=norm)
    cset = ax1.contourf(yx2, np.transpose(T2_y), yz2, 100, zdir='y', offset=y2_cut, cmap='rainbow', alpha=1, norm=norm)
    cset = ax1.contourf(zx2, zy2, np.transpose(T2_z), 100, zdir='z', offset=z2_cut, cmap='rainbow', alpha=1, norm=norm)
    ax1.set_title('3D')
    ax1.set_zlim((0, z2))
    ax1.set_xlim((0, x2))
    ax1.set_ylim((0, y2))
    ax1.set(aspect='equal')
    plt.colorbar(cset)


    # plt.subplot(1, 3, 3)
    # fig.suptitle('3D_cut')
    # ax1 = fig.add_subplot(221)
    # ax1.imshow(np.transpose(T2_x), cmap='rainbow', interpolation='nearest', origin='lower', extent=[0, y2, 0, z2])
    # ax1.set(aspect='equal')
    # ax1.set_title('x_to')
    # ax2 = fig.add_subplot(222)
    # ax2.imshow(np.transpose(T2_y), cmap='rainbow', interpolation='nearest', origin='lower', extent=[0, x2, 0, z2])
    # ax2.set(aspect='equal')
    # ax2.set_title('y_to')
    # ax3 = fig.add_subplot(223)
    # ax3.imshow(np.transpose(T2_z), cmap='rainbow', interpolation='nearest', origin='lower', extent=[0, x2, 0, y2])
    # ax3.set(aspect='equal')
    # ax3.set_title('z_to')
    ax2 = fig.add_subplot(133, projection='3d')
    cset = ax2.contourf(np.transpose(T3_x), xy3, xz3, 100, zdir='x', offset=x3_cut, cmap='rainbow', alpha=1)
    cset = ax2.contourf(yx3, np.transpose(T3_y), yz3, 100, zdir='y', offset=y3_cut, cmap='rainbow', alpha=1)
    cset = ax2.contourf(zx3, zy3, np.transpose(T3_z), 100, zdir='z', offset=z3_cut, cmap='rainbow', alpha=1)
    ax2.set_title('3D')
    ax2.set_zlim((0, z3))
    ax2.set_xlim((0, x3))
    ax2.set_ylim((0, y3))
    ax2.set(aspect='equal')
    plt.colorbar(cset)
    plt.show()
def unet():
    """ set up unet model """
    inputs = Input(shape=(256,128,128,3),name='input')
    conv1_1 = Conv3D(8, (3,3,3), padding='same', activation='relu', name='conv1_1')(inputs)
    conv1_2 = Conv3D(8, (3,3,3), padding='same', activation='relu', name='conv1_2')(conv1_1)
    conv1_3 = Conv3D(8, (3,3,3), padding='same', activation='relu', name='conv1_3')(conv1_2)

    max_pool1 = MaxPooling3D(name='max_pool1')(conv1_3)  # maxpooling2D默认stride=filer size
    conv2_1 = Conv3D(16, (3,3,3), padding='same', activation='relu', name='conv2_1')(max_pool1)
    conv2_2 = Conv3D(16, (3,3,3), padding='same', activation='relu', name='conv2_2')(conv2_1)
    conv2_3 = Conv3D(16, (3,3,3), padding='same', activation='relu', name='conv2_3')(conv2_2)

    max_pool2 = MaxPooling3D(name='max_pool2')(conv2_3)
    conv3_1 = Conv3D(32, (3,3,3), padding='same', activation='relu', name='conv3_1')(max_pool2)
    conv3_2 = Conv3D(32, (3,3,3), padding='same', activation='relu', name='conv3_2')(conv3_1)
    conv3_3 = Conv3D(32, (3,3,3), padding='same', activation='relu', name='conv3_3')(conv3_2)

    max_pool3 = MaxPooling3D(name='max_pool3')(conv3_3)
    conv4_1 = Conv3D(64, (3,3,3), padding='same', activation='relu', name='conv4_1')(max_pool3)
    conv4_2 = Conv3D(64, (3,3,3), padding='same', activation='relu', name='conv4_2')(conv4_1)
    conv4_3 = Conv3D(64, (3,3,3), padding='same', activation='relu', name='conv4_3')(conv4_2)

    max_pool4 = MaxPooling3D(name='max_pool4')(conv4_3)
    conv5_1 = Conv3D(128, (3,3,3), padding='same', activation='relu', name='conv5_1')(max_pool4)
    conv5_2 = Conv3D(128, (3,3,3), padding='same', activation='relu', name='conv5_2')(conv5_1)
    conv5_3 = Conv3D(128, (3,3,3), padding='same', activation='relu', name='conv5_3')(conv5_2)

    max_pool5 = MaxPooling3D(name='max_pool5')(conv5_3)
    conv6_1 = Conv3D(256, (3, 3, 3), padding='same', activation='relu', name='conv6_1')(max_pool5)
    conv6_2 = Conv3D(256, (3, 3, 3), padding='same', activation='relu', name='conv6_2')(conv6_1)
    conv6_3 = Conv3D(256, (3, 3, 3), padding='same', activation='relu', name='conv6_3')(conv6_2)

    up6_1 = UpSampling3D(name='up6_1')(conv6_3)
    up6_conv_1 = Conv3D(128, (2,2,2), padding='same', name='up6_conv_1')(up6_1)
    conv5_feature_1 = Conv3DTranspose(128, (2,2,2), strides=(2,2,2), padding='same', use_bias=False,activation='relu')(conv6_3)
    # up5_2 = UpSampling2D(name='up5_2')(conv5_3)
    # up5_conv_2 = Conv2D(32, 2, padding='same', name='up5_conv_2')(up5_2)
    # conv4_feature_2 = tf.image.resize(conv4_3, (up5_conv_2.shape[1], up5_conv_2.shape[2]),tf.image.ResizeMethod.NEAREST_NEIGHBOR, name='conv4_feature_2')
    #
    # up5_3 = UpSampling2D(name='up5_3')(conv5_3)
    # up5_conv_3 = Conv2D(32, 2, padding='same', name='up5_conv_3')(up5_3)
    # conv4_feature_3 = tf.image.resize(conv4_3, (up5_conv_3.shape[1], up5_conv_3.shape[2]),tf.image.ResizeMethod.NEAREST_NEIGHBOR, name='conv4_feature_3')

    concat1_1 = Concatenate(name='concat1_1')([up6_conv_1, conv5_feature_1])
    # concat1_2 = Concatenate(name='concat1_2')([up5_conv_2, conv4_feature_2])
    # concat1_3 = Concatenate(name='concat1_3')([up5_conv_3, conv4_feature_3])

    conv7_1_1 = Conv3D(128, (3,3,3), padding='same', activation='relu', name='conv7_1_1')(concat1_1)
    conv7_2_1 = Conv3D(128, (3,3,3), padding='same', activation='relu', name='conv7_2_1')(conv7_1_1)
    conv7_3_1 = Conv3D(128, (3,3,3), padding='same', activation='relu', name='conv7_3_1')(conv7_2_1)

    # conv6_1_2 = Conv2D(32, 3, padding='same', activation='relu', name='conv6_1_2')(concat1_2)
    # conv6_2_2 = Conv2D(32, 3, padding='same', activation='tanh', name='conv6_2_2')(conv6_1_2)
    # conv6_3_2 = Conv2D(32, 3, padding='same', activation='selu', name='conv6_3_2')(conv6_2_2)
    #
    # conv6_1_3 = Conv2D(32, 3, padding='same', activation='relu', name='conv6_1_3')(concat1_3)
    # conv6_2_3 = Conv2D(32, 3, padding='same', activation='tanh', name='conv6_2_3')(conv6_1_3)
    # conv6_3_3 = Conv2D(32, 3, padding='same', activation='selu', name='conv6_3_3')(conv6_2_3)
    #
    up7_1 = UpSampling3D(name='up7_1')(conv7_3_1)
    up7_conv_1 = Conv3D(64, (2,2,2), padding='same', name='up7_conv_1')(up7_1)
    conv4_feature_1 = Conv3DTranspose(64, (2,2,2), strides=(2,2,2), padding='same', use_bias=False,activation='relu')(conv5_3)

    # up6_2 = UpSampling2D(name='up6_2')(conv6_3_2)
    # up6_conv_2 = Conv2D(32, 2, padding='same', name='up6_conv_2')(up6_2)
    # conv3_feature_2 = tf.image.resize(conv3_3, (up6_conv_2.shape[1], up6_conv_2.shape[2]),tf.image.ResizeMethod.NEAREST_NEIGHBOR, name='conv3_feature_2')
    #
    # up6_3 = UpSampling2D(name='up6_3')(conv6_3_3)
    # up6_conv_3 = Conv2D(32, 2, padding='same', name='up6_conv_3')(up6_3)
    # conv3_feature_3 = tf.image.resize(conv3_3, (up6_conv_3.shape[1], up6_conv_3.shape[2]),tf.image.ResizeMethod.NEAREST_NEIGHBOR, name='conv3_feature_3')

    concat2_1 = Concatenate(name='concat2_1')([up7_conv_1, conv4_feature_1])
    # concat2_2 = Concatenate(name='concat2_2')([up6_conv_2, conv3_feature_2])
    # concat2_3 = Concatenate(name='concat2_3')([up6_conv_3, conv3_feature_3])

    conv8_1_1 = Conv3D(64, (3,3,3), padding='same', activation='relu', name='conv8_1_1')(concat2_1)
    conv8_2_1 = Conv3D(64, (3,3,3), padding='same', activation='relu', name='conv8_2_1')(conv8_1_1)
    conv8_3_1 = Conv3D(64, (3,3,3), padding='same', activation='relu', name='conv8_3_1')(conv8_2_1)

    # conv7_1_2 = Conv2D(32, 3, padding='same', activation='relu', name='conv7_1_2')(concat2_2)
    # conv7_2_2 = Conv2D(32, 3, padding='same', activation='tanh', name='conv7_2_2')(conv7_1_2)
    # conv7_3_2 = Conv2D(32, 3, padding='same', activation='selu', name='conv7_3_2')(conv7_2_2)
    #
    # conv7_1_3 = Conv2D(32, 3, padding='same', activation='relu', name='conv7_1_3')(concat2_3)
    # conv7_2_3 = Conv2D(32, 3, padding='same', activation='tanh', name='conv7_2_3')(conv7_1_3)
    # conv7_3_3 = Conv2D(32, 3, padding='same', activation='selu', name='conv7_3_3')(conv7_2_3)

    up8_1 = UpSampling3D(name='up8_1')(conv8_3_1)
    up8_conv_1 = Conv3D(32, (2,2,2), padding='same', name='up8_conv_1')(up8_1)
    conv3_feature_1 = Conv3DTranspose(32, (2,2,2), strides=(2,2,2), padding='same', use_bias=False,activation='relu')(conv4_3)

    # up7_2 = UpSampling2D(name='up7_2')(conv7_3_2)
    # up7_conv_2 = Conv2D(16, 2, padding='same', name='up7_conv_2')(up7_2)
    # conv2_feature_2 = tf.image.resize(conv2_3, (up7_conv_2.shape[1], up7_conv_2.shape[2]),tf.image.ResizeMethod.NEAREST_NEIGHBOR, name='conv2_feature_2')
    #
    # up7_3 = UpSampling2D(name='up7_3')(conv7_3_3)
    # up7_conv_3 = Conv2D(16, 2, padding='same', name='up7_conv_3')(up7_3)
    # conv2_feature_3 = tf.image.resize(conv2_3, (up7_conv_3.shape[1], up7_conv_3.shape[2]),tf.image.ResizeMethod.NEAREST_NEIGHBOR, name='conv2_feature_3')

    concat3_1 = Concatenate(name='concat3_1')([up8_conv_1, conv3_feature_1])
    # concat3_2 = Concatenate(name='concat3_2')([up7_conv_2, conv2_feature_2])
    # concat3_3 = Concatenate(name='concat3_3')([up7_conv_3, conv2_feature_3])

    conv9_1_1 = Conv3D(32, (3,3,3), padding='same', activation='relu', name='conv9_1_1')(concat3_1)
    conv9_2_1 = Conv3D(32, (3,3,3), padding='same', activation='relu', name='conv9_2_1')(conv9_1_1)
    conv9_3_1 = Conv3D(32, (3,3,3), padding='same', activation='relu', name='conv9_3_1')(conv9_2_1)

    # conv8_1_2 = Conv2D(16, 3, padding='same', activation='relu', name='conv8_1_2')(concat3_2)
    # conv8_2_2 = Conv2D(16, 3, padding='same', activation='tanh', name='conv8_2_2')(conv8_1_2)
    # conv8_3_2 = Conv2D(16, 3, padding='same', activation='selu', name='conv8_3_2')(conv8_2_2)
    #
    # conv8_1_3 = Conv2D(16, 3, padding='same', activation='relu', name='conv8_1_3')(concat3_3)
    # conv8_2_3 = Conv2D(16, 3, padding='same', activation='tanh', name='conv8_2_3')(conv8_1_3)
    # conv8_3_3 = Conv2D(16, 3, padding='same', activation='selu', name='conv8_3_3')(conv8_2_3)

    up9_1 = UpSampling3D(name='up9_1')(conv9_3_1)
    up9_conv_1 = Conv3D(16, (2,2,2), padding='same', name='up9_conv_1')(up9_1)
    conv2_feature_1 = Conv3DTranspose(16, (2,2,2), strides=(2,2,2), padding='same', use_bias=False,activation='relu')(conv3_3)

    # up8_2 = UpSampling2D(name='up8_2')(conv8_3_2)
    # up8_conv_2 = Conv2D(8, 2, padding='same', name='up8_conv_2')(up8_2)
    # conv1_feature_2 = tf.image.resize(conv1_3, (up8_conv_2.shape[1], up8_conv_2.shape[2]),tf.image.ResizeMethod.NEAREST_NEIGHBOR, name='conv1_feature_2')
    #
    # up8_3 = UpSampling2D(name='up8_3')(conv8_3_3)
    # up8_conv_3 = Conv2D(8, 2, padding='same', name='up8_conv_3')(up8_3)
    # conv1_feature_3 = tf.image.resize(conv1_3, (up8_conv_3.shape[1], up8_conv_3.shape[2]),tf.image.ResizeMethod.NEAREST_NEIGHBOR, name='conv1_feature_3')

    concat4_1 = Concatenate(name='concat4_1')([up9_conv_1, conv2_feature_1])
    # concat4_2 = Concatenate(name='concat4_2')([up8_conv_2, conv1_feature_2])
    # concat4_3 = Concatenate(name='concat4_3')([up8_conv_3, conv1_feature_3])

    conv10_1_1 = Conv3D(16, (3,3,3), padding='same', activation='relu', name='conv10_1_1')(concat4_1)
    conv10_2_1 = Conv3D(16, (3,3,3), padding='same', activation='relu', name='conv10_2_1')(conv10_1_1)
    conv10_3_1 = Conv3D(16, (3,3,3), padding='same', activation='relu', name='conv10_3_1')(conv10_2_1)

    up10_1 = UpSampling3D(name='up10_1')(conv10_3_1)
    up10_conv_1 = Conv3D(8, (2, 2, 2), padding='same', name='up10_conv_1')(up10_1)
    conv1_feature_1 = Conv3DTranspose(8, (2, 2, 2), strides=(2, 2, 2), padding='same', use_bias=False,activation='relu')(conv2_3)

    concat5_1 = Concatenate(name='concat5_1')([up10_conv_1, conv1_feature_1])

    conv11_1_1 = Conv3D(8, (3, 3, 3), padding='same', activation='relu', name='conv11_1_1')(concat5_1)
    conv11_2_1 = Conv3D(8, (3, 3, 3), padding='same', activation='relu', name='conv11_2_1')(conv11_1_1)
    conv11_3_1 = Conv3D(8, (3, 3, 3), padding='same', activation='relu', name='conv11_3_1')(conv11_2_1)

    # conv9_1_2 = Conv3D(8, 3, padding='same', activation='relu', name='conv9_1_2')(concat4_2)
    # conv9_2_2 = Conv3D(8, 3, padding='same', activation='tanh', name='conv9_2_2')(conv9_1_2)
    # conv9_3_2 = Conv3D(8, 3, padding='same', activation='selu', name='conv9_3_2')(conv9_2_2)
    #
    # conv9_1_3 = Conv2D(8, 3, padding='same', activation='relu', name='conv9_1_3')(concat4_3)
    # conv9_2_3 = Conv2D(8, 3, padding='same', activation='tanh', name='conv9_2_3')(conv9_1_3)
    # conv9_3_3 = Conv2D(8, 3, padding='same', activation='selu', name='conv9_3_3')(conv9_2_3)

    out_1 = Conv3D(1, (1,1,1), activation='relu', name='out_1')(conv11_3_1)
    # out_2 = Conv2D(1, 1, activation='tanh', name='out_2')(conv9_3_2)
    # out_3 = Conv2D(1, 1, activation='selu', name='out_3')(conv9_3_3)

    # out = tf.concat([out_1,out_2,out_3],axis=3)
    out = out_1

    model = Model(inputs = inputs,outputs=out, name='mytf2_unet')
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # def loss1(outputs,y_train):
    #     lossu = ((outputs[:, :, :, 0] - y_train[:, :, :, 0]) ** 2)*1.2
    #     lossv = ((outputs[:, :, :, 1] - y_train[:, :, :, 1]) ** 2)
    #     lossp = tf.abs((outputs[:, :, :, 2] - y_train[:, :, :, 2]))
    #     loss = (lossu + lossv + lossp) / 3
    #     return loss
    def acc(y_train,outputs):
        MRE = 1 - tf.abs(y_train[:,:,:,:,0] - outputs[:,:,:,:,0])/(5e-4+y_train[:,:,:,:,0])
        return MRE

    model.compile(
        optimizer,
        metrics=['accuracy',acc],
        loss = 'mse')
    # 设置优化器，损失函数

    return model


def main():
     # model = tf.keras.models.load_model(r'1232_T.h5')
     model = unet()


     def scheduler(epoch):
         # 每隔100个epoch，学习率减小为原来的1/10
         if epoch % 100 == 0 and epoch != 0:
             lr = K.get_value(model.optimizer.lr)
             K.set_value(model.optimizer.lr, lr * 0.5)
             print("lr changed to {}".format(lr * 0.5))
         return K.get_value(model.optimizer.lr)

     reduce_lr = LearningRateScheduler(scheduler)

     model.fit(x_train, y_train, batch_size=16, epochs=1000, callbacks=[reduce_lr])

     model.save('1232_Tlayer6final.h5')
     # out = model.predict(x_test)
     # out = tf.math.log((out/(1-out)))/tf.math.log(2.71828182845905)

     # yp_test = tf.math.log((yp_test / (1 - yp_test))) / tf.math.log(2.71828182845905)
     # test_loss, test_acc = model.evaluate(x_test, ddr)
    # print(out)
    #  print('loss:', test_loss, 'acc:', test_acc)
    #  error = tf.abs(out - ddr)

     #
     # y_test = np.squeeze(ddr[7, :, :, :,0])
     # out = np.squeeze(out[7,:,:,:,0])
     # error = np.squeeze(error[7,:,:,:,0])
     #
     # cut_3D(y_test, out, error)
     # print(np.mean(abs(error/y_test)))
     # for I in range(0, 3):
     #     ux = np.squeeze(resize_3D(out[I, :, :, :, 0], 210, 98, 98))
     #     yp = np.squeeze(resize_3D(ddr[I, :, :, :, 0], 210, 98, 98))
     #
     #     permeability = 1.243 * 210 * 3 / 0.00005 * np.sum(np.abs(ux))/1000000 / 2100000
     #     permeability1 = 1.243 * 210 * 3 / 0.00005 * np.sum(np.abs(yp)) / 1000000 / 2100000
     #     print(permeability)
     #     print(permeability1)
     #     print((permeability1-permeability)/permeability1)




if __name__ == '__main__':

    main()



