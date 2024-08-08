import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv3D, UpSampling3D,
                                     Conv3DTranspose, Concatenate, MaxPooling3D,
                                     )
import keras.backend as K
from keras.models import Model
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
# 判断显卡和CUDA是否可用 服务器上训练模型一定要这一步
# print(tf.test.is_built_with_cuda())
# print(tf.test.is_gpu_available())
import os
# 消除警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
# 可以选择禁用CUDA用cpu跑，不过这样巨慢无比，考虑清楚啊
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def resize_3D_ass(data_3D,x,y,z):
    import numpy as np
    import tensorflow as tf
    # 读取3D数据并且检测类型

    if data_3D.ndim != 4:
        print("输入数组不为3维度数组合集")
    (n, a, b,  c) = data_3D.shape
    data_3D2 = []
    for I in range(0, n):
        data_3D1 = data_3D[I, :, :, :]
        data_3D1 = tf.image.resize(data_3D1[:, :, :], [x, y])
        data_3D1 = tf.image.resize(np.transpose(data_3D1, (1, 2, 0)), [y, z])
        data_3D1 = np.transpose(data_3D1, (2, 0, 1))
        data_3D2 = data_3D2 + [data_3D1]
    data_3D = np.array(data_3D2, dtype=np.float32)
    return data_3D

# 五层UNET神经网络
# uy uz 速度场要替换最后的输出函数为tanh
def unet_5(x,y,z):
    """ set up unet model """

    inputs = Input(shape=(x, y, z, 1), name='input')
    conv1_1 = Conv3D(8, (3, 3, 3), padding='same', activation='tanh', name='conv1_1')(inputs)
    conv1_2 = Conv3D(8, (3, 3, 3), padding='same', activation='tanh', name='conv1_2')(conv1_1)
    conv1_3 = Conv3D(8, (3, 3, 3), padding='same', activation='tanh', name='conv1_3')(conv1_2)

    max_pool1 = MaxPooling3D(name='max_pool1')(conv1_3)  # maxpooling2D默认stride=filer size
    conv2_1 = Conv3D(16, (3, 3, 3), padding='same', activation='tanh', name='conv2_1')(max_pool1)
    conv2_2 = Conv3D(16, (3, 3, 3), padding='same', activation='tanh', name='conv2_2')(conv2_1)
    conv2_3 = Conv3D(16, (3, 3, 3), padding='same', activation='tanh', name='conv2_3')(conv2_2)

    max_pool2 = MaxPooling3D(name='max_pool2')(conv2_3)
    conv3_1 = Conv3D(32, (3, 3, 3), padding='same', activation='tanh', name='conv3_1')(max_pool2)
    conv3_2 = Conv3D(32, (3, 3, 3), padding='same', activation='tanh', name='conv3_2')(conv3_1)
    conv3_3 = Conv3D(32, (3, 3, 3), padding='same', activation='tanh', name='conv3_3')(conv3_2)

    max_pool3 = MaxPooling3D(name='max_pool3')(conv3_3)
    conv4_1 = Conv3D(64, (3, 3, 3), padding='same', activation='tanh', name='conv4_1')(max_pool3)
    conv4_2 = Conv3D(64, (3, 3, 3), padding='same', activation='tanh', name='conv4_2')(conv4_1)
    conv4_3 = Conv3D(64, (3, 3, 3), padding='same', activation='tanh', name='conv4_3')(conv4_2)

    max_pool4 = MaxPooling3D(name='max_pool4')(conv4_3)
    conv5_1 = Conv3D(128, (3, 3, 3), padding='same', activation='tanh', name='conv5_1')(max_pool4)
    conv5_2 = Conv3D(128, (3, 3, 3), padding='same', activation='tanh', name='conv5_2')(conv5_1)
    conv5_3 = Conv3D(128, (3, 3, 3), padding='same', activation='tanh', name='conv5_3')(conv5_2)

    max_pool5 = MaxPooling3D(name='max_pool5')(conv5_3)
    conv6_1 = Conv3D(256, (3, 3, 3), padding='same', activation='tanh', name='conv6_1')(max_pool5)
    conv6_2 = Conv3D(256, (3, 3, 3), padding='same', activation='tanh', name='conv6_2')(conv6_1)
    conv6_3 = Conv3D(256, (3, 3, 3), padding='same', activation='tanh', name='conv6_3')(conv6_2)

    up6_1 = UpSampling3D(name='up6_1')(conv6_3)
    up6_conv_1 = Conv3D(128, (2, 2, 2), padding='same', name='up6_conv_1')(up6_1)
    conv5_feature_1 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same', use_bias=False,activation='tanh')(conv6_3)
    concat1_1 = Concatenate(name='concat1_1')([up6_conv_1, conv5_feature_1])


    conv7_1_1 = Conv3D(128, (3, 3, 3), padding='same', activation='tanh', name='conv7_1_1')(concat1_1)
    conv7_2_1 = Conv3D(128, (3, 3, 3), padding='same', activation='tanh', name='conv7_2_1')(conv7_1_1)
    conv7_3_1 = Conv3D(128, (3, 3, 3), padding='same', activation='tanh', name='conv7_3_1')(conv7_2_1)

    up7_1 = UpSampling3D(name='up7_1')(conv7_3_1)
    up7_conv_1 = Conv3D(64, (2, 2, 2), padding='same', name='up7_conv_1')(up7_1)
    conv4_feature_1 = Conv3DTranspose(64, (2, 2, 2), strides=(2,2,2), padding='same', use_bias=False,activation='tanh')(conv5_3)

    concat2_1 = Concatenate(name='concat2_1')([up7_conv_1, conv4_feature_1])

    conv8_1_1 = Conv3D(64, (3,3,3), padding='same', activation='tanh', name='conv8_1_1')(concat2_1)
    conv8_2_1 = Conv3D(64, (3,3,3), padding='same', activation='tanh', name='conv8_2_1')(conv8_1_1)
    conv8_3_1 = Conv3D(64, (3,3,3), padding='same', activation='tanh', name='conv8_3_1')(conv8_2_1)

    up8_1 = UpSampling3D(name='up8_1')(conv8_3_1)
    up8_conv_1 = Conv3D(32, (2, 2, 2), padding='same', name='up8_conv_1')(up8_1)
    conv3_feature_1 = Conv3DTranspose(32, (2, 2, 2), strides=(2,2,2), padding='same', use_bias=False,activation='tanh')(conv4_3)


    concat3_1 = Concatenate(name='concat3_1')([up8_conv_1, conv3_feature_1])


    conv9_1_1 = Conv3D(32, (3, 3, 3), padding='same', activation='tanh', name='conv9_1_1')(concat3_1)
    conv9_2_1 = Conv3D(32, (3, 3, 3), padding='same', activation='tanh', name='conv9_2_1')(conv9_1_1)
    conv9_3_1 = Conv3D(32, (3, 3, 3), padding='same', activation='tanh', name='conv9_3_1')(conv9_2_1)

    up9_1 = UpSampling3D(name='up9_1')(conv9_3_1)
    up9_conv_1 = Conv3D(16, (2, 2, 2), padding='same', name='up9_conv_1')(up9_1)
    conv2_feature_1 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same', use_bias=False,activation='tanh')(conv3_3)


    concat4_1 = Concatenate(name='concat4_1')([up9_conv_1, conv2_feature_1])
    conv10_1_1 = Conv3D(16, (3, 3, 3), padding='same', activation='tanh', name='conv10_1_1')(concat4_1)
    conv10_2_1 = Conv3D(16, (3, 3, 3), padding='same', activation='tanh', name='conv10_2_1')(conv10_1_1)
    conv10_3_1 = Conv3D(16, (3, 3, 3), padding='same', activation='tanh', name='conv10_3_1')(conv10_2_1)

    up10_1 = UpSampling3D(name='up10_1')(conv10_3_1)
    up10_conv_1 = Conv3D(8, (2, 2, 2), padding='same', name='up10_conv_1')(up10_1)
    conv1_feature_1 = Conv3DTranspose(8, (2, 2, 2), strides=(2, 2, 2), padding='same', use_bias=False, activation='tanh')(conv2_3)

    concat5_1 = Concatenate(name='concat5_1')([up10_conv_1, conv1_feature_1])

    conv11_1_1 = Conv3D(8, (3, 3, 3), padding='same', activation='tanh', name='conv11_1_1')(concat5_1)
    conv11_2_1 = Conv3D(8, (3, 3, 3), padding='same', activation='tanh', name='conv11_2_1')(conv11_1_1)
    conv11_3_1 = Conv3D(8, (3, 3, 3), padding='same', activation='tanh', name='conv11_3_1')(conv11_2_1)

    out_1 = Conv3D(1, (1, 1, 1), activation='tanh', name='out_1')(conv11_3_1)
    out = out_1

    model = Model(inputs=inputs, outputs=out, name='tf2_unet')
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    def loss_me(output, y_train):
        # 前几次loss大的时侯，采用这种激loss函数更合适
        loss = tf.math.reduce_sum(tf.abs(output - y_train))/x/y/z
        if loss < 0.6:
            loss = tf.math.reduce_sum(tf.abs(output - y_train)) / tf.math.reduce_sum(tf.abs(y_train))
        return loss
    model.compile(
        optimizer,
        metrics=['accuracy'],
        loss=loss_me)
    # 设置优化器，损失函数

    return model

# resize 架构 缩放运行 适合个人电脑回归
# 如果想要用256×128×128的数据最好要用A系列以上的显卡
# 或者用cpu慢慢跑也行
x = 128
y = 64
z = 64
data = np.load('LBM_3D120_ux_train.npz')
y_train0 = resize_3D_ass(np.squeeze(data['y']), x, y, z)
# 寻找最大值归一化速度场模型
y_max = np.max(abs(y_train0))
# 这个要记得模型使用也要用
print(y_max)
y_train = y_train0/y_max
x_train = resize_3D_ass(np.squeeze(data['x']), x, y, z)
print(np.shape(y_train))
print(np.shape(x_train))
data = np.load('LBM_3D32_ux_test.npz')
y_test0 = resize_3D_ass(np.squeeze(data['y']), x, y, z)
y_test = y_test0/y_max
x_test = resize_3D_ass(np.squeeze(data['x']), x, y, z)
print(np.shape(y_test))
print(np.shape(x_test))


#
def scheduler_1(epoch):
    # 开始训练侯，学习率减小为原来的1/2
    if epoch == 1:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
        print("lr changed to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)
reduce_lr_1 = LearningRateScheduler(scheduler_1)
model = unet_5(x, y, z)
# 设置改变学习率的点
change_N = [5, 10, 15]
# 初始
dI = 20
NI_max = 45
# dI 表示间隔数
# NI_max表示取点数
# 总迭代次数为NI_max*dI
test_loss_all = []
train_loss_all = []

name_model = 'ux_test_model.h5'
model.save(name_model)
# 前I_first次没有判断是否过拟合的价值 所以可以先迭代
I_first = 100
model.fit(x_train, y_train, batch_size=4, epochs=I_first)
test_loss1 = 1
test_loss2 = 2
def loss_me(output, y_train):
    loss = tf.math.reduce_sum(tf.abs(output - y_train)) / x / y / z
    if loss < 0.6:
        loss = tf.math.reduce_sum(tf.abs(output - y_train)) / tf.math.reduce_sum(tf.abs(y_train))
    return loss

for I_1 in range(1, NI_max+1):

    print('The epoch times is')
    print((I_1-1)*dI+I_first)
    # 每次跑完后保存一下子，避免跑崩了重新跑
    model.save(name_model)
    model.reset_states()
    model = tf.keras.models.load_model(name_model, custom_objects={'loss_me': loss_me})


    if I_1 in change_N:
        model.fit(x_train, y_train, batch_size=4, epochs=dI, callbacks=[reduce_lr_1])
    else:
        model.fit(x_train, y_train, batch_size=4, epochs=dI)

    test_loss = []
    train_loss = []
    test_loss2 = test_loss1
    test_loss1, test_acc = model.evaluate(x_test, y_test, batch_size=4)
    train_loss1, train_acc = model.evaluate(x_train, y_train, batch_size=4)
    test_loss = test_loss + [test_loss1]
    train_loss = train_loss + [train_loss1]
    # 数据比较大拆开来一组一组跑 如果同时评估100组数据不知道A100能不能顶得住 顶不住用下面的代码
    # for I in range(0, n_test):
    #     test_loss1, test_acc = model.evaluate([x_test[[I], :, :, :]], y_test[[I], :, :, :])
    #     test_loss = test_loss + [test_loss1]
    # for I in range(0, n_train):
    #     train_loss1, train_acc = model.evaluate([x_train[[I], :, :, :]], y_train[[I], :, :, :])
    #     train_loss = train_loss + [train_loss1]
    test_loss_all = test_loss_all + [np.mean(test_loss)]
    train_loss_all = train_loss_all + [np.mean(train_loss)]
    # 打印测试集和训练集loss随迭代次数变化图
    if I_1 > 3:
        n_x = np.arange(1, I_1+1)

        plt.plot(n_x, test_loss_all, color='red', marker="o", label='test', markerfacecolor='none')
        plt.plot(n_x, train_loss_all, color='blue', marker="o", label='train', markerfacecolor='none')
        plt.legend()
        plt.xlabel('number')
        plt.ylabel('loss')
        plt.show()
# 记录每次迭代数据
np.savez('loss_ux.npz', test_loss_all=test_loss_all, train_loss_all=train_loss_all)
