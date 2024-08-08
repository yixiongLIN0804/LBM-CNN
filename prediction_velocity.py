
import matplotlib.pyplot as plt
import numpy as np
import xlwt
import pandas as pd
import tensorflow as tf
# 绘图切片函数不过最终效果图最好用MATLAB做合成uyuzux反向的速度场做
def cut_3D(data_3D,II,title):
    import matplotlib.pyplot as plt
    import numpy as np
    # 读取3D数据并且检测类型
    T = data_3D
    if data_3D.ndim != 3:
        print("输入数组不为3维度数组")
    (x, y, z) = T.shape
    x_cut = round(x/2)
    y_cut = round(y/2)
    z_cut = round(z/2)

    T_x = np.squeeze(T[x_cut, :, :])
    T_y = np.squeeze(T[:, y_cut, :])
    T_z = np.squeeze(T[:, :, z_cut])
    zx, zy = np.meshgrid(np.linspace(0, x-1, x), np.linspace(0, y-1, y))
    yx, yz = np.meshgrid(np.linspace(0, x-1, x), np.linspace(0, z-1, z))
    xz, xy = np.meshgrid(np.linspace(0, z-1, z), np.linspace(0, y-1, y))
    fig = plt.figure(II)
    fig.suptitle(title)
    ax1 = fig.add_subplot(221)
    ax1.imshow(np.transpose(T_x), cmap='rainbow', interpolation='nearest', origin='lower', extent=[0, y, 0, z])
    ax1.set(aspect='equal')
    ax1.set_title('x_to')
    ax2 = fig.add_subplot(222)
    ax2.imshow(np.transpose(T_y), cmap='rainbow', interpolation='nearest', origin='lower', extent=[0, x, 0, z])
    ax2.set(aspect='equal')
    ax2.set_title('y_to')
    ax3 = fig.add_subplot(223)
    ax3.imshow(np.transpose(T_z), cmap='rainbow', interpolation='nearest', origin='lower', extent=[0, x, 0, y])
    ax3.set(aspect='equal')
    ax3.set_title('z_to')
    ax = fig.add_subplot(224, projection='3d', proj_type='ortho')
    cset = ax.contourf(T_x, xy, xz,  100, zdir='x', offset=x_cut, cmap='rainbow', alpha=1)
    cset = ax.contourf(yx, np.transpose(T_y), yz, 100, zdir='y', offset=y_cut, cmap='rainbow', alpha=1)
    cset = ax.contourf(zx, zy, np.transpose(T_z), 100, zdir='z', offset=z_cut, cmap='rainbow', alpha=1)
    ax.set_title('3D')
    ax.set_zlim((0, z))
    ax.set_xlim((0, x))
    ax.set_ylim((0, y))
    ax.set(aspect='equal')
    plt.colorbar(cset)
# 对3D数据集合进行缩放
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
# 对3D数据进行缩放
def resize_3D(data_3D,x,y,z):
    import numpy as np
    import tensorflow as tf
    # 读取3D数据并且检测类型
    if data_3D.ndim != 3:
        print("输入数组不为3维度数组")
    data_3D = tf.image.resize(data_3D[:, :, :], [x, y])
    data_3D = tf.image.resize(np.transpose(data_3D, (1, 2, 0)), [y, z])
    data_3D = np.transpose(data_3D, (2, 0, 1))
    return data_3D

x = 256
y = 128
z = 128
data = np.load('LBM_3D120_ux_train.npz')
y_train0 = resize_3D_ass(np.squeeze(data['y']), x, y, z)
print(np.max(y_train0))
# 为ux方向上速度的最大值，归一化使用，最终测试集如果最大速度比这个略大，也没有太多影响
y_max = 6.9584116e-06
print(y_max)
y_test = y_train0/y_max
x_test = resize_3D_ass(np.squeeze(data['x']), x, y, z)
# 集合个数
n_test = 120
print(np.shape(x_test))
print(np.shape(y_test))
LBM1 = np.squeeze(data['y'])/y_max


def loss_me(output, y_train):
    loss = loss = tf.math.reduce_sum(tf.abs(output - y_train)) / tf.math.reduce_sum(tf.abs(y_train))
    return loss
model_name = 'ux_model.h5'
model = tf.keras.models.load_model(model_name, compile=False)

model.compile(optimizer=tf.optimizers.Adam(),
              loss=loss_me,
              metrics=['accuracy'])
# test_loss1, test_acc = model.evaluate(x_test, y_test, batch_size=1)
test_loss =[]
predictions = []

# 使用模型预测速度场
# 数据比较大拆开来一组一组跑，一起跑，不然显卡太差容易炸
for I in range(0, n_test):
    test_loss1, test_acc = model.evaluate([x_test[[I], :, :, :]], y_test[[I], :, :, :])
    test_loss = test_loss + [test_loss1]
    predictions1 = model.predict(x_test[[I], :, :, :])
    predictions = predictions + [predictions1]



print('loss 平均')
print(np.mean(test_loss))
predictions = np.squeeze(predictions)



print(np.shape(y_test))
print(np.shape(predictions))
err1 = []
permeability_cnn = []
permeability_lbm = []
err2 = []
CNN1 = []
for _ in range(0, n_test):
    cnn = np.squeeze(predictions[_, :, :, :])
    lbm = np.squeeze(y_test[_, :, :, :])

    tj = np.squeeze(np.reshape(abs(cnn[lbm != 0] - lbm[lbm != 0])/lbm[lbm != 0], (1, len(lbm[lbm != 0]))))

    print('最大点绝对误差')
    print(np.max(tj))
    print('UX平均误差')
    print(abs((np.sum(abs(cnn-lbm))))/np.sum(lbm))
    ux_lbm = resize_3D(lbm, 210, 100, 100)
    ux_cnn = resize_3D(cnn, 210, 100, 100)

    # 是否打印每次的数据3D截面图数据
    # 不过效果不好，最好是用MATLAB画
    # cut_3D(ux_lbm, 1, 'LBM')
    # cut_3D(ux_cnn, 2, 'CNN')
    # cut_3D(np.abs(ux_lbm-ux_cnn), 3, 'ERR')
    # plt.show()


    CNN1 = CNN1+[ux_cnn]
    # 这个是渗透率判断 如果是uy uz 反向的速度场就可以不要看这些了
    permeability = 1.243 * 210 * 3 / 0.00005 * np.sum(ux_lbm) * y_max / 2100000
    permeability1 = 1.243 * 210 * 3 / 0.00005 * np.sum(ux_cnn) * y_max / 2100000
    print('LBM渗透率')
    print(permeability)
    print('CNN渗透率')
    print(permeability1)
    print('渗透率绝对误差')
    print(permeability1-permeability)
    print('渗透率相对误差')
    print(abs((permeability1 - permeability)/permeability))
    print('渗透率绝对相对误差')
    print(abs((np.sum(abs(cnn-lbm))))/np.sum(lbm))
    err1 = err1 + [abs((permeability1 - permeability)/permeability)]
    err2 = err2 + [abs((np.sum(abs(cnn-lbm))))/np.sum(lbm)]
    permeability_cnn = permeability_cnn + [permeability1]
    permeability_lbm = permeability_lbm + [permeability]
    sections = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    cuts = pd.cut(tj, sections)


print('平均渗透率误差')
x = range(1, n_test+1)
print(np.mean(err1))
print('平均误差')
print(np.mean(err2))
plt.figure(1)
plt.plot(x, err1, color='red', marker="o")
plt.xlabel('number')
plt.ylabel('relative error of permeability')
plt.figure(2)
plt.plot(x, permeability_cnn, color='red', marker="o", label='cnn', markerfacecolor='none')
plt.scatter(x, permeability_lbm, color='blue', marker="s", label='lbm', facecolor='none')
plt.legend()
plt.xlabel('number')
plt.ylabel('permeability')
plt.show()
book = xlwt.Workbook()
al = xlwt.Alignment()
style = xlwt.XFStyle()
al.horz = 0x02  # 设置水平居中
al.vert = 0x01  # 设置垂直居中
style.alignment = al
sheet = book.add_sheet(model_name)
# 保存LBM和CNN的渗透率数据
for I in range(0, n_test):
    sheet.write(I, 0, '{:.4f}'.format(permeability_cnn[I]), style)
    sheet.write(I, 1, '{:.4f}'.format(permeability_lbm[I]), style)
    sheet.write(I, 2, '{:.4f}'.format(err1[I]), style)
    sheet.write(I, 3, '{:.4f}'.format(err2[I]), style)
book.save(model_name+'.xls')



