# coding:utf-8
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import xlrd
from sklearn import metrics

from convext import R2AttU_ConvNext

data = xlrd.open_workbook('1.xlsx')

testlabel = data.sheet_by_name(u'testl')
test_label = []
for i in range(testlabel.nrows):
    test_label.append(testlabel.row_values(i))
test_label = np.array(test_label)
print(len(test_label))

trainlabel = data.sheet_by_name(u'trainl')
train_label = []
for i in range(trainlabel.nrows):
    train_label.append(trainlabel.row_values(i))
train_label = np.array(train_label)
print(len(train_label))

test_ = data.sheet_by_name(u'test')
test = []
for i in range(test_.nrows):
    test.append(test_.row_values(i))
test = np.array(test)
print(len(test))

train_ = data.sheet_by_name(u'train')
traind = []
for i in range(train_.nrows):
    traind.append(train_.row_values(i))
traind = np.array(traind)
print(len(traind))

# 交互式session方式
sess = tf.InteractiveSession()

'''
def weight_variable_L2(shape):
    var = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    weight_loss = tf.multiply(tf.nn.l2_loss(var), 1, name='weight_loss')
    tf.add_to_collection('losses', weight_loss)
    return var
'''


def conv2d(x, W):
    # same表示零填充 x填充的数据   w代表滤波器
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


'''    
#Leaky ReLU activation functions 
def LeakyRelu(x, leak=5.5, name="LeakyRelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * tf.abs(x)
'''


# 激活函数
# Parametric Rectified Linear Unit activation functions
def PReLU(_x, scope=None):
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.001))
    return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


x = tf.placeholder(tf.float32, [None, 576])
y_ = tf.placeholder(tf.float32, [None, 2])

# 重塑张量 四维数组
# 源码：x_image = tf.reshape(x, [-1, 28, 28, 1])
# 这里是将一组图像矩阵x重建为新的矩阵，该新矩阵的维数为（a，28，28，1），其中-1表示a由实际情况来定。例如，x是一组图像的矩阵（假设是50张，大小为56×56），则执行
# x_image = tf.reshape(x, [-1, 28, 28, 1])
# 可以计算a=50×56×56/28/28/1=200。即x_image的维数为（200，28，28，1）。
# ————————————————
# 版权声明：本文为CSDN博主「冰雪棋书」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/zml194849/article/details/113351646
x_image = tf.reshape(x, [-1, 24, 24, 1])

# 这部分被精简了
h_fc1 = R2AttU_ConvNext(x_image)

# 设置dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 这是一个softmax层，全连接层
W_fc2 = tf.get_variable(name='W_fc2', shape=[64, 2], initializer=tf.contrib.layers.xavier_initializer_conv2d())
b_fc2 = tf.get_variable(name='b_fc2', shape=[2], initializer=tf.constant_initializer(value=0.01, dtype=tf.float32))
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 定义一个损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# cross_entropy=tf.reduce_mean(tf.square(y_conv - y_))
# 定义优化器
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

loss_stochastic = []

# 定义评测标准的准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 开始训练模型
tf.global_variables_initializer().run()
batch_size = 200
for i in range(500):
    rand_index = np.random.choice(9177, size=batch_size)
    rand_x = traind[rand_index]
    rand_y = train_label[rand_index]
    # print('x是' + str(rand_x) + '----' )
    # print('y是' + str(rand_y))y67
    train_step.run(feed_dict={x: rand_x, y_: rand_y, keep_prob: 0.5})
    if i % 50 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: rand_x, y_: rand_y, keep_prob: 1.0})
        print("step %d, train accuracy:%g" % (i, train_accuracy))
        temp_loss = sess.run(cross_entropy, feed_dict={x: rand_x, y_: rand_y, keep_prob: 0.5})
        loss_stochastic.append(temp_loss)
        # add
        saver = tf.train.Saver()
        saver.save(sess, "./modelvgg0609/cnn.ckpt")

plt.plot(loss_stochastic, 'b-', label='Stochastic Loss')
plt.show()

y_p = tf.argmax(y_conv, 1)
test_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: test, y_: test_label, keep_prob: 1.0})
print("test accuracy: ", test_accuracy)
y_true = np.argmax(test_label, 1)

# true positive
TP = np.sum(np.multiply(y_true, y_pred))
print("True positive:", TP)

# false positive
FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))
print("False positive:", FP)

# false negative
FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))
print("False negative:", FN)

# true negative
TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))
print("True negative:", TN)

# summarize the fit of the model
print("Result:\n", metrics.classification_report(y_true, y_pred))
# confusion matrix, rows are the true values and the columns are the predicted values
print("Confusion_matrix:\n", metrics.confusion_matrix(y_true, y_pred))

# Precision(TP/(TP+FP))
print("Precision:", metrics.precision_score(y_true, y_pred))

# Recall(TP/(TP+FN))=Sensitivity(TP/(TP+FN))
print("Recall:", metrics.recall_score(y_true, y_pred))

# F1=(2*Precision*Recall)/(Precision+Recall)
print("F1_score:", metrics.f1_score(y_true, y_pred))

fpr, tpr, tresholds = metrics.roc_curve(y_true, y_pred)
# Sensitivity(TP/(TP+FN))=Recall(TP/(TP+FN))
print("Sensitivity:", tpr)
print("Sensitivity敏感度:",TP/(TP + FP))
# Specificity(TN/(TN+FN))
print("Specificity:", 1 - fpr)
print("Specificity特异度:",TN/(FN + TN))
# Matthew's correlation coefficient
print("MCC:", metrics.matthews_corrcoef(y_true, y_pred))
# AUC
roc_auc = metrics.auc(fpr, tpr)
print("AUC:", roc_auc)
# print("AUROC:\n",metrics.roc_auc_score(y_true,y_pred))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

