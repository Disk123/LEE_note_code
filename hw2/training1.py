import numpy as np
import matplotlib.pyplot as plt

####  数据的预处理阶段
np.random.seed(0)
X_train=np.load("train_x.npy")
Y_train=np.load("train_y.npy")
X_dev=np.load("dev_x.npy")
Y_dev = np.load("dev_y.npy")
X_test = np.load("test_x.npy")


# #  训练数据的数量
# train_size = X_train.shape[0]
# #   验证数据的数量
# dev_size = X_dev.shape[0]
# #   测试数据的数量
# test_size = X_test.shape[0]
# #    输入参数的维度
# data_dim = X_train.shape[1]



# 线性回归
class LogictRegression:

    def __init__(self):
        # 初始化 Linear Regression 模型
        self.coef_ = None
        self.intercept_ = None
        self._theta = None
    #  这个函数的作用是将训练数据的排序进行打乱，降低数据样本之间的相关性
    def shuffle(self,X, Y):
        # This function shuffles two equal-length list/array, X and Y, together.
        randomize = np.arange(len(X))
        #  生成随机序列
        np.random.shuffle(randomize)
        #   将训练数据重新的排序一下，这可能有助于打破数据的相关性
        return (X[randomize], Y[randomize])

    #   定义sigmoid函数
    def sigmoid(self,z):
        #  改函数计算了样本属于那一类的概率
        #  np.clip 是用于限制概率值在一个比较合理的范围，即在[1e-8,1-(le-8)]范围内
        return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))


    def f(self,X,w,b):
        # This is the logistic regression function, parameterized by w and b
        #
        # Arguements:
        #     X: input data, shape = [batch_size, data_dimension]
        #     w: weight vector, shape = [data_dimension, ]
        #     b: bias, scalar
        # Output:
        #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
        #  这个函数主要用于处理，在X,w，b进行了相应的运算之后，需要把运算之后的结果放到sigmoid函数中进行非线性激活
        #  输出作为下一个神经元的输入或者作为预测值输出
        aa = np.matmul(X, w)
        return self.sigmoid(np.matmul(X, w) + b)

    #  定义预测函数，采用np.round的方法判断这训练样本属于那一类
    def predict(self,X, w, b):
        # This function returns a truth value prediction for each row of X
        # by rounding the result of logistic regression function.

        #  这个函数主要是用于判断预测值是否大于0.5,如果大于0.5则输出1,否则输出0
        return np.round(self.f(X, w, b)).astype(np.int)

    #  计算模型预测的正确率
    def accuracy(self,Y_pred, Y_label):
        # This function calculates prediction accuracy
        #  np.abs(Y_pred - Y_label)用于计算有多少是预测正确的，如果预测正确的则返回0,否则返回1或-1,经过绝对值之后都变成1
        #  np.mean  计算正确率的平均值
        acc = 1 - np.mean(np.abs(Y_pred - Y_label))
        return acc


    #  逻辑回归的损失函数计算方法
    def cross_entropy_loss(self,y_pred, Y_label):

        cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
        return cross_entropy

    #  计算权重的导数
    def gradient(self,X, Y_label,w,b):
        # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
        y_pred = self.f(X,w,b)
        pred_error = Y_label - y_pred
        w_grad = -np.sum(pred_error * X.T, 1)           #  w的求导公式
        b_grad = -np.sum(pred_error)                    #  b的求导公式
        return w_grad, b_grad

    def train(self,X_train,Y_train,X_dev,Y_dev,max_iter=10,batch_size=8,learning_rate=1e-2):
        '''
                :param X_train: 训练数据特征向量
                :param Y_train: 训练数据的lebel
                :param X_dev:   验证数据特征向量
                :param Y_dev:   验证数据的lebel
                :param max_iter: 训练回合数
                :param batch_size: 每一次训练的样本数量
                :return:learning_rate：学习率
                '''

        #  用于判断训练数据样本个数是否与标签数据个数是否相等
        assert X_train.shape[0] == Y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        self.train_loss = []
        self.dev_loss = []
        self.train_acc = []
        self.dev_acc = []

        data_dim = X_train.shape[1]
        w=np.zeros((data_dim,))
        b=np.zeros((1,))
        w_grad_ada=np.zeros(data_dim)
        b_grad_ada=np.zeros(1)
        train_size = X_train.shape[0]
        #   验证数据的数量
        dev_size = X_dev.shape[0]

        # Iterative training
        step = 1
        for epoch in range(max_iter):
            # Random shuffle at the begging of each epoch
            #  首先把训练数据进行打乱，避免数据间的相关性
            X_train, Y_train = self.shuffle(X_train, Y_train)

            # Mini-batch training
            for idx in range(int(np.floor(train_size / batch_size))):
                #  截取batch_size的小数据量进行训练
                X = X_train[idx * batch_size:(idx + 1) * batch_size]
                Y = Y_train[idx * batch_size:(idx + 1) * batch_size]

                # Compute the gradient
                # 计算损失权重参数的导数
                w_grad, b_grad = self.gradient(X, Y,w,b)

                w = w - learning_rate / np.sqrt(step) * w_grad
                b = b - learning_rate / np.sqrt(step) * b_grad

                #  添加了ADA方法
                # w_grad_ada += w_grad ** 2 + 1e-8
                # b_grad_ada += b_grad ** 2 + 1e-8
                # w_ada = np.sqrt(w_grad_ada)
                # b_ada = np.sqrt(b_grad_ada)
                #
                # # gradient descent update
                # # learning rate decay with time
                # #  更新参数
                # w = w - learning_rate * w_grad / w_ada
                # b = b - learning_rate  * b_grad / b_ada



                step = step + 1
                print(step)


            # Compute loss and accuracy of training set and development set
            #  经过一轮训练之后，得到相应的权重参数，利用这一次权重参数进行预测
            #  计算没一个眼本数据属于那一类的概率值大小
            y_train_pred = self.f(X_train, w , b)
            Y_train_pred = np.round(y_train_pred)  # 利用四舍五入的方法进行归类
            #   计算准确性
            self.train_acc.append(self.accuracy(Y_train_pred, Y_train))
            #   计算这一次的loss
            self.train_loss.append(self.cross_entropy_loss(y_train_pred, Y_train) / train_size)

            #  验证数据的测试
            y_dev_pred = self.f(X_dev, w, b)
            Y_dev_pred = np.round(y_dev_pred)
            self.dev_acc.append(self.accuracy(Y_dev_pred, Y_dev))
            self.dev_loss.append(self.cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)
        self.coef_=b
        self.intercept_=w

    def plot_loss(self):
        # Loss curve
        plt.plot(self.train_loss)
        plt.plot(self.dev_loss)
        plt.title('Loss')
        plt.legend(['train', 'dev'])
        # plt.savefig('loss.png')
        plt.show()

            # Accuracy curve
        plt.plot(self.train_acc)
        plt.plot(self.dev_acc)
        plt.title('Accuracy')
        plt.legend(['train', 'dev'])
        # plt.savefig('acc.png')
        plt.show()


# Some parameters for training
max_iter = 10           #  训练回合
batch_size = 8          #  采用mini_batch 的方法训练莫新，batch_size=8
learning_rate = 0.01     #  学习效率

LR = LogictRegression()
LR.train(X_train, Y_train,X_dev,Y_dev,max_iter,batch_size,learning_rate)
LR.plot_loss()

print('Training loss: {}'.format(LR.train_loss[-1]))
print('Development loss: {}'.format(LR.dev_loss[-1]))
print('Training accuracy: {}'.format(LR.train_acc[-1]))
print('Development accuracy: {}'.format(LR.dev_acc[-1]))

predictions = LR.predict(X_test, LR.intercept_, LR.coef_)

print('断点')