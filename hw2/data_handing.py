import numpy as np


####  数据的预处理阶段
np.random.seed(0)
X_train_fpath = './data/X_train'
Y_train_fpath = './data/Y_train'
X_test_fpath = './data/X_test'
# output_fpath = './output_{}.csv'

# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)


def _normalize(X, train=True, specified_column=None, X_mean=None, X_std=None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data

    #  对输入的参数进行归一化
    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        # .reshape是把数组变成一行
        #  np.mean(x,0)对每一列求平均
        #  np.std(x,0)对每一列求方差
        X_mean = np.mean(X[:, specified_column], 0).reshape(1, -1)
        X_std = np.std(X[:, specified_column], 0).reshape(1, -1)
    #  得到方差和均值之后，对输入参数进行归一化处理
    X[:, specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)

    return X, X_mean, X_std


#   该函数将训练数据分为训练数据和验证数据
def _train_dev_split(X, Y, dev_ratio=0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    # bb = X[:train_size]
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]


# Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train=True)
X_test, _, _ = _normalize(X_test, train=False, specified_column=None, X_mean=X_mean, X_std=X_std)

# Split data into training set and development set
dev_ratio = 0.1             #  切分训练集和验证集大小的权重
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio=dev_ratio)
# print(X_train.shape)

#  存储处理好的数据
np.save("train_x.npy",X_train)
np.save("train_y.npy",Y_train)
np.save("dev_x",X_dev)
np.save("dev_y.npy",Y_dev)
np.save("test_x",X_test)
print('断点')

# #  训练数据的数量
# train_size = X_train.shape[0]
# #   验证数据的数量
# dev_size = X_dev.shape[0]
# #   测试数据的数量
# test_size = X_test.shape[0]
# #    输入参数的维度
# data_dim = X_train.shape[1]