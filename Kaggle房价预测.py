import torch
import myutils
import torch.nn as nn
import pandas as pd

print(torch.__version__)   #查看pytorch的版本
torch.set_default_tensor_type(torch.FloatTensor)  # 设置pytorch中默认的浮点类型为32bit浮点


#获取和读取数据集
train_data = pd.read_csv(r'C:\Users\林剑艺\Desktop\Kaggle房价预测\data\train.csv')
test_data = pd.read_csv(r'C:\Users\林剑艺\Desktop\Kaggle房价预测\data\test.csv')
print(train_data.shape) # 输出 (1460, 81)，训练数据集包括1460个样本、80个特征和1个标签
print(test_data.shape) # 输出 (1459, 80)，测试数据集包括1459个样本、80个特征
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]) # 查看前4个样本的前4个特征、后2个特征和标签（SalePrice）
# 第一个特征是Id，它能帮助模型记住每个训练样本，但难以推广到测试样本，所以不使用它来训练。我们将所有的训练数据和测试数据的79个特征按样本连结。
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
print(all_features.shape)  # 输出 (2919, 79)


#预处理数据
'''
对连续数值的特征做标准化（standardization）：设该特征在整个数据集上的均值为μ，标准差为σ。
那么，我们可以将该特征的每个值先减去μ再除以σ得到标准化后的每个特征值；对于缺失的特征值，我们将其替换成该特征的均值
'''
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index  #找出所有非字符串，即数值类的特征列，共36个(非object类）存储的是特征，如MSSubClass
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 标准化后，每个数值特征的均值变为0，所以可以直接用0来替换缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0) # fillna()会填充nan数据，并返回填充后的结果

# 接下来将离散数值转成指示特征
'''
举个例子，假设特征MSZoning里面有两个不同的离散值RL和RM，那么这一步转换将去掉MSZoning特征，并新加两个特征MSZoning_RL和MSZoning_RM，其值为0或1
如果一个样本原来在MSZoning里的值为RL，那么有MSZoning_RL=1且MSZoning_RM=0。
'''
# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape) # (2919, 331) ，将特征数从79增加到了331
print(all_features.iloc[0:2, -4:-1]) # 可以看到SaleCondition被分成了SaleCondition_Family  SaleCondition_Normal  SaleCondition_Partial等

#通过values属性得到NumPy格式的数据，并转成Tensor方便后面的训练
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1) # 取出SalePrice的数据，并转化成1460×1的向量


# 训练模型
loss = torch.nn.MSELoss()
# 线性回归模型和平方损失函数
def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net

#对数均方根误差
def log_rmse(net, features, labels):
    with torch.no_grad(): # 不记录梯度
        # 将小于1的值设成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log())) # pytorch里的MSELoss并没有除以 2
    return rmse.item()

#使用了Adam优化算法
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    # 这里使用了Adam优化算法
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# K折交叉验证
def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1   # k>1才执行的下去
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    # 这个循环实现了取出一折用来做验证集，剩下的部分全部做训练集
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # 返回一个切片对象，用来索引
        X_part, y_part = X[idx, :], y[idx]
        if j == i: # j == i 保证了每次取出来的验证集都是不一样的
            X_valid, y_valid = X_part, y_part  # valid 验证集
        elif X_train is None:
            X_train, y_train = X_part, y_part  # 先给训练集做个开头
        else:   # 拼接剩下所有的训练集
            X_train = torch.cat((X_train, X_part), dim=0) # 在给定维度上对输入的张量序列seq 进行连接操作，dim=0表示上下连接
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid

# 在K折交叉验证中训练K次并返回训练和验证的平均误差
'''
每一次，我们使用一个子数据集验证模型，并使用其他K−1个子数据集来训练模型。
在这K次训练和验证中，每次用来验证模型的子数据集都不同。最后，我们对这K次训练误差和验证误差分别求平均。
'''
def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train) # 返回的是元组类型
        net = get_net(X_train.shape[1]) # X_train.shape[1]返回特征的个数
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)  # *data可选参数，其实有四个变量
        train_l_sum += train_ls[-1]  # 返回最后一次训练周期的loss
        valid_l_sum += valid_ls[-1]
        if i == 0:
            myutils.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',   # rmse均方根误差
                         range(1, num_epochs + 1), valid_ls,
                         ['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1])) # %d 有符号的十进制整数; %f 浮点实数
    return train_l_sum / k, valid_l_sum / k

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))  # validation验证


# 预测并在Kaggle提交结果
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net(train_features.shape[1])
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    myutils.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).detach().numpy()   # detach()不被继续追踪,preds的shape是（1459，1）
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0]) # Series是一维数组；转成（1，1459）是[[]]样式，所以需取[0]
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)  # axis=1是左右拼接
    submission.to_csv(r'C:\Users\林剑艺\Desktop\Kaggle房价预测\data\submission.csv', index=False)  # index=False表示不保存行索引

train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)



