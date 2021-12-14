import torch
import matplotlib.pyplot as plt

#定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data

#计算分类准确率，train_ch3函数要用
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

#训练模型
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    loss_list, train_acc_list, test_acc_list = [], [], []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()  # 对optimizer实例调用step函数，从而更新权重和偏差


            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
        loss_list.append(train_l_sum / n)
        train_acc_list.append(train_acc_sum / n)
        test_acc_list.append(test_acc)

    return loss_list, train_acc_list, test_acc_list

#画图展示
def show_img(num_epochs, loss_list, train_acc_list, test_acc_list):
    #epochs_list = list(map(lambda x: x + 1, list(range(num_epochs))))  ## 构造训练周期的横坐标
    epochs_list = range(1, num_epochs + 1)

    plt.subplot(211)
    plt.plot(epochs_list, train_acc_list, label='train acc', color='r')
    plt.plot(epochs_list, test_acc_list, label='test acc', color='g')
    plt.title('Train acc and test acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.tight_layout()
    plt.legend()

    plt.subplot(212)
    plt.plot(epochs_list, loss_list, label='train loss', color='b')
    plt.title('Train loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.legend()
    plt.show()

#先定义作图函数semilogy，其中 yy 轴使用了对数尺度
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    #d2l.set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()

# 我们在对模型评估的时候不应该进行丢弃，所以我们修改一下evaluate_accuracy函数,v2表示第二个版本
def evaluate_accuracy_v2(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        if isinstance(net, torch.nn.Module):  #nn的核心数据结构是Module
            net.eval() # 评估模式, 这会关闭dropout
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            net.train() # 改回训练模式
        else: # 自定义的模型
            if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                # 将is_training设置成False
                acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
            else:
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n