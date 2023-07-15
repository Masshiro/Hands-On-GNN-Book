import torch 
# import torchvision 
# from torch.utils import data 
# from torchvision import transforms
# import matplotlib.pyplot as plt
import d2l_utils as d2l
import numpy as np
from torch import nn

"""
0. 初始化参数
"""
num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
lr = 0.1
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

"""
1. 从零实现
"""
def softmax(X):
    """定义softmax操作"""
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def net(X):
    """定义模型"""
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def cross_entropy(y_hat, y):
    """定义交叉熵损失函数"""
    """
        y_hat：包含2个样本在3个类别的预测概率
        y：2个样本的对应标签
        使用y作为y_hat中概率的索引
            y = torch.tensor([0, 2])
            y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
            y_hat[[0, 1], y]
            >> tensor([0.1000, 0.5000])
    """
    return - torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y):
    """定义分类精度"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval() # 将模型设置为评估模式
    metric = d2l.Accumulator(2) # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():           # 上下文管理器来禁用自动求导功能，因为我们不需要计算参数的二阶导数
        for param in params:
            # 参数更新方法：
            # 用参数当前的值减去学习率乘以参数当前的梯度除以小批量数据的大小
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def updater(batch_size):
    return sgd([W, b], lr, batch_size)

def train_epoch_ch3(net, train_iter, loss, updater):
    """训练一个迭代周期"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = d2l.Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater, draw=False):
    """训练模型"""
    all_train_loss = []
    all_train_acc = []
    all_test_acc = []

    for epoch in range(num_epochs):
        print("epoch ", epoch)
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        all_train_loss.append(train_metrics[0])
        all_train_acc.append(train_metrics[1])
        all_test_acc.append(test_acc)
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
    if draw:
        y = [list(all_train_acc), list(all_train_loss), list(all_test_acc)]
        x = [i+1 for i in range(num_epochs)]
        # print("x", x)
        # print("y", y)
        linenames = ["train acc", "train loss", "test acc"]
        d2l.plot_train_result(x, y, linenames=linenames, xlabel="epoch", ylim=(0.2, 1.0), xlim=(0.5, 10.5), legend=True)

def main_scratch():
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater, True)


"""
2. 简洁实现
"""
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

def main_concise():
    # 初始化参数
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    net.apply(init_weights)
    # 定义损失函数
    loss = nn.CrossEntropyLoss(reduction='none')
    # 定义优化算法
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    # 训练
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


if __name__ == "__main__":
    # main_scratch()
    main_concise()