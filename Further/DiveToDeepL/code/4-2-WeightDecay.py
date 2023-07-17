import torch
from torch import nn
import d2l_utils as d2l

"""
高维线性回归
"""
# 实验settings
#   为使过拟合效果更加明显，将问题维度增至d=200
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size) 
test_data = d2l.synthetic_data(true_w, true_b, n_test) 
test_iter = d2l.load_array(test_data, batch_size, is_train=False)


"""
1 从零实现
"""
def init_params():
    """初始化模型参数"""
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    """定义L2范数惩罚项"""
    return torch.sum(w.pow(2)) / 2

def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    """均⽅损失""" 
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def train_scratch(lambd):
    """拟合训练数据集，并在测试数据集上进行评估"""
    w, b = init_params()
    net, loss = lambda X: linreg(X, w, b), squared_loss
    num_epochs, lr = 100, 0.003
    
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            train_loss = d2l.evaluate_loss(net, train_iter, loss)
            test_loss = d2l.evaluate_loss(net, test_iter, loss)
            print(f'Epoch {epoch:>3} | Train-set Loss: {train_loss:.3f} | Test-set Loss: {test_loss:.3f}')
    print('w的L2范数为：', torch.norm(w).item())


"""
2 简洁实现
"""
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数衰减
    trainer = torch.optim.SGD([
        {"params": net[0].weight, 'weight_decay': wd},
        {"params": net[0].bias}], lr=lr)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            train_loss = d2l.evaluate_loss(net, train_iter, loss)
            test_loss = d2l.evaluate_loss(net, test_iter, loss)
            print(f'Epoch {epoch:>3} | Train-set Loss: {train_loss:>3.3f} | Test-set Loss: {test_loss:.3f}')
    print('w的L2范数为：', net[0].weight.norm().item())


if __name__ == "__main__":
    # print("\nDisable Penalty:")
    # train_scratch(0)
    # print("\nEnable Penalty:")
    # train_scratch(3)

    print("\nDisable Penalty:")
    train_concise(0)
    print("\nEnable Penalty:")
    train_concise(3)