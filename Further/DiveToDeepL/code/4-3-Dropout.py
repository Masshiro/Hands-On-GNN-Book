import torch
from torch import nn
import d2l_utils as d2l

"""
1 从零实现
"""
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 该情况中，所有元素均被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 该情况中，所有元素均被保留
    if dropout == 0:
        return X
    # 以dropout的概率丢弃张量输入X中的元素
    mask = (torch.rand(X.shape) > dropout).float()
    # 重新缩放剩余部分
    return mask * X / (1.0 - dropout)

def test_dropout_layer(row, col, p):
    assert 0 <= p <= 1
    X = torch.arange(row*col, dtype = torch.float32).reshape((row, col))
    print(dropout_layer(X, p))

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    """定义模型"""
    # 将暂退法应用于每个隐藏层的输出（在激活函数之后）
    # 常 ⻅的技巧是在靠近输⼊层的地⽅设置较低的暂退概率：
    #   模型将第⼀个和第⼆个隐藏层的暂退概率分别设置为0.2和0.5，
    #   且暂退法只在训练期间有效
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, 
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs 
        self.training = is_training 
        self.lin1 = nn.Linear(num_inputs, num_hiddens1) 
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2) 
        self.lin3 = nn.Linear(num_hiddens2, num_outputs) 
        self.relu = nn.ReLU()
    
    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H2))
        if self.training == True:
            # 在第二个全链接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out