import torch
from torch import nn
import torchvision 
from torch.utils import data 
from torchvision import transforms
import matplotlib.pyplot as plt


"""
通用代码
"""
# def use_svg_display():
#     """使⽤svg格式在Jupyter中显⽰绘图"""
#     set_matplotlib_formats('svg')

# class Animator:
#     """在动画中绘制数据"""
#     def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
#                  ylim=None, xscale='linear', yscale='linear',
#                  fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
#                  figsize=(3.5, 2.5)):
#         # 增量地绘制多条线
#         if legend is None:
#             legend = []
#         ;

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表""" 
    figsize = (num_cols * scale, num_rows * scale) 
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize) 
    axes = axes.flatten() 
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图⽚张量 
            ax.imshow(img.numpy()) 
        else:
            # PIL图⽚ 
            ax.imshow(img) 
        ax.axes.get_xaxis().set_visible(False) 
        ax.axes.get_yaxis().set_visible(False) 
        if titles:
            ax.set_title(titles[i]) 
    return axes

def plot_train_result(x, y, fs=(6,4), linenames=None, xlabel=None, ylabel=None, legend=False, xlim=None, ylim=None):
    plt.figure(figsize=fs)
    for i in range(len(y)):
        assert len(y[i]) == len(x)
        plt.plot(x, y[i], label=linenames[i] if linenames is not None else None)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if legend:
        plt.legend()
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    plt.show()

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def set_device(dev='cpu'):
    if dev == "cude" and torch.cuda.is_available():
        return torch.device('cuda')
    elif dev == "mps" and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


"""
Chapter 3: Linear Regression
"""
def synthetic_data(w, b, nums_examples):
    """生成形如 y = Xw + b + noise 的数据集"""
    X = torch.normal(0, 1, (nums_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():           # 上下文管理器来禁用自动求导功能，因为我们不需要计算参数的二阶导数
        for param in params:
            # 参数更新方法：
            # 用参数当前的值减去学习率乘以参数当前的梯度除以小批量数据的大小
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def get_dataloader_workers():
    return 4

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的⽂本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'] 
    return [text_labels[int(i)] for i in labels]

def load_data_fashion_mnist(batch_size, resize=None, workers=4):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../../dataset", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../../dataset", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=workers),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=workers))

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

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
    metric = Accumulator(2) # 正确预测数、预测总数
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

def train_epoch_ch3(net, train_iter, loss, updater):
    """训练一个迭代周期"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
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
        print("epoch ", epoch+1)
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
        plot_train_result(x, y, linenames=linenames, xlabel="epoch", ylim=(0.2, 1.0), xlim=(0.5, 10.5), legend=True)


"""
Chapter 4: MLP
"""
def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失""" 
    metric = Accumulator(2) # 损失的总和,样本数量 
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel()) 
    return metric[0] / metric[1]

"""
Chapter 6: CNN
"""
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]









# if __name__ == "__main__":
#     train_iter, test_iter = load_data_fashion_mnist(256)