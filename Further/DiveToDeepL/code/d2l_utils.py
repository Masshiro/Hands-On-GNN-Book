import torch
from torch import nn
import torchvision 
from torch.utils import data 
from torchvision import transforms
import matplotlib.pyplot as plt


"""
画图相关
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

"""
Chapter 3: Linear Regression
"""
def get_dataloader_workers():
    return 4

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的⽂本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'] 
    return [text_labels[int(i)] for i in labels]

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../../dataset", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../../dataset", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))



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














# if __name__ == "__main__":
#     train_iter, test_iter = load_data_fashion_mnist(256)