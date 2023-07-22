import torch
from torch import nn
import d2l_utils as d2l


""" 读取数据集 """
batch_size= 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224, workers=4)
# print("len of train_iter: ", len(train_iter))


""" 定义训练函数"""
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    
    net.apply(init_weights)
    print("training on device ", device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    num_batches = len(train_iter)
    
    for epoch in range(num_epochs):
        print(f"\n Epoch {epoch+1} starts.")
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            # X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            # test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
            test_acc = d2l.evaluate_accuracy(net, test_iter)
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                # print(f"\t batch {i+1}")
                print(f'\t Batch {i+1}:\t train loss: {train_l:>.3f} | train acc: {train_acc:>.3f} | test acc: {test_acc:>.3f}') 
        print(f"\n Epoch {epoch+1} finished.")
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, ' f'test acc {test_acc:.3f}')



if __name__ == "__main__":
    """ 定义网络 """
    net = nn.Sequential(
        # 这⾥使⽤⼀个11*11的更⼤窗⼝来捕捉对象。 
        # 同时，步幅为4，以减少输出的⾼度和宽度。
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 减⼩卷积窗⼝，使⽤填充为2来使得输⼊与输出的⾼和宽⼀致，且增⼤输出通道数
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 使⽤三个连续的卷积层和较⼩的卷积窗⼝。
        # 除了最后的卷积层，输出通道的数量进⼀步增加。
        # 在前两个卷积层之后，汇聚层不⽤于减少输⼊的⾼度和宽度
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        # 这⾥，全连接层的输出数量是LeNet中的好⼏倍。使⽤dropout层来减轻过拟合
        nn.Linear(6400, 4096), nn.ReLU(),
        nn.Dropout(p=0.5), 
        nn.Linear(4096, 4096), nn.ReLU(), 
        nn.Dropout(p=0.5), 
        # 最后是输出层。由于这⾥使⽤Fashion-MNIST，所以⽤类别数为10，⽽⾮论⽂中的1000 
        nn.Linear(4096, 10)
    )
    """ 观察网络 """
    # 因为输入是3*224*224的图片，这里构造一个224*224的单通道数据，
    # 用以观察每一层输出的形状
    # X = torch.randn(1, 1, 224, 224)
    # print("Input data:\t", X)
    # for layer in net:
    #     X = layer(X)
    #     print(layer.__class__.__name__,'output shape:\t', X.shape)
    lr, num_epochs = 0.01, 10
    train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.set_device())
