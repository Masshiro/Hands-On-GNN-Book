# 多层感知机

## 1 引入隐藏层

**线性模型可能出错**

- 线性意味着<u>单调假设</u>：任何特征的增⼤都会导致模型输出的增⼤（如果对应的权重为正）
- 任何像素的重要性都以复杂的⽅式取决于该像素的上下⽂（周围像素的值）

**在网络中加入隐藏层**

- 加入一个或多个隐藏层以克服模型限制

	- 最简单：将许多全链接层堆叠在一起

	<img src="https://raw.githubusercontent.com/Masshiro/TyporaImages/master/20230715203142.png" style="zoom:80%;" />

**从线性到非线性**

- 符号：
  - $\mathbf{X}\in \mathbb{R}^{n\times d}$: $n$个样本的小批量，每个样本$d$个输入特征
  - $\mathbf{H}\in\mathbb{R}^{n\times h}$：隐藏层输出，称为隐藏表示(hidden representations)
    - $\mathbf{W}^{(1)}\in\mathbb{R}^{d\times h}$：隐藏层权重
    - $\mathbf{b}^{(1)}\in\mathbb{R}^{1\times h}$：隐藏层偏置
  - $\mathbf{O}\in\mathbb{R}^{n\times q}$：输出层输出
    - $\mathbf{W}^{(2)}\in\mathbb{R}^{h\times q}$：输出层权重
    - $\mathbf{b}^{(2)}\in\mathbb{R}^{1\times q}$：输出层偏置

- 合并隐藏层，得到等价的单层模型：
  $$
  \begin{aligned}
  \mathbf{O} &=(\mathbf{XW}^{(1)}+\mathbf{b}^{(1)} )\mathbf{W}^{(2)}+\mathbf{b}^{(2)}\\
  &=\mathbf{XW}+\mathbf{b}
  \end{aligned}
  $$

  - $\mathbf{W}=\mathbf{W}^{(1)}\mathbf{W}^{(2)}$, $\mathbf{b}=\mathbf{b}^{(1)}\mathbf{W}^{(2)}+\mathbf{b}^{(2)}$

- 对每个隐藏单元应用非线性的激活函数以发挥多层架构的潜力：

  - 有了激活函数，多层感知机不会退化为线性模型：
    $$
    \begin{aligned}
    \mathbf{H} &= \sigma(\mathbf{XW}^{(1)}+\mathbf{b}^{(1)})\\
    \mathbf{O} &= \mathbf{HW}^{(2)}+\mathbf{b}^{(2)}
    \end{aligned}
    $$

## 2 激活函数

- 通过计算加权和并加上偏置来确定神经元是否应该被激活

**ReLU**

- 实现简单，对给定元素$x$，ReLU函数被定义为该元素与$0$的最大元素
  $$
  \text{ReLU}=\max(x,0)
  $$

- 有很多变体：

  - 参数化ReLU
    $$
    p\text{ReLU}(x)=\max(x,0)+\alpha \min(0,x)
    $$

**sigmoid**

- 通常称为压缩函数：将范围$(-\infty,\infty)$中任意输出压缩至$(0,1)$

$$
\text{sigmoid}(x)=\frac{1}{1+\exp(-x)}
$$

- sigmoid在隐藏层中已经较少使⽤，它在⼤部分时候被更简单、 更容易训练的ReLU所取代

**tanh**

- tanh(双曲正切)函数也能将其输⼊压缩转换到区间$(-1,1)$上
  $$
  \tanh(x)=\frac{1-\exp(-2x)}{1+\exp(-2x)}
  $$

## 3 模型选择、欠/过拟合

