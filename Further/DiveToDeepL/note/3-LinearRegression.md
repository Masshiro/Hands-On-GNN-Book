# Softmax回归

## 1. 网络结构

### 1.1 分类问题

表示分类数据的简单方法：独热编码(one-hot encoding)

- 是一个向量，分量和类别一样多
- 类别对应的分量设为1，其余为0

## 1.2 网络结构

为了估计所有可能类别的概率，需要一个多输出模型：

- 每个类别对应一个输出

- 需要一个和输出一样多的仿射函数

- 以4个特征与3个可能的输出类别为例

- softmax回归也是⼀个单层神经⽹络

	![](https://raw.githubusercontent.com/Masshiro/TyporaImages/master/20230714141029.png)

	- 向量形式：$\mathbf{o}=\mathbf{Wx+b}$

全链接的参数开销问题：

- 具体而言：对于任何具有$d$个输入和$q$个输出的全链接层，参数开销为$\mathcal{O}(dq)$
- 可自定义超参数$n$来减少开销至$\mathcal{O}(\frac{dq}{n})$

## 1.3 softmax运算

- 若要将输出视为概率，必须保证在任何数据上的输出都是非负且总和为1

- 同时需要一个训练的目标函数，来激励模型估计概率

**softmax函数**：

- 将为规范化的预测变换为⾮负数并且总和为1，同时让模型保持可导的性质：

$$
\hat{\mathbf{y}}=\text{softmax}(\mathbf{o}), \;\text{ where } \hat{y}_j=\frac{\exp(o_j)}{\sum_{k}\exp(o_k)}
$$

- softmax不会改变未规范化的预测$\mathbf{o}$之间的大小次序，只会确定分配给每个类别的概率

## 1.4 小批量样本矢量化

- 读取一个批量的样本$\mathbf{X}$：

	- 特征维度（输入数量）：$d$
	- 批大小：$n$

- 假设输出$q$个类别，则：$\mathbf{X}\in\mathbb{R}^{n\times d},\mathbf{W}\in\mathbb{R}^{d\times q}, \mathbf{b}\in\mathbb{R}^{1\times q}$

- 则矢量计算式：
	$$
	\begin{aligned}
	\mathbf{O} &= \mathbf{XW+b}\\
	\mathbf{\hat{Y}} &=\text{softmax}(\mathbf{O})
	\end{aligned}
	$$

	- $\mathbf{X}$的每一行表示一个数据样本，则softmax可以按行运算

## 1.5 损失函数

**对数似然**

- softmax给出的向量$\mathbf{\hat{y}}$：对给定任意输⼊$\mathbf{x}$的每个类的条件概率

- 设整个数据集$\{\mathbf{X,Y} \}$ 有$n$个样本：

	- 索引为$i$的样本 = 特征向量$\mathbf{x}^{(i)}$ + 独热标签向量$\mathbf{y}^{(i)}$ 

	- 将估计值与实际值比较：
		$$
		P(\mathbf{Y}\vert\mathbf{X})=\prod_{i=1}^{n}P(\mathbf{y}^{(i)}\vert\mathbf{x}^{(i)})
		$$

	- 最大化$P$，相当于最小化负对数似然：
		$$
		\begin{aligned}
		-\log P(\mathbf{X}\vert\mathbf{Y}) &= \sum_{i=1}^n-\log P(\mathbf{y}^{(i)}\vert\mathbf{x}^{(i)})\\
		&=\sum_{i=1}^{n}l(\mathbf{y}^{(i)},\mathbf{\hat{y}}^{(i)})
		\end{aligned}
		$$

	- 对于任何标签$\mathbf{y}$和模型预测$\mathbf{\hat{y}}$ ：
		$$
		\begin{aligned}
		l(\mathbf{y},\mathbf{\hat{y}}) &=-\sum_{j=1}^qy_i\log \hat{y}_j\\
		&= -\sum_{j=1}^{q}y_j\log \frac{\exp(o_j)}{\sum_{k=1}^{q}\exp(o_k)}\\
		&=\sum_{j=1}^qy_j\log\sum_{k=1}^q\exp(o_k)-\sum_{j=1}^qy_jo_j \\
		&=\log\sum_{k=1}^q\exp(o_k) - \sum_{j=1}^q y_jo_j
		\end{aligned}
		$$

- 

**softmax及其导数**
$$
\partial_{o_j} l(\mathbf{y},\mathbf{\hat{y}})=\frac{\exp(o_j)}{\sum_{k=1}^q\exp(o_k)}-y_j
$$


