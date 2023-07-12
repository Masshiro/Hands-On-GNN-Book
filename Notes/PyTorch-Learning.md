# 《动手学深度学习》PyTorch-v2

# 1. 基础

## 1.1 数据操作

**<u>Tensor</u>**：$n$维数组

- 与`numpy`的`ndarray`相似
- ==NumPy仅支持CPU，而深度学习中的张量类可以很好地支持GPU加速==

**理解广播机制**

```python
a = torch.arange(3).reshape((3, 1)) 
b = torch.arange(2).reshape((1, 2))
a + b
```

- `a`和 `b`形状不匹配：
	- 矩阵`a`复制列，矩阵`b`复制行，然后按元素相加

**节省内存**

```python
before = id(Y) 
Y = Y + X 
id(Y) == before
>> False
```

- `id()`：用于提供内存中引用对象的确切地址

- Python首先计算Y+X，随后<u>*为结果分配新的内存*</u>，然后使Y指向内存中的这个新位置

- 不可取：

	- 不必要，数百兆参数，希望原地更新
	- 代码可能无意引用指向旧的内存位置的参数

- 解决：

	- 使用<u>*切片*</u>将操作结果分配给先前的数组

	- ```python
		before = id(x)
		X += Y # or X[:] = X + Y
		id(x) == before
		>> True
		```

## 1.2 数据预处理

TODO

## 1.3 线性代数

**标量**

- 标量变量由<u>*普通⼩写字⺟*</u>表⽰（$x$, $y$）

- 由只有⼀个元素的张量表⽰

	```python
	x = torch.tensor(3.0)
	```

**向量**

- 向量通常记为<u>*粗体⼩写*</u>的符号（$\mathbf{x}, \mathbf{y}$）

- 由一维张量表示

	```python
	x = torch.arange(4)
	x
	>> tensor([0, 1, 2, 3])
	```

- 长度：向量即为一个数字数组，向量也有长度
	- 数学上，一个向量$\mathbf{x}$有$n$个实值标量组成，则$\mathbf{x}\in\mathbb{R}^n$
	- `len(x)`
- 维度：
	- <u>向量或轴</u>的维度 $\Rightarrow$ 向量或轴的⻓度(向量或轴的元素数量)
	- <u>张量</u>的维度 $\Rightarrow$ 张量具有的轴数

**矩阵**

- 通常⽤<u>*粗体⼤写字⺟*</u>来表⽰（$\mathbf{X},\mathbf{Y}$）
- $\mathbf{A}\in\mathbb{R}^{m\times n}$: 则$\mathbf{A}$的形状为$(m,n)$

**张量**

- ⽤特殊字体的⼤写字⺟表⽰（$\mathsf{X}$）

- 描述具有任意数量轴的$n$维数组通用方法
	- 向量：一阶张量
	- 矩阵：二阶张量

- Hadamard积：按元素乘法

	```python
	A = torch.arange(20, dtype=torch.float32).reshape(5, 4) 
	B = A.clone() # 通过分配新内存，将A的⼀个副本分配给B
	A * B
	```

**降维**

- 可通过指定`axis`来指定沿哪一个轴来降维
- 若不想改变轴数：`sum_A = A.sum(axis=1, keepdims=True)`

**范数**

- 通俗讲，向量的范数是表⽰⼀个向量有多⼤

- $L2$范数：$\vert\vert \mathbf{x}\vert\vert_2=\sqrt{\sum_{i=1}^nx_i^2}$

	```python
	u = torch.tensor([3.0, -4.0]) 
	torch.norm(u)
	>> tensor(5.)
	```

- $L1$范数：$\vert\vert \mathbf{x}\vert\vert_1=\sum_{i=1}^n\vert x_i\vert$

	```python
	torch.abs(u).sum()
	>> tensor(7.)
	```

- 矩阵的Frobenius范数：$\vert\vert\mathbf{X}\vert\vert_F=\sqrt{\sum_{i=1}^{m}\sum_{j=1}^nx^2_{ij}}$, $\mathbf{X}\in \mathbb{R}^{m\times n}$

	```python
	torch.norm(torch.ones((4, 9)))
	>> tensor(6.)
	```

## 1.4 微积分

深度学习中，我们“训练”模型，不断更新它们，使它们在看到越来越多的数据时变得越来越好：

- 变得更好意味着最⼩化⼀个损失函数（loss function）
- 但“训练”模型只能将模型 与我们实际能看到的数据相拟合

拟合模型的任务可分解为两个关键问题：

- 优化（optimization）：⽤模型拟合观测数据的过程
- 泛化（generalization）：数学原理和实践者的智慧，能够指导我们⽣成出有效性超出⽤于训练的数据集 本⾝的模型。



**导数与微分**

- 假设函数$f,g$可微，$C$是一个常数，则有如下法则：

	- 常数相乘法则：
		$$
		\frac{d}{dx}[Cf(x)]=C\frac{d}{dx}f(x)
		$$

	- 加法法则：
		$$
		\frac{d}{dx}[f(x)+g(x)]=\frac{d}{dx}f(x)+\frac{d}{dx}g(x)
		$$

	- 乘法法则：
		$$
		\frac{d}{dx}[f(x)g(x)]=f(x)\frac{d}{dx}[g(x)]+g(x)\frac{d}{dx}[f(x)]
		$$

	- 除法法则：
		$$
		\frac{d}{dx}\left[\frac{f(x)}{g(x)}\right]=\frac{g(x)\frac{d}{dx}[f(x)]-f(x)\frac{d}{dx}[g(x)]}{[g(x)]^2}
		$$

**梯度**

- 函数$f:\mathbb{R}^n\rightarrow \mathbb{R}$的输入是一个$n$维向量$\mathbf{x}=[x_1,x_2,\dots, x_n]^\top$，且输出是一个标量
	$$
	\nabla_\mathbf{x}f(\mathbf{x})=\left[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2},\cdots,\frac{\partial f(\mathbf{x})}{\partial x_n} \right]^\top
	$$

- 设$\mathbf{x}$是$n$维向量，在微分多元函数时有如下常用规则：

	- 对于所有$\mathbf{A}\in\mathbb{R}^{m\times n}$，都有$\nabla_\mathbf{x}\mathbf{A}\mathbf{x}=\mathbf{A}^\top$
	- 对于所有$\mathbf{A}\in\mathbb{R}^{n\times m}$，都有$\nabla_\mathbf{x}\mathbf{x}^\top\mathbf{A}=\mathbf{A}$
	- 对于所有$\mathbf{A}\in\mathbb{R}^{n\times n}$，都有$\nabla_\mathbf{x}\mathbf{x}^\top\mathbf{A}\mathbf{x}=(\mathbf{A}+\mathbf{A}^\top)\mathbf{x}$
	- $\nabla_\mathbf{x}\Vert x\Vert^2=\nabla_\mathbf{x}\mathbf{x}^\top\mathbf{x}=2\mathbf{x}$

## 1.5 自动微分

- 深度学习框架通过⾃动计算导数来加快求导
	- 根据设计 好的模型，系统会构建⼀个计算图（computational graph），来跟踪计算是哪些数据通过哪些操作组合起来产⽣输出
	- ⾃动微分使系统能够随后反向传播梯度
	- 反向传播（backpropagate）意味着跟踪整个计算图，填充关于每个参数的偏导数

**简单例子**

对函数$y=2\mathbf{x}^\top\mathbf{x}$关于列向量$\mathbf{x}$求导

- 创建`x`:

	```python
	x = torch.arange(4.0)
	```

- 计算前，需要一个地方存储梯度，且不会在每次对一个参数求导时都分配新的内存

	```python
	x.requires_grad_(True) # 等价于x=torch.arange(4.0,requires_grad=True)
	x.grad # 默认值是None
	```

- 计算$y$

	```python
	y = 2 * torch.dot(x, x)
	y
	>> tensor(28., grad_fn=<MulBackward0>)
	```

- 通过<u>*调⽤反向传播函数*</u>来⾃动计算`y`关于`x`每个分量的梯度，并打印这些梯度

	```python
	y.backward() 
	x.grad
	>> tensor([ 0., 4., 8., 12.])
	```

**分离计算**

- 需求：希望将某些计算移动到记录的计算图之外

	- 假设y是作为x的函数计算的，⽽z则是作为y和x的函数计算的
	- 想计算z关于x的梯度，但由于某种原因，希望将y视为⼀个常数
	- 只考虑到x在y被计算后发挥的作⽤

- 解决：

	- 分离y来返回⼀个新变量u，该变量与y具有相同的值，但丢弃计算图中如何计算y的任何信息

	- 梯度不会向后流经u到x

	- 反向传播函数计算`z=u*x`关于x的偏导数，同时将u作为常数处理，⽽不是`z=x*x*x`关于x的偏导数

		```python
		x.grad.zero_()
		y = x * x
		u = y.detach()
		z = u * x
		
		z.sum().backward() 
		x.grad == u
		>> tensor([True, True, True, True])
		```

	- 由于记录了y的计算结果，可以随后在y上调⽤反向传播，得到`y=x*x`关于的x的导数，即`2*x`

		```python
		x.grad.zero_() 
		y.sum().backward() 
		x.grad == 2 * x
		>> tensor([True, True, True, True])
		```

## 1.6 查阅文档

**查找模块中的所有函数与类**

```python
print(dir(torch.distributions))
```

- 调用`dir`函数
- 查询随机数生成模块中的所有属性

**查找特定函数和类的用法**

```python
help(torch.ones)
```

---



# 2. 线性神经网络

## 2.1 线性回归

**术语**

- 训练数据集
- 每⾏数据：样本（sample）、数据点（data point）或数据样本（data instance）
- 试图预测的⽬标：标签（label）或⽬标（target）
- 预测所依据的⾃变量：特征（feature）或协变量（covariate）

**基本要素**

- 线性模型：

	- 将所有特征放到向量$\mathbf{x}\in\mathbb{R}^d$

	- 将所有权重放到向量$\mathbf{w}\in\mathbb{R}^d$
		$$
		\hat{y}=\mathbf{w}^\top\mathbf{x}+b
		$$

		- 其中$\mathbf{x}$对应单个数据样本的特征

	- 用矩阵$\mathbf{X}\in\mathbb{R}^{n\times d}$可以⽅便地引⽤我们整个数据集的$n$个样本

		- 每一行：一个样本；每一列：一种特征
			$$
			\hat{\mathbf{y}}=\mathbf{X}\mathbf{w}+b
			$$

	- 开始寻找最好的model parameters $\mathbf{w}$和$b$之前，仍需：

		- 一种模型质量的度量方式
		- 一种能够更新模型以提高模型预测质量的方法

- 损失函数：

	- $$
		l^{(i)}(\mathbf{w}, b)=\frac{1}{2}\left(\hat{y}^{(i)}-y^{(i)} \right)^2
		$$

		- 系数$1/2$无本质影响

	- $$
		L(\mathbf{w},b)=\frac{1}{n}\sum_{i=1}^{n}l^{(i)}(\mathbf{w},b)
		$$

		- 度量模型在整个数据集上的质量
		- 计算在训练集$n$个样本上的损失均值（也等价于求和）

	- 训练时，希望找到一组参数$(\mathbf{w}^*,b^*)$从而最小化训练样本上的总损失：
		$$
		\mathbf{w}^*,b^* = \arg\min_{\mathbf{w},b} L(\mathbf{w},b)
		$$
		