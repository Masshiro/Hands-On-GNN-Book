# 1 层和块

- 单个神经网络：
	- 接受一些输入
	- 生成相应的<u>*标量*</u>输出
	- 具有一组相关参数，更新这些参数可以优化某目标函数

- 多个输出的网络，**层**：

	- 接受一组输入
	- 生成相应的输出
	- 由一组可调整参数描述

- 块：比单个层大，比整个模型小

	- 可以描述单个层、由多个层组成的组件或整个模型本⾝
	- 编程角度，块由`class`表示
		- 任何⼦类须定义⼀个将其输⼊转换为输出的<u>前向传播函数</u>
		- 且必须存储任何必需的参数

- 示例：生成一个网络，其中包含：

	- 一个具有256个单元和ReLU激活函数的全链接层

	- 一个具有10个隐藏单元且不含激活函数的全链接层

		```python
		import torch 
		from torch import nn 
		from torch.nn import functional as F
		
		net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
		X = torch.rand(2, 20)
		```

		- 通过实例化`nn.Sequential`来构建模型
		- 层的执行顺序作为参数传递
		- `nn.Sequential`定义了⼀种特殊的`Module`，即在PyTorch中表⽰⼀个块的类，它维护了⼀个由`Module`组成 的有序列表

## 1.1 自定义块

- 每个块必须提供的基本功能：

	1. 将输⼊数据作为其前向传播函数的参数
	2. 通过前向传播函数来⽣成输出
		- 输出的形状可能与输⼊的形状不同
	3. 计算其输出关于输⼊的梯度，可通过其反向传播函数进⾏访问。通常这是⾃动发⽣的。
	4. 存储和访问前向传播计算所需的参数
	5. 根据需要初始化模型参数

- 示例：编写块，包含一个MLP，其具有256个隐藏单元的隐藏层和⼀ 个10维输出层

	- `MLP`类继承了表示块的类

	- 仅需提供构造函数以及前向传播函数

		```python
		class MLP(nn.Module):
		    # ⽤模型参数声明层。这⾥，我们声明两个全连接的层
		    def __init__(self):
		        # 调⽤MLP的⽗类Module的构造函数来执⾏必要的初始化。
		        # 这样，在类实例化时也可以指定其他函数参数，
		        # 		例如模型参数params（稍后将介绍）
		        super().__init__()
		        self.hidden = nn.Linear(20, 256) # 隐藏层
		        self.out = nn.Linear(256, 10) # 输出层
		    
		    # 定义模型的前向传播，即如何根据输⼊X返回所需的模型输出
		    def forward(self, X):
		        # 注意，这⾥我们使⽤ReLU的函数版本，其在nn.functional模块中定义
		        return self.out(F.relu(self.hidden(X)))
		```

		- `forward()`：以`X`为输入，计算带有激活函数的隐藏表示，并输出其未规范化的输出值

## 1.2 顺序块

- `Sequential`：把其他模块串联起来

- 构建自定义的`MySequential`，只需定义两个关键函数：

	- 一种将块逐个追加到列表中的函数

	- 一种前向传播函数，用于将输入按追加块的顺序传递给组成的链条

		```python
		class MySequential(nn.Module):
			def __init__(self, *args):
		        super().__init__() 
		        for idx, module in enumerate(args):
		            # 这⾥，module是Module⼦类的⼀个实例。我们把它保存在'Module'类的成员
		            # 变量_modules中。_module的类型是OrderedDict
		            self._modules[str(idx)] = module
		
		    def forward(self, X):
		        # OrderedDict保证了按照成员添加的顺序遍历它们 
		        for block in self._modules.values():
		            X = block(X) 
		        return X
		    
		net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10)) net(X)
		```

		- ​	`__init__`函数将每个模块逐个添加到有序字典`_modules`中

## 1.3 前向传播函数中执行代码

- 并非所有架构都是简单的顺序架构，当需要更强的灵活性时，需要定义自己的块

- 希望合并既不是上层的结果也不是可更新参数的项目：常数参数

	- 例如：需要一个计算函数$f(\mathbf{x,w})=c\cdot\mathbf{w}^\top\mathbf{x}$的层，$c$：某个在优化过程中没有更新的指定常量

	```python
	class FixedHiddenMLP(nn.Module):
	    def __init__(self):
	        super().__init__()
	        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
	        self.rand_weight = torch.rand((20, 20), requires_grad=False)
	        self.linear = nn.Linear(20, 20)
	  
	    def forward(self, X):
	        X = self.linear(X)
	        # 使⽤创建的常量参数以及relu和mm函数
	        X = F.relu(torch.mm(X, self.rand_weight) + 1)
	        # 复⽤全连接层。这相当于两个全连接层共享参数
	        X = self.linear(X)
	        # 控制流
	        while X.abs().sum() > 1:
	            X /= 2
	       return X.sum()
	```

	- 该模型中，实现一个隐藏层，其权重`self.rand_weight`在实例化时被随机初始化
		- 不会被反向更新
		- 不是一个模型参数

# 2 参数管理

- 选择了架构并设置超参数后，即进入训练阶段：
	- 此时目标：找到使损失函数最小化的魔心参数值
	- 经过训练后，需使用这些参数做出未来的预测
- 之前仅依靠深度学习框架完成训练的工作，忽视了操作参数的具体细节，现在考虑如下问题：
	- 访问参数，用于调试、诊断和可视化
	- 参数初始化
	- 在不同模型间共享参数

- **延后初始化**
	- 直到数据第⼀次通过模型传递时，框架才会<u>*动态地*</u>推断出每个层的⼤⼩
	- 当使⽤卷积神经⽹络时，由于输⼊维度（即图像的分辨率）将影响每个后续层的维数，有了该技术将更加⽅便
	- 在编写代码时⽆须知道维度是什么就可以设置参数，这种能⼒可以⼤⼤简化定义和修 改模型的任务

# 3 读写文件

- 运行一个耗时较长的深度学习模型时，最佳做法为定期保存中间过程

## 3.1 加载和保存张量

- 对于单个张量，调用`load`和`save`函数分别读写

	- 两个函数都要求我们提供⼀个名称
	- `save`要求将要保存的变量作为输⼊

	```python
	import torch
	from torch import nn
	from torch.nn import functional as F
	
	x = torch.arange(4)
	torch.save(x, 'x-file')
	
	# 重新读回内存
	x2 = torch.load('x-file')
	torch.save(x, 'x-file')
	x2
	>> tensor([0, 1, 2, 3])
	```

- 可以存储一个**<u>张量列表</u>**，然后将其读回内存

	```python
	y = torch.zeros(4)
	torch.save([x, y], 'x-files')
	x2, y2 = torch.load('x-files')
	(x2, y2)
	>> (tensor([0, 1, 2, 3]), tensor([0., 0., 0., 0.]))
	```

- 可以写入或读取从**<u>字符串映射到张量的字典</u>**

	- 当需要读取或写入模型中的所有权重时很方便

	```python
	mydict = {'x': x, 'y': y}
	torch.save(mydict, 'mydict')
	mydict = torch.load('mydict')
	mydict2
	>> {'x': tensor([0, 1, 2, 3]), 'y': tensor([0., 0., 0., 0.])}
	```

## 3.2 加载和保存模型参数

- 深度学习框架提供了内置函数来保存和加载整个⽹络

- 将保存模型的参数**<u>*⽽不是*</u>**保存整个模型

- 模型本身难以序列化

	- 为了恢复模型，需用代码生成架构，然后从磁盘加载参数

- 以MLP为例；

	```python
	class MLP(nn.Module):
	    def __init__(self):
	        super().__init__()
	        self.hidden = nn.Linear(20, 256)
	        self.output = nn.linear(256, 10)
	       
	    def forward(self, x):
	        return self.output(F.relu(self.hidden(x)))
	
	net = MLP()
	X = torch.randn(size=(2, 20))
	Y = net(X)
	torch.save(net.state_dict(), 'mlp.params') # 存储模型参数
	
	clone = MLP()
	clone.load_state_dict(torch.load('mlp.params'))
	clone.eval()
	>> MLP( 
	    (hidden): Linear(in_features=20, out_features=256, bias=True) 
	    (output): Linear(in_features=256, out_features=10, bias=True))
	```

# 4 其余内容

参数管理&自定义层均在`5-1-ParamsManage.ipynb`中记录