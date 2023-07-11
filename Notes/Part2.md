# Ch 4: Improving Embeddings with Biased Random Walks in Node2Vec

 

# 4.1 Differences

**Two components of DeepWalk & Node2Vec:**

- random walks
- Word2Vec

**Difference is on random walks**

- DeepWalk: obtains sequence of nodes with a uniform distribution
- Node2Vec: biased



---

# Ch5: Including Node Features with Vanilla Neural Networks

## 5.0 Visualization Tools

**yEd Live**

https://www.yworks.com/yed-live/

**Gephi**

https://gephi.org/

## 5.1 Two Datasets

`Cora` **dataset**

- For node classification
- A network of 2708 publications, where each connection is a reference
	- Each publication as a binary vector of 1433 unique words
	- 0 for presence 1 for absence
- Bag of words in NLP
- Goal: classify each node into one of seven categories

**Facebook Page-Page dataset**

- 22470 nodes
	- Each represents an official Facebook page
	- Pages are connected when there are mutual likes btw them
- Node features: 128-dim vectors
- Doesn't have training, evaluation and test masks by default

---

# Ch6: Introducing Graph Convolutional Networks

**<u>*Basic neural network layer*</u>** corresponds to a linear transformation

- $$
	h_A=x_AW^\top
	$$

	- $x_A$: the input vector of node $A$
	- $W$: weight matrix

**<u>*Graph linear layer:*</u>**

- For node $A$
	$$
	h_A = \sum_{i\in \mathcal{N}_A}x_iW^\top
	$$

- For all nodes
	$$
	\begin{aligned}
	H &=\widetilde{A}^\top XW^\top\\
	\widetilde{A} &=A+I
	\end{aligned}
	$$

	- Multiplying the input matrix by this adjacency matrix will directly sum up the neighboring node features.
	- $\widetilde{A}$: self loops are added

<u>***Normalized graph linear layer***</u>:

- $$
	h_A=\frac{1}{\text{deg(i)}}\sum_{j\in \mathcal{N}_i}x_jW^\top
	$$

- In this case, we can use degree matrix $D$:

	- $D^{-1}$: provides the normalization coefficients $\frac{1}{\text{deg}(i)}$
	- $\widetilde{D}^{-1}=(D+I)^{-1}$: add self loops

- Two options to put normalization coefficients into the formula:

	- $\widetilde{D}^{-1}\widetilde{A}XW^\top$: will normalize every <u>*row*</u> of features
	- $\widetilde{A}\widetilde{D}^{-1}XW^\top$: will normalize every <u>*column*</u> of features

- Put it into codes:

	```python
	A = np.array([
	    [1, 1, 1, 1],
	    [1, 1, 0, 0],
	    [1, 0, 1, 1],
	    [1, 0, 1, 1]
	])
	
	np.linalg.inv(D + np.identity(4)) @ A
	```

	- `@`: python built-in matrix multiplication operator
	- `np.identity(n)`: create an identity matrix $I$ of $n$ dimensions
	- `np.linalg.inv()`: calculate inverse matrix

<u>***GCN's implementation***</u>

- $$
	H=\widetilde{D}^{-\frac{1}{2}}\widetilde{A}^\top\widetilde{D}^{-\frac{1}{2}}XW^\top
	$$

	- Assign higher weights to nodes with few neighbors

- $$
	h_i=\sum_{j\in\mathcal{N}_i}\frac{1}{\sqrt{deg(i)}\sqrt{deg(j)}}x_jW^\top
	$$

---

# Ch 7: Graph Attention Networks (GATs)

**<u>*Graph attention layer*</u>**

- Main idea: some nodes are more important than others

	- GCN layer: only consider degree
	- GAT layer: also consider importance of node features

- Weighting factors: attention scores between node $i$ and $j$:  $\alpha_{ij}$

	- $$
		h_i=\sum_{j\in\mathcal{N}_i}\alpha_{ij}\mathbf{W}x_j
		$$

- Four steps to calculate scores & how to make improvement to layer:

	- Linear transformation
	- Activation function
	- Softmax normalization
	- Multi-head attention
	- Improved graph attention layer

## 7.1 Linear Transformation

- Attention score represents: importance btw a central node $i$ and a neighbor $j$

	- Requires node features from both nodes
	- Represented by a concatenation btw hidden vectors $\mathbf{W}x_i$ and $\mathbf{W}x_j$
		- $\mathbf{W}$: classic shared weight matrix to compute hidden vectors

- An additional linear transformation is applied to the result with a learnable weight matrix $W_{att}$

	- During training, matrix learns weights to produce attention coefficients $a_{ij}$

- $$
	a_{ij}=W_{att}^\top[\mathbf{W}x_i\vert\vert \mathbf{W}x_j]
	$$

## 7.2 Activation Function

- Nonlinearity is an <u>*essential component*</u> in neural networks to approximate nonlinear target functions.

- The official implementation: Leaky Rectified Linear Unit (ReLU)

- $$
	e_{ij}=LeakyReLU(a_{ij})
	$$

- The resulting values are NOT normalized

## 7.3 Softmax Function

- Compare different scores $\Rightarrow$ need normalized values on the same scale

- $$
	\alpha_{ij}=softmax(e_{ij})=\frac{\exp(e_{ij})}{\sum_{k\in\mathcal{N}_i}\exp(e_{ik})}
	$$

- Final attention scores $\alpha_{ij}$ are obtained

- Self-attention is NOT very stable

## 7.4 Multi-head Attention

- Solution: calculating multiple embeddings

	- Repeat three previous steps multiple times
	- Each instance produces an embedding $h_i^k$
		- $k$: index of attention head

- Two ways of combining:

	- <u>*Averaging*</u>:
		$$
		h_i=\frac{1}{n}\sum_{k=1}^{n}h_i^k=\frac{1}{n}\sum_{k=1}^{n}\sum_{j\in\mathcal{N}_i}\alpha_{ij}^k\mathbf{W}^kx_j
		$$

	- <u>*Concatenation*</u>:
		$$
		h_i=\vert\vert_{k=1}^{n}h_i^k=\vert\vert_{k=1}^n \sum_{j\in\mathcal{N}_i}\alpha_{ij}^k\mathbf{W}^kx_j
		$$

- In practice

	- Concatenation: when it's hidden layer
	- Averaging: when it's the last layer

## 7.5 Improvement

- Graph attention layer only computes a static type of attention

	- there are simple graph problems we cannot express with a GAT

- 静态注意力问题

	- 标准GAT中，线性层是连续应用的，所以注意力得分是不依赖于查询节点的。这意味着<u>每个节点都会对相同的邻居节点给出相同的注意力权重</u>。
	- GATv2中，每个节点可以对任何其他节点进行注意，注意力得分是依赖于查询节点和目标节点的特征的

- GATv2's approach:

	- weight matrix $W$ is applied after the concatenation
	- the attention weight matrix $W_{attr}$ after the $LeakyReLU$ function

- Comparison of formula:

	- GAT:
		$$
		\alpha_{ij}=\frac{\exp(LeackReLU(W_{att}^\top[\mathbf{W}x_i\vert\vert \mathbf{W}x_j]))}{\sum_{k\in\mathcal{N}_i}\exp(LeakyReLU(W_{att}^\top[\mathbf{W}x_i\vert\vert \mathbf{W}x_k]))}
		$$

	- GATv2:
		$$
		\alpha_{ij}=\frac{\exp(W_{att}^\top LeackReLU(\mathbf{W}[x_i\vert\vert x_j]))}{\sum_{k\in\mathcal{N}_i}\exp(W_{att}^\top LeakyReLU(\mathbf{W}[x_i\vert\vert x_k]))}
		$$

## 7.6 Implementation in `NumPy`

- Notations:

	- $$
		h_i=\sum_{j\in\mathcal{N}_i}\alpha_{ij}\mathbf{W}x_j
		$$

	- $$
		H=\widetilde{A}^\top W_\alpha X\mathbf{W}^\top
		$$

		- $W_\alpha$: matrix stores every $\alpha_{ij}$

- Example

	<img src="https://raw.githubusercontent.com/Masshiro/TyporaImages/master/20230710160417.png" style="zoom:50%;" />

	- Construct $A$:

		```python
		A = np.array([ [1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 1, 1], [1, 0, 1, 1] ])
		```

	- Randomly generate node feature $X$:

		```python
		X = np.random.uniform(-1, 1, (4, 4))
		```

	- Define weight matrix $\mathbf{W}$

		- Dimensions are (# of hidden dims, # of nodes)
		- \# of hidden dims ($dim_h$) is arbitrary, here we set as $2$

		```python
		W = np.random.uniform(-1, 1, (2, 4))
		```

	- Define attention weight matrix $W_{att}$

		- It's applied to the concatenation of hidden vectors to produce a unique value
		- Thus, its size needs to be $(1, dim_h\times 2)$

		```python
		W_att = np.random.uniform(-1, 1, (1, 4))
		```

	- Obtain pairs of source and destination nodes

		- Can be done to look at $\widetilde{A}$ in COO format

		```python
		connections = np.where(A > 0)
		```

	- Compute $a$

		```python
		a = W_att @ 
		np.concatenate([
		    (X @ W.T)[connections[0]], 
		    (X @ W.T)[connections[1]]], axis=1).T
		```

		- `np.concatenate()`: concatenates hidden vectors of  source and destination nodes
		- `(X @ W.T)[connections[0]]`: 根据`connections[0]`中的每个数字，分别选取`(X @。W.T)`结果里对应的<u>**行**</u>

	- Apply a Leaky ReLU function:

		```python
		def leaky_relu(x, alpha=0.2):
		    return np.maximum(alpha*x, x)
		
		e = leaky_relu(a)
		```

	- Place values to corresponding positions in the matrix:

		```python
		E = np.zeros(A.shape)
		E[connections[0], connections[1]] = e[0]
		```

	- Normalize every row of attention scores

		- Requires a custom `softmax` function

		```python
		def softmax2D(x, axis):
		    e = np.exp(x - np.expand_dims(np.max(x, axis=axis), axis))
		    sum = np.expand_dims(np.sum(e, axis=axis), axis)
		    return e / sum
		
		W_alpha = softmax2D(E, 1)
		```

		- `np.max(x, axis=1)`: 对第1轴进行最大值求解（列向量）

	- Calculate the matrix of embedding $H$, which should give two-dim vectors for each node:

		```python
		H = A.T @ W_alpha @ X @ W.T
		```
