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

**GCN's implementation**

- $$
	H=\widetilde{D}^{-\frac{1}{2}}\widetilde{A}^\top\widetilde{D}^{-\frac{1}{2}}XW^\top
	$$

	- Assign higher weights to nodes with few neighbors

- $$
	h_i=\sum_{j\in\mathcal{N}_i}\frac{1}{\sqrt{deg(i)}\sqrt{deg(j)}}x_jW^\top
	$$

- 
