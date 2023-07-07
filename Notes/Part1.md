# Ch1: Getting Started with Graph Learning

## 1.1 Graph Learning

Four prominent families of graph learning techniques:

- **Graph signal processing**
- **Matrix factorization**, seeks to find low-dimensional representations of large matrices
- **Random walk**. By simulating random walks over a graph, information about the relationships between nodes can be gathered.
- **Deep learning**. Deep learning methods can effectively encode and represent graph data as vectors.

These techniques are not mutually exclusive and often overlap in applications.

## 1.2 GNN

- GNNs are most effective when applied to specific problems. 
- These problems are characterized by high complexity, meaning that learning good representations is critical to solving the task at hand.

---

# Ch2: Graph Theory for GNN

## 2.1 Types of Graphs

**Tree**

- Connected undirected graph without cycles
- Often used to model hierarchical structures

**Rooted tree**

- A tree in which one node is designated as root

**Directed Acyclic Graph (DAG)**

- directed graph that has no cycles
- edges can only be traversed in a particular direction
- there are no loops or cycles
- often used to model dependencies between tasks or events

**Bipartie graph**

- a graph in which the vertices can be divided into two disjoint sets
- all edges connect vertices in different sets

**Complete graph**

- a graph in which every pair of vertices is connected by an edge

## 2.2 Graph concepts

### 2.2.1 Fundamental objects

- An edge is said to be <u>***incident***</u> on a node if that node is one of the edge’s endpoints.
- **<u>*Degree*</u>** can be defined for both directed and undirected graphs
	- In an undirected graph, the degree of a vertex is the number of edges that are connected to it. (self-loop adds two to the degree)
	- In a directed graph, the degree is divided into two types: indegree ($\text{deg}^-(v)$) and outdegree  ($\text{deg}^+(v)$)
- Two nodes are said to be **<u>*adjacent*</u>** if they share at least one common neighbor.

### 2.2.2 Graph measures

- **Degree centrality**
	- defined as the degree of the node. 
	- A high degree centrality indicates that a vertex is highly connected to other vertices in the graph, and thus significantly influences the network.
- **Closeness centrality** 
	- measures how close a node is to all other nodes in the graph. 
	- It corresponds to the <u>*average length of the shortest path between the target node and all other nodes in the graph*</u>. 
	- A node with high closeness centrality can quickly reach all other vertices in the network.
- **Betweenness centrality** 
	- measures the number of times a node lies on the shortest path between pairs of other nodes in the graph. 
	- A node with high betweenness centrality acts as a bottleneck or bridge between different parts of the graph.
- **Density**
	- indicates how connected a graph is
	- It is a ratio between the actual number of edges and the maximum possible number of edges in the graph.
	- Undirected graph with $n$ nodes: maximum possible num of edges is $\frac{n(n-1)}{2}$
	- Directed graph with $n$ nodes, $n(n-1)$

### 2.2.3 Adjacency matrix representation

- **adjacency matrix** has a space complexity of $O(\vert V\vert)^2$
	- $\vert V\vert=$ # of nodes in graph
- **Edge list**
	- Each edge is represented by a tuple or a pair of vertices.
	- Has space complexity of $O(\vert E\vert)$
- **Adjacency list**
	- consists of a list of pairs, where each pair represents a node in the graph and its adjacent nodes
	- adding a node or an edge can be done in constant time
	- checking whether two vertices are connected can be slower than with an adjacency matrix

## 2.3 Graph Algorithms

### 2.3.1 Breadth-First Search

### 2.3.2 Depth-First Search

---

# Ch3: Creating Node Representation with DeepWalk

## 3.1 Overview

Two major components of DeepWalk architecture:

- Word2Vec
- Random walks

## 3.2 Word2Vec

### 3.2.1 Intro

- A technique to translate words into <u>*vectors*</u> (also known as embeddings)
- Cosine similarity can be used to measure the likeness of these words

### 3.2.2 CBOW vs skip-gram

- A model must be trined on a pretext task to produce these vectors
- its only goal is to produce high-quality embeddings

Two architectures with similar tasks:

- **The continuous bag-of-words (CBOW) model**: This is trained to predict a word using its <u>*surrounding context*</u>
- **The continuous skip-gram model**: feed a single word to the model and try to predict the words around it.

### 3.2.3 Creating skip-gram

- Skip-grams are implemented as pairs of words `(target word, context word)`
	- `target word`: the input
	- `context word`: the word to predict
- The number of skip grams for the same target word depends on a parameter called **<u>*context size*</u>**

### 3.2.4 The skip-gram model

- Goal: to produce high-quality word embeddings

	- Maximize the sum of every probability of seeing a context word given a target word in an entire text ($c$ is the size of the context vector):
		$$
		\frac{1}{N}\sum_{n=1}^{N}\sum_{-c\le j\le c,j\ne0} \log p(w_{n+1}\vert w_n)
		$$

- Why $\log$ ?: 

	- products become additions, and multiplications are more computationally expensive than additions
	- The way computers store very small numbers (such as 3.14e-128) is not perfectly accurate, unlike the log of the same numbers (-127.5 in this case). These small errors can add up and bias the final results when events are extremely unlikely.

- The basic skip-gram model uses the **<u>*softmax*</u>** function to calculate the probability of a context word embedding $h_c$ given a target word embedding $h_t$:

	- $$
		p(w_c\vert w_t)=\frac{\exp(h_ch_t^\top)}{\sum_{i=1}^{\vert V\vert}\exp(h_i h_t^\top)}
		$$

- Only two layers

	- **Projection layer** with a weight matrix $W_{embed}$, which takes a one-hot encoded-word vector as an input and returns the corresponding $N$-dim word embedding. It acts as a simple lookup table that stores embeddings of a predefined dimensionality
	- **Fully connected layer** with a  weight matrix $W_{output}$, which takes a word embedding as input and outputs $\vert V\vert$-dim logits. A softmax function is applied to these predictions to transform logits into probabilities.

