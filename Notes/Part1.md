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

## 3.2 Introducing Word2Vec

