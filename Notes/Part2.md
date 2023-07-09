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
