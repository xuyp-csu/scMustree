# scMustree
## Introduction
Inference and visualization of multi-scale cell tree for decoding functional diversity.

![alt text](https://github.com/xuyp-csu/scMustree/blob/main/scMustree_overview.png)

## Getting Started
### Hardware requirements
`scMustree` package requires only a standard computer with enough RAM to support the in-memory operations. For large datasets, bigger RAM is recommended.

### System Requirements
#### OS Support
- **Linux**: (tested on Ubuntu 16.04.7)
- **macOS**: (tested on Monterey 12.6.3)

#### Prerequisites

	Python --- 3.9.19
	h5py --- 3.11.0
	networkx --- 3.2.1
	numpy --- 1.26.4
	pandas --- 2.2.2
	python-louvain --- 0.16
	Scanpy --- 1.10.2
	scikit-learn --- 1.5.1
	scipy --- 1.13.1
	tqdm --- 4.66.5
 	pydot --- 3.0.1
 	faiss-cpu --- 1.8.0
	pygraphviz --- 1.9
 
```markdown
Note: Dependencies are pinned to specific versions to ensure reproducibility. If you encounter compatibility issues, try relaxing version constraints.
```
### Installation
1. **Necessary Step:** Download from Github:
   	```bash
	git clone https://github.com/xuyp-csu/scMustree.git
	cd scMustree
 	```

2. **Necessary Step:** Create and activate the environment:
	```bash
	conda create -n scMustree_env python=3.9.19
 	source activate scMustree_env
 	```
 	-Use `conda activate` instead of `source activate`, if required.
   
	-Building typically completes in about 30 seconds.

3. **Necessary Step:** Install dependencies with pip and conda:

	```bash
	pip install -r requirements.txt
 	conda install -c pytorch/label/nightly faiss-cpu=1.8.0
	conda install conda-forge::pygraphviz=1.9
	```
 	-Installation typically requires around a few minutes, depending on network conditions.
