# scMustree
## Introduction
Cluster decomposition-based Anomaly Detection method (scCAD) is used to effectively identify rare cell types in scRNA-seq data.

![alt text](https://github.com/xuyp-csu/scMustree/blob/main/scMustree_overview.png)

## Getting Started
## Hardware requirements
`scMustree` package requires only a standard computer with enough RAM to support the in-memory operations.

### OS Requirements
This package is supported for *Linux* and *macOS*. The package has been tested on the following systems:
+ Linux: Ubuntu 16.04.7
+ macOS: Monterey 12.6.3

### Prerequisites

	Python --- 3.7.13
	h5py --- 3.7.0
	networkx --- 2.6.3
	numpy --- 1.21.5
	pandas --- 1.3.5
	python-louvain --- 0.16
	Scanpy --- 1.9.1
	scikit-learn --- 1.0.2
	scipy --- 1.7.3
	tqdm --- 4.64.0

### Installation
1. **Necessary Step:** Download from Github:
   	```
	git clone https://github.com/xuyp-csu/scCAD.git
	cd scCAD
 	```

2. **Recommended Step:** (Conda users, Conda version: 4.12.0) Create your environment and activate it:
	```
	conda create -n scCAD_env python=3.7
 	source activate scCAD_env
 	```
 	-Use `conda activate` instead of `source activate`, if required.
   
	-Building typically completes in about 30 seconds.

3. **Necessary Step:** Install dependencies with pip:

	```
	pip install -r requirements.txt
	```
 	-Installation typically requires around 3 to 5 minutes, depending on network conditions.
