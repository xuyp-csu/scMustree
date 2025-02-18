# scMustree
## Introduction
Inference and visualization of multi-scale cell tree for decoding functional diversity.

![alt text](https://github.com/xuyp-csu/scMustree/blob/main/scMustree_overview.png)

## Getting Started
## Hardware requirements
`scMustree` package requires only a standard computer with enough RAM to support the in-memory operations.

### OS Requirements
This package is supported for *Linux* and *macOS*. The package has been tested on the following systems:
+ Linux: Ubuntu 16.04.7
+ macOS: Monterey 12.6.3

### Prerequisites

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
