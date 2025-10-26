## Introduction

This repository contains an implementation of a lookahead decision tree which is a modification
of the lookahead decision tree by Organ et al.. Their approach can be found 
at [their repository](https://github.com/koftezz/rolling-lookahead-DT) which is an implementation 
of their lookahead decision tree approach
["Rolling Lookahead Learning for Optimal Classification Trees"](https://doi.org/10.48550/arXiv.2304.10830).
This 
repository also contains an implementation of a lookahead random forest, which uses
the lookahead decision trees in an ensemble. It provides bootstrapping, random feature subset
selection as well as multithreading.
The classifiers are designed specifically for binary data but can work for both
binary & multi-class classification tasks. Tools for binarization are provided.

This project uses PuLP (Copyright (c) 2002-2005, Jean-Sebastien Roy
Modifications Copyright (c) 2007- Stuart Anthony Mitchell).
The full license text is available in licenses/pulp_LICENSE.txt.

This repository also contains several testing datasets from the 
UC Irvine Machine Learning Repository ([UCI Machine Learning Repository](https://archive.ics.uci.edu/))
as well as microbiome datasets to evaluate various machine learning classification
methods for colorectal cancer risk (CRC) stratification, specifically 
the metagenomic species abundance pseudo-counts provided by [Barbet et al. ](https://doi.org/10.57745/7IVO3E).
Furthermore, it still contains lots of auxiliary testing data and prototypes.

The modified lookahead decision tree class can be found in `tree_refactored/class_tree.py`
and the lookahead random forest class can be found in `forest/forest_refactored_tree/class_forest_refactored.py`.

Consult `./makeReady_dasets.ipynb` to get an idea how data can be preprocessed 
and binarized.

To understand how to use the lookahead classification tree and the lookahead random
forest consult `./refactored_tree_server_simple_data.py` and `./forest_refactored_simple_data.py`
respectively.


## Installation
`requirements.txt` specifies all packages, their version and dependencies.

```bash
git clone https://github.com/Morgall/LARF_BA.git
cd LARF_BA
pip install -r requirements.txt
```


