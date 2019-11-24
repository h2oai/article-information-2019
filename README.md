# Responsible Machine Learning: Interpretable Models, Post-hoc Explanation, and Disparate Impact Testing

### Authors: Patrick Hall <sup>1,2</sup>, Navdeep Gill <sup>1</sup>, Kim Montgomery <sup>1</sup>, and Nicholas Schmidt <sup>3</sup>
### Affiliations: <sup>1</sup> H2O.ai; <sup>2</sup> George Washington University; <sup>3</sup> BLDS, LLC

### Abstract: 
This text outlines a viable approach for training and evaluating complex machine learning systems for high-stakes, human-centered, or regulated applications using common Python programming tools. The accuracy and intrinsic interpretability of two types of constrained models, monotonic gradient boosting machines (M-GBM) and explainable neural networks (XNN), a deep learning architecture well-suited for structured data, are assessed on simulated datasets with known feature importance and sociological bias characteristics and on realistic, publicly available example datasets. For maximum transparency and the potential generation of personalized adverse action notices, the constrained models are analyzed using post-hoc explanation techniques including plots of individual conditional expectation (ICE) and global and local gradient-based or Shapley feature importance. The constrained model predictions are also tested for disparate impact and other types of sociological bias using straightforward group fairness measures. By combining innovations in interpretable models, post-hoc explanation, and bias testing with accessible software tools, this text aims to provide a template workflow for important machine learning applications that require high accuracy and interpretability and low disparate impact.

### Current Working Draft:

See [article-information-2019.pdf](article-information-2019.pdf).

### Current Python 3.6 Environment Setup: 

```
$ pip install virtualenv
$ cd notebooks
$ virtualenv -p python3.6 env
$ source env/bin/activate
$ pip install -r requirements.txt
$ ipython kernel install --user --name=information-article # Set up Jupyter kernel based on virtualenv
$ jupyter notebook
```

### Current Results:

* For simulated data, see [article-information-2019-sim-data-training-results.ipynb](notebooks/article-information-2019-sim-data-training-results.ipynb)
* For lending data, see [article-information-2019-loan-data-training-results.ipynb](notebooks/article-information-2019-loan-data-training-results.ipynb)
