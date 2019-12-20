# Responsible Machine Learning: Interpretable Models, Post-hoc Explanation, and Disparate Impact Testing

### Authors: Patrick Hall <sup>1,2</sup>, Navdeep Gill <sup>1</sup>, Kim Montgomery <sup>1</sup>, and Nicholas Schmidt <sup>3</sup>
### Affiliations: <sup>1</sup> H2O.ai; <sup>2</sup> George Washington University; <sup>3</sup> BLDS, LLC

### Abstract: 
This manuscript outlines a viable approach for training and evaluating machine learning (ML) systems for high-stakes, human-centered, or regulated applications using common Python programming tools. The accuracy and intrinsic interpretability of two types of constrained models, monotonic gradient boosting machines (MGBM) and explainable neural networks (XNN), a deep learning architecture well-suited for structured data, are assessed on simulated data with known feature importance and discrimination characteristics and on publicly available mortgage data. For maximum transparency and the potential generation of personalized adverse action notices, the constrained models are analyzed using post-hoc explanation techniques including plots of partial dependence (PD) and individual conditional expectation (ICE) and global and local Shapley feature importance. The constrained model predictions are also tested for disparate impact (DI) and other types of discrimination using adverse impact ratio (AIR), marginal effect (ME), standardized mean difference (SMD), and additional straightforward group fairness measures. By combining interpretable models, post-hoc explanation, and discrimination testing with accessible software tools, this text aims to present the art of the possible for important ML applications that require high accuracy and interpretability and minimal discrimination.

### Current Working Draft:

See [article-information-2019.pdf](article-information-2019.pdf).

### Current Python 3.6 Environment Setup for Linux and OSX: 

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

#### Data Summaries and Preprocessing
* For simulated data, see [article-information-2019-sim-data-training-results.ipynb](notebooks/article-information-2019-sim-data-training-results.ipynb) (WIP)
* For lending data, see [article-information-2019-loan-data-training-results.ipynb](notebooks/article-information-2019-loan-data-training-results.ipynb) (WIP)
* For lending data preprocessing, see [hmda_preprocessing.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/hmda_preprocessing.ipynb)
* For simulated data preprocessing, see [simulated_preprocessing.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/simulated_preprocessing.ipynb)

#### Modelling
* For GBM and MGBM model building on the lending dataset, see [mgbm_hmda.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/mgbm_hmda.ipynb)
* For GBM and MGBM model building on the simulated dataset, see [simulated_preprocessing.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/simulated_preprocessing.ipynb)
* For XNN model building on the lending dataset, see [xnn_notebook_hmda.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/xnn_notebook_hmda.ipynb)
* For XNN model building on the simulated dataset, see [xnn_notebook_simulated_data.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/xnn_notebook_simulated_data.ipynb)
* For ANN model building on the lending dataset, see [ann_hdma_code.py](https://github.com/h2oai/article-information-2019/blob/master/notebooks/scripts/ann_hdma_code.py)
* For ANN model building on the simulated dataset, see [ann_simulation_code.py](https://github.com/h2oai/article-information-2019/blob/master/notebooks/scripts/ann_simulation_code.py)

#### Model Performance and Interpretation
* For GBM and MGBM performance evaluation and interpretation on the lending dataset, see [perf_pdp_ice_shap_mgbm_hmda.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/perf_pdp_ice_shap_mgbm_hmda.ipynb)
* For GBM and MGBM performance evaluation and interpretation on the simulated dataset, see [perf_pdp_ice_shap_mgbm_sim.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/perf_pdp_ice_shap_mgbm_sim.ipynb)
* For XNN performance evaluation and interpretation on the lending dataset, see [xnn_analysis_hmda_from_files.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/xnn_analysis_hmda_from_files.ipynb)
* For XNN performance evaluation and interpretation on the simulated dataset, see [xnn_analysis_simulation_from_files.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/xnn_analysis_simulation_from_files.ipynb)
* For ANN performance evaluation and interpretation on the lending dataset, see [ann_analysis_hmda_from_files.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/ann_analysis_hmda_from_files.ipynb)
* For ANN performance evaluation and interpretation on the simulated dataset, see [ann_analysis_simulation_from_files.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/ann_analysis_simulation_from_files.ipynb)


