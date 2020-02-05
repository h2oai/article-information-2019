# Responsible Machine Learning: Interpretable Models, Post-hoc Explanation, and Disparate Impact Testing

### Authors: Patrick Hall <sup>1,2</sup>, Navdeep Gill <sup>1</sup>, Kim Montgomery <sup>1</sup>, and Nicholas Schmidt <sup>3</sup>
### Affiliations: <sup>1</sup> H2O.ai; <sup>2</sup> George Washington University; <sup>3</sup> BLDS, LLC

### Abstract: 
This manuscript outlines a viable approach for training and evaluating machine learning (ML) systems for high-stakes, human-centered, or regulated applications using common Python programming tools. The accuracy and intrinsic interpretability of two types of constrained models, monotonic gradient boosting machines (MGBM) and explainable neural networks (XNN), a deep learning architecture well-suited for structured data, are assessed on simulated data with known feature importance and discrimination characteristics and on publicly available mortgage data. For maximum transparency and the potential generation of personalized adverse action notices, the constrained models are analyzed using post-hoc explanation techniques including plots of partial dependence (PD) and individual conditional expectation (ICE) and global and local Shapley feature importance. The constrained model predictions are also tested for disparate impact (DI) and other types of discrimination using adverse impact ratio (AIR), marginal effect (ME), standardized mean difference (SMD), and additional straightforward group fairness measures. By combining interpretable models, post-hoc explanation, and discrimination testing with accessible software tools, this text aims to present the art of the possible for important ML applications that require high accuracy and interpretability and that mitigate discrimination risks.

### Current Working Draft:

See [article-information-2019.pdf](article-information-2019.pdf).

### Current Python 3.6 Environment Setup for Linux and OSX: 

```
$ pip install virtualenv
$ cd notebooks
$ virtualenv -p python3.6 env
$ source env/bin/activate
$ pip install -r ../requirements.txt
$ ipython kernel install --user --name=information-article # Set up Jupyter kernel based on virtualenv
$ jupyter notebook
```

### Current Results:

#### Datasets:
* For lending trainset before preprocessing, see [hmda_train.csv](https://github.com/h2oai/article-information-2019/blob/master/data/output/hmda_train.csv)
* For lending testset before preprocessing, see [hmda_test.csv](https://github.com/h2oai/article-information-2019/blob/master/data/output/hmda_test.csv)
* For lending trainset after preprocessing, see [hmda_train_processed.csv](https://github.com/h2oai/article-information-2019/blob/master/data/output/hmda_train_processed.csv)
* For lending testset after preprocessing, see [hmda_test_processed.csv](https://github.com/h2oai/article-information-2019/blob/master/data/output/hmda_test_processed.csv)
* For simulated trainset before preprocessing, see [simu_train.csv](https://github.com/h2oai/article-information-2019/blob/master/data/output/simu_train.csv)
* For simulated testset before preprocessing, see [simu_test.csv](https://github.com/h2oai/article-information-2019/blob/master/data/output/simu_test.csv)
* For simulated trainset after preprocessing, see [train_simulated_processed.csv](https://github.com/h2oai/article-information-2019/blob/master/data/output/train_simulated_processed.csv)
* For simulated testset after preprocessing, see [test_simulated_processed.csv](https://github.com/h2oai/article-information-2019/blob/master/data/output/test_simulated_processed.csv)

#### Data Summaries and Preprocessing
* For lending data preprocessing, see [hmda_preprocessing.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/hmda_preprocessing.ipynb)
* For simulated data preprocessing, see [simulated_preprocessing.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/simulated_preprocessing.ipynb)

#### Modelling
* For GBM and MGBM model building on the lending dataset, see [mgbm_hmda.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/mgbm_hmda.ipynb)
* For GBM and MGBM model building on the simulated dataset, see [simulated_preprocessing.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/simulated_preprocessing.ipynb)
* For XNN model building on the lending dataset, see [xnn_notebook_hmda.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/xnn_notebook_hmda.ipynb)
* For XNN model building on the simulated dataset, see [xnn_notebook_simulated_data.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/xnn_notebook_simulated_data.ipynb)
* For ANN model building on the lending dataset, see [hmda_ann.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/hmda_ann.ipynb)
* For ANN model building on the simulated dataset, see [simulation_ann.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/simulation_ann.ipynb)

#### Model Performance and Interpretation

##### GBM and MGBM
* For GBM and MGBM performance evaluation and interpretation on the lending dataset (Table 4, Figure 7, Figure 8, Figure 9), see [perf_pdp_ice_shap_mgbm_hmda.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/perf_pdp_ice_shap_mgbm_hmda.ipynb)
* For GBM and MGBM performance evaluation and interpretation on the simulated dataset (Table 1, Figure 1, Figure 2, Figure 3), see [perf_pdp_ice_shap_mgbm_sim.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/perf_pdp_ice_shap_mgbm_sim.ipynb)

##### XNN and ANN
* For XNN performance evaluation and interpretation on the lending dataset (Table 4, Figure 10, Figure 11, Figure 12), see [xnn_analysis_hmda_from_files.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/xnn_analysis_hmda_from_files.ipynb)
* For XNN performance evaluation and interpretation on the simulated dataset (Table 1, Figure 4, Figure 5, Figure 6), see [xnn_analysis_simulation_from_files.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/xnn_analysis_simulation_from_files.ipynb)
* For ANN performance evaluation and interpretation on the lending dataset (Table 1), see [ann_analysis_hmda_from_files.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/ann_analysis_hmda_from_files.ipynb)
* For ANN performance evaluation and interpretation on the simulated dataset (Table 1), see [ann_analysis_simulation_from_files.ipynb](https://github.com/h2oai/article-information-2019/blob/master/notebooks/ann_analysis_simulation_from_files.ipynb)

#### Discrimination Testing Results
* For discrimination testing analysis, see [disparity_measurement.py](https://github.com/h2oai/article-information-2019/blob/master/notebooks/scripts/disparity_measurement.py)
* For discrimination testing results (Table 2, Table 3, Table 5, Table 6), see [Disparity Tables for Paper.xlsx](https://github.com/h2oai/article-information-2019/blob/master/data/output/Disparity%20Tables%20for%20Paper.xlsx)


