
"""
License
Copyright 2019 Navdeep Gill, Patrick Hall, Kim Montgomery, Nick Schmidt

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License
for the specific language governing permissions and limitations under the License.

DISCLAIMER: This notebook is not legal compliance advice.
"""

import numpy as np
import pandas as pd
from sklearn import datasets as ds
from scipy import stats
from data.scripts import di_testing


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


if __name__ == "__main__":

    n_corr_normal = 10  # do not change; variable naming is specific to 20
    n_obs = 100000

    # Each call of a random number generator will call a value from the list of random seeds.
    np.random.seed(42)
    rand_seed_list = list(np.unique(np.random.randint(0, 1000000, 100)))

    # Creates matrix of correlated normal variables, first by creating the
    # positive definite matrix, which serves as the variance-covariance matrix.
    var_covar_cor_normal = ds.make_spd_matrix(n_dim=n_corr_normal, random_state=rand_seed_list.pop())
    np.random.seed(rand_seed_list.pop())
    features_cor_normal = pd.DataFrame(np.random.multivariate_normal(mean=[0] * n_corr_normal,
                                                                     cov=var_covar_cor_normal,
                                                                     size=n_obs))
    features_cor_normal = (features_cor_normal - features_cor_normal.mean()) / features_cor_normal.std()
    print(f'\nEmpirical Correlation Matrix for Correlated Normal Variables:\n{features_cor_normal.corr()}')

    # Creates uniform variables based the CDF of the correlated normal variables.
    X = pd.DataFrame(stats.norm.cdf(features_cor_normal))
    X.columns = ["fried" + str(x) for x in range(1, 6)] + \
                ["binary1", "binary2", "cat1", "pc1", "pc2"]

    X["binary1"] = np.where(X["binary1"] >= 0.5, 1, 0)
    X["binary2"] = np.where(X["binary2"] >= 0.20, 1, 0)
    for cci, ccj in enumerate([0.2, 0.35, 0.6, 0.975, 1.0]):
        X["cat1"] = np.where(X["cat1"] <= ccj, cci + 2, X["cat1"])
    X["cat1"] = X["cat1"] - 2
    X["cat1"] = np.int8(X["cat1"])
    for vci in ["binary1", "binary2", "cat1"]:
        print(X[vci].value_counts(dropna=False, normalize=True))

    dum = pd.get_dummies(X["cat1"], drop_first=True, prefix="cat1")
    X = pd.concat([X, dum], sort=False, axis=1)

    # X["bin1_effect"] = np.where(X["binary1"] == 1, 1.0, 0)
    # X["bin2_effect"] = np.where(X["binary2"] == 1, -1.5, 0)
    # X["cat1_effect"] = np.select([X["cat1"] == x for x in range(5)],
    #                              [0.0, 2.0, -2.5, 1.25, -3.0])

    # Friedman + binaries & categoricals:
    names_weights = ["f1f2", "f3", "f4", "f5", "b1", "b2", "c1_1", "c1_2", "c1_3", "c1_4"]
    namesVarsUsed = ["fried1", "fried2", "fried3", "fried4", "fried5", "binary1", "binary2",
                     "cat1_1", "cat1_2", "cat1_3", "cat1_4"]

    friedman_weights = [10., 20., 10., 5.]
    binary_categorical_weights = [1.0, -1.5, 2.0, -2.5, 1.25, -3.0]
    coefficients = pd.Series(friedman_weights + binary_categorical_weights, index=names_weights)

    # Friedman-Only Variables:
    contrib = pd.DataFrame(np.full(shape=(len(X), len(names_weights)), fill_value=np.nan),
                           index=X.index, columns=names_weights)
    contrib["f1f2"] = coefficients["f1f2"] * np.sin(np.pi * X.fried1 * X.fried2)
    contrib["f3"] = coefficients["f3"] * (X.fried3 - 0.5) ** 2
    contrib["f4"] = coefficients["f4"] * X.fried4
    contrib["f5"] = coefficients["f5"] * X.fried5

    # Add Variables that Augment the Friedman Set Here:
    contrib["b1"] = coefficients["b1"] * X["binary1"]
    contrib["b2"] = coefficients["b2"] * X["binary2"]
    contrib["c1_1"] = coefficients["c1_1"] * X["cat1_1"]
    contrib["c1_2"] = coefficients["c1_2"] * X["cat1_2"]
    contrib["c1_3"] = coefficients["c1_3"] * X["cat1_3"]
    contrib["c1_4"] = coefficients["c1_4"] * X["cat1_4"]
    intercept = -1 * contrib.sum(axis=1).mean()
    contrib["intercept"] = intercept

    print(f'Contributions Statistics:\n{contrib.describe()}')

    X["latent_no_noise"] = contrib.sum(axis=1)
    X["latent_no_noise"].hist()
    print(X["latent_no_noise"].describe())

    X["intercept"] = intercept

    np.random.seed(rand_seed_list.pop())
    X["logistic_noise"] = np.random.logistic(scale=3, size=len(X))
    X["logistic_noise"] = X["logistic_noise"] - X["logistic_noise"].mean()
    X['latent_with_noise'] = X["latent_no_noise"] + X["logistic_noise"]
    X["outcome"] = pd.DataFrame(np.where(X["latent_with_noise"] > 0, 1, 0))

    print(f'Outcome Variable Frequency:\n{X["outcome"].value_counts(normalize=True)}')

    X["prot_class1"], X["prot_class2"] = 0, 0
    X.loc[(X["pc1"] <= 0.20) & (X["outcome"] == 0), "prot_class1"] = 1
    X.loc[(X["pc1"] <= 0.10) & (X["outcome"] == 1), "prot_class1"] = 1
    X["ctrl_class1"] = 1 - X["prot_class1"]
    X["prot_class1"].value_counts(normalize=True)
    print(f'\nPercent of People with Favorable Results by Protected Class 1 Status:\n'
          f'{X.groupby(by="prot_class1")["outcome"].mean()}')
    print(f'\nCross-tab of Protected Class 1 Status and Outcome:\n'
          f'{pd.crosstab(X["prot_class1"], X["outcome"], margins=True, dropna=False, normalize=False)}')

    X.loc[(X["pc2"] > 0.42) & (X["outcome"] == 0), "prot_class2"] = 1
    X.loc[(X["pc2"] > 0.58) & (X["outcome"] == 1), "prot_class2"] = 1
    X["ctrl_class2"] = 1 - X["prot_class2"]
    print(f'\nPercent of People with Favorable Results by Protected Class 2 Status:\n'
          f'{X.groupby(by="prot_class2")["outcome"].mean()}')
    print(f'\nCross-tab of Protected Class 2 Status and Outcome:\n'
          f'{pd.crosstab(X["prot_class2"], X["outcome"], margins=True, dropna=False, normalize=False)}')

    di_analysis = di_testing.DisparateImpactTesting(lower_value_favorable=False,
                                                    pg_names=["prot_class1", "prot_class2"],
                                                    cg_names=["ctrl_class1", "ctrl_class2"],
                                                    pgcg_names=["prot_class1", "ctrl_class1",
                                                                "prot_class2", "ctrl_class2"])
    print(f'\nResults of DI Analysis:\n{di_analysis.adverse_impact_ratio(data=X, label="outcome")}')

    X = X[['outcome'] + ["fried" + str(x) for x in range(1, 6)] + ["binary1", "binary2", "cat1", 'intercept',
                                                                   "prot_class1", "ctrl_class1", "prot_class2",
                                                                   "ctrl_class2", "latent_no_noise",
                                                                   "logistic_noise"]]
    print(X.head())

    X_train = X[:int(round(n_obs) * (4 / 5))].copy()
    np.random.seed(rand_seed_list.pop())
    X_train["fold"] = np.random.randint(low=1, high=6, size=len(X_train))
    X_test = X[int(round(n_obs) * (4 / 5)) + 1:].copy()

    X_train.to_csv("./data/output/X_train.csv")
    X_test.to_csv("./data/output/X_test.csv")
