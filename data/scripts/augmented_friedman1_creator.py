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
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

    #####################################################
    # Test uncorrelated version
    # var_covar_cor_normal = np.identity(n=n_corr_normal)
    #####################################################

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
    coefficients = coefficients.divide(5)

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
    ############
    # intercept = -30
    ############
    contrib["intercept"] = intercept

    print(f'Contributions Statistics:\n{contrib.describe()}')

    X["latent_no_noise"] = contrib.sum(axis=1)
    X["latent_no_noise"].hist()
    print(X["latent_no_noise"].describe())

    X["intercept"] = intercept

    np.random.seed(rand_seed_list.pop())
    X["logistic_noise"] = np.random.logistic(scale=0.5, size=len(X))
    X["logistic_noise"] = (X["logistic_noise"] - X["logistic_noise"].mean())  # / X["logistic_noise"].std()
    print(f'\nLogistic Random Noise Distribution Characteristics:\n{X["logistic_noise"].agg(["mean", "std"])}')
    X['latent_with_noise'] = X["latent_no_noise"] + X["logistic_noise"]
    X["latent_with_noise"].describe()
    X["latent_with_noise"].hist(alpha=0.5)
    X["logistic_noise"].hist(alpha=0.5)
    X["outcome"] = pd.DataFrame(np.where(X["latent_with_noise"] > 0, 1, 0))

    X["outcome_no_noise"] = pd.DataFrame(np.where(X["latent_no_noise"] > 0, 1, 0))

    # percent_random_outcomes = 0.25
    # np.random.seed(rand_seed_list.pop())
    # rand_outcome_prep1 = np.random.rand(len(X)) <= percent_random_outcomes
    # np.random.seed(rand_seed_list.pop())
    # rand_outcome_prep2 = np.random.rand(len(X)) >= 0.50
    # X["outcome"] = np.where(rand_outcome_prep1, 1 - X["outcome_no_noise"], X["outcome_no_noise"])
    # X["logistic_noise"] = np.nan
    # X["latent_with_noise"] = np.nan

    test_latent_nn = (1 / (1 + np.exp(-X["latent_no_noise"])))
    # These are all of the places where the model should not be able to correctly ID the output:
    test_latent_nn.loc[(X["outcome"] == 0) & (test_latent_nn > 0.5)].hist(alpha=0.5)
    test_latent_nn.loc[(X["outcome"] == 1) & (test_latent_nn <= 0.5)].hist(alpha=0.5)
    # (1 / (1 + np.exp(-X["latent_with_noise"]))).hist(alpha=0.25)
    print(f'Outcome with Noise Frequency:\n{X["outcome"].value_counts(normalize=True, dropna=False)}')
    print(f'Outcome: Noise v. No-Noise Comparison:\n'
          f'{pd.crosstab(X["outcome"], X["outcome_no_noise"], normalize=True, margins=True, dropna=False)}')

    # Create protected/control class variables (DI will be present in both):
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

    X = X[['outcome', "outcome_no_noise"] +
          ["fried" + str(x) for x in range(1, 6)] +
          ["binary1", "binary2", "cat1", 'intercept', "prot_class1", "ctrl_class1", "prot_class2", "ctrl_class2",
           "latent_no_noise", "logistic_noise", "latent_with_noise"]] # , "random_outcome"]]
    print("\n", X.head())

    X_train, X_test = train_test_split(X, test_size=0.20, random_state=rand_seed_list.pop(),
                                       shuffle=True, stratify=X["outcome"])
    X_train, X_test = X_train.copy(), X_test.copy()
    X_train["fold"] = np.random.randint(low=1, high=6, size=len(X_train))
    print(X_train.shape, X_test.shape, f"\n{X_train['fold'].value_counts(normalize=True)}")

    X_train.to_csv("./data/output/simu_train.csv")
    X_test.to_csv("./data/output/simu_test.csv")

    rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    # label = "outcome_no_noise"
    label = "outcome"
    train = X_train.copy()
    train["cat1"] = train["cat1"].astype('object')
    train = pd.get_dummies(train)

    test = X_test.copy()  # [[label] + features].copy()
    test["cat1"] = test["cat1"].astype('object')
    test = pd.get_dummies(test)

    features = [x for x in train.columns if str.startswith(x, "cat1_") or
                str.startswith(x, "fried") or
                str.startswith(x, "binary")]
    rfc.fit(X=train[features], y=train[label])
    pred_test = pd.Series(rfc.predict_proba(X=test[features])[:, 1], index=test.index)
    pred_train = pd.Series(rfc.predict_proba(X=train[features])[:, 1], index=train.index)
    print(f'\nTrain AUC: {roc_auc_score(y_true=test[label], y_score=pred_test)}\n'
          f'Test AUC: {roc_auc_score(y_true=test[label], y_score=pred_test)}')
    print(f'Classification Report:\n'
          f'{classification_report(y_true=test[label], y_pred=(pred_test > 0.5).astype("int"))}')
    xTest = test.copy()
    xTest["pred_proba"] = pred_test
    xTest["pred_proba"].hist()
    xTest["pred"] = (pred_test > 0.5).astype("int")
    xTest = xTest.loc[(np.sign(xTest["latent_no_noise"]) != np.sign(xTest["latent_with_noise"]))]
    print(np.corrcoef(xTest["latent_no_noise"], xTest["pred"])[0, 1])
