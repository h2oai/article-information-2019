
import numpy as np
import pandas as pd
from sklearn import datasets as ds
from scipy import stats


if __name__ == "__main__":

    n_corr_normal = 20
    n_normal = 20
    n_obs = 100000
    intercept = -13.8955411

    # Each call of a random number generator will call a value from the list
    # of random seeds.
    np.random.seed(61803)
    rand_seed_list = list(np.unique(np.random.randint(0, 1000000, 100)))

    # Creates matrix of uncorrelated normal variables
    features_normal = pd.DataFrame(np.random.randn(n_obs, n_normal),
                                   columns=['norm' + str(x + 1) for x in range(n_normal)])

    # Creates matrix of correlated normal variables, first by creating the
    # positive definite matrix, which serves as the variance-covariance matrix.
    var_covar_cor_normal = ds.make_spd_matrix(n_dim=n_corr_normal, random_state=rand_seed_list.pop())
    np.random.seed(rand_seed_list.pop())
    features_cor_normal = np.random.multivariate_normal(mean=[0] * n_corr_normal,
                                                        cov=var_covar_cor_normal,
                                                        size=n_obs)

    features_cor_normal = pd.DataFrame(features_cor_normal,
                                       columns=["cor_norm" + str(x + 1) for x in range(n_corr_normal)])
    features_cor_normal = (features_cor_normal - features_cor_normal.mean()) / features_cor_normal.std()
    # cnCorr = features_cor_normal.cov()
    print(f'\nEmpirical Correlation Matrix for Correlated Normal Variables:\n{features_cor_normal.corr()}')

    # Creates uniform variables based the CDF of the correlated normal variables.
    features_uni = pd.DataFrame(stats.norm.cdf(features_cor_normal))
    features_uni.columns = ["uni" + str(x + 1) for x in range(features_uni.shape[1])]

    '''
    Variables:
        Friedman's #1.....................uni1 - uni5
        Binary 1 & 2......................uni6 and 7
        Categorical.......................uni8
        Other Uncorrelated Normal.........normal1 - 2
        Other Correlated Normal 1 - 5.....cor_normal9 - 14
        Other Correlated Uniform 1 - 5....uni15 - 20
    '''

    X = pd.concat([pd.Series(np.full(shape=(n_obs,), fill_value=intercept)),
                   features_uni.iloc[:, range(0, 9)],
                   features_normal.iloc[:, [0, 2]],
                   features_cor_normal.iloc[:, [x for x in range(9, 15)]],
                   features_uni.iloc[:, 15:]], axis=1)

    X.columns = ["intercept"] + ["fried" + str(x + 1) for x in range(5)] + \
                ["binary1", "binary2"] + \
                ["cat1"] + \
                ["other_uncor_norm" + str(x + 1) for x in range(2)] + \
                ["other_cor_norm" + str(x + 1) for x in range(6)] + \
                ["other_cor_uni" + str(x + 1) for x in range(6)]

    X["binary1"] = np.where(X["binary1"] >= 0.5, 1, 0)
    X["binary2"] = np.where(X["binary2"] >= 0.20, 1, 0)
    for cci, ccj in enumerate([0.2, 0.35, 0.6, 0.975, 1.0]):
        X["cat1"] = np.where(X["cat1"] <= ccj, cci + 2, X["cat1"])
    X["cat1"] = X["cat1"] - 2
    X["cat1"] = np.int8(X["cat1"])
    for vci in ["binary1", "binary2", "cat1"]:
        print(X[vci].value_counts(dropna=False, normalize=True))

    dum = pd.get_dummies(X["cat1"], drop_first=False, prefix="cat1")
    X = pd.concat([X, dum], sort=False, axis=1)

    # Friedman + binaries & categoricals, no noise:
    names_weights = ["intercept", "f1f2", "f3", "f4", "f5", "b1", "b2", "c1_1", "c1_2", "c1_3", "c1_4"]
    namesVarsUsed = ["fried1", "fried2", "fried3", "fried4", "fried5", "binary1", "binary2", "cat1_1", "cat1_2", "cat1_3",
                     "cat1_4"]

    friedman_weights = [10, 20, 10, 5]
    binary_categorical_weights = [1.0, -1.5, 2.0, -2.5, 1.25, -3.0]
    coefficients = pd.Series([intercept] + friedman_weights + binary_categorical_weights, index=names_weights)

    # Friedman-Only Variables:
    contrib = pd.DataFrame(np.full(shape=(len(X), len(names_weights)), fill_value=np.nan),
                           index=X.index, columns=names_weights)
    contrib["intercept"] = intercept
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

    stats_contrib = contrib.describe()

    friedman_latent = contrib.sum(axis=1)
    friedman_latent.hist()
    print(friedman_latent.describe())

    # A lower score is preferable:
    label_set = pd.DataFrame(np.where(friedman_latent < 0, 1, 0),
                             columns=["label_no_noise"],
                             index=friedman_latent.index)
    label_set["latent_no_noise"] = friedman_latent

    np.random.seed(rand_seed_list.pop())
    label_set["logistic_noise"] = np.random.logistic(scale=3, size=len(label_set))
    label_set['latent_with_noise'] = label_set["latent_no_noise"] + label_set["logistic_noise"]
    label_set["label_with_noise"] = pd.DataFrame(np.where(label_set["latent_with_noise"] < 0, 1, 0))

    print(pd.crosstab(label_set["label_with_noise"], label_set["label_no_noise"], normalize=True, margins=True))

    X["other_binary1"] = np.where(X["other_cor_norm1"] >= X["other_cor_norm1"].quantile(.5), 0, 1)
    X["other_binary1"].value_counts()
    X["other_binary2"] = np.where(X["other_cor_norm2"] >= X["other_cor_norm2"].quantile(.4), 0, 1)
    X["other_binary2"].value_counts()
    X["other_cat1"] = np.where(X["other_cor_uni6"] <= 0.15, 1,
                               np.where(X["other_cor_uni6"] <= 0.55, 2,
                                        np.where(X["other_cor_uni6"] <= 0.75, 3,
                                                 np.where(X["other_cor_uni6"] <= 0.975, 4, 0))))
    X["other_cat2"] = np.where(X["other_cor_uni5"] <= 0.15, 1,
                            np.where(X["other_cor_uni5"] <= 0.55, 2,
                                     np.where(X["other_cor_uni5"] <= 0.75, 3,
                                              np.where(X["other_cor_uni5"] <= 0.975, 4, 0))))

    droppers = [x for x in X.columns if str.startswith(x, "other_cor_norm")
                or str.startswith(x, "cat1_")
                or str.startswith(x, "other_uncor_norm")]
    X.drop(inplace=True, columns=droppers + ["other_cor_uni5", "other_cor_uni6"])

    simulated = pd.merge(label_set, X, left_index=True, right_index=True, how='inner')

    simu_train = simulated[:int(round(n_obs) * (4 / 5))].copy()
    np.random.seed(rand_seed_list.pop())
    simu_train["fold"] = np.random.randint(low=1, high=6, size=len(simu_train))
    simu_test = simulated[int(round(n_obs) * (4 / 5)) + 1:].copy()

    simu_train.to_csv("./data/output/simu_train.csv")
    simu_test.to_csv("./data/output/simu_test.csv")
