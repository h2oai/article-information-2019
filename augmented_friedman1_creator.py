
import numpy as np
import pandas as pd
from sklearn import datasets as ds
from scipy import stats
import os
import datetime


if __name__ == "__main__":

    nCorrNormal = 20
    nNormal = 20
    nObs = 100000
    intercept = -13.8955411
    getsOfferPct = 0.5
    pctFlippedByNoise = 0.20

    todaysDate = datetime.date.today().strftime("%Y%m%d")

    # Each call of a random number generator will call a value from the list
    # of random seeds.
    rnd = np.random.RandomState(61803)
    randSeed = list(np.unique(rnd.randint(0, 1000000, 100)))
    randIter = 0

    dataFolder = '/mnt/rdrive/Artificial Intelligence/article-information-2019/'
    os.chdir(dataFolder)
    
    # Creates matrix of uncorrelated normal variables
    rnd = np.random.RandomState(randSeed[randIter])
    varsNormal = pd.DataFrame(rnd.randn(nObs, nNormal), columns=['norm ' + str(x + 1) for x in range(nNormal)])

    # Creates matrix of correlated normal variables, first by creating the
    # positive definite matrix, which serves as the variance-covariance matrix.
    randIter = + 1
    rnd = np.random.RandomState(randSeed[randIter])
    corrNormalVarCovar = ds.make_spd_matrix(n_dim=nCorrNormal, random_state=randSeed[randIter])

    randIter = +1
    rnd = np.random.RandomState(randSeed[randIter])
    varsCorrNormal = rnd.multivariate_normal(mean=[0] * nCorrNormal,
                                             cov=corrNormalVarCovar,
                                             size=nObs)

    varsCorrNormal = pd.DataFrame(varsCorrNormal,
                                  columns=["corrNorm" + str(x + 1) for x in range(nCorrNormal)])
    varsCorrNormal = (varsCorrNormal - varsCorrNormal.mean()) / varsCorrNormal.std()
    cnCorr = varsCorrNormal.cov()
    print(f'\nEmpirical Correlation Matrix for Correlated Normal Variables:\n{varsCorrNormal.corr()}')

    # Creates uniform variables based the CDF of the correlated normal variables.
    varsUni = pd.DataFrame(stats.norm.cdf(varsCorrNormal))
    varsUni.columns = ["uni" + str(x + 1) for x in range(varsUni.shape[1])]

    '''
    Variables:
        Friedman's #1.....................uni1 - uni5
        Binary 1 & 2......................uni6 and 7
        Categorical.......................uni8
        Other Uncorrelated Normal.........normal1 - 2
        Other Correlated Normal 1 - 5.....corrNormal9 - 14
        Other Correlated Uniform 1 - 5....uni15 - 20
    '''

    X = pd.concat([pd.Series(np.full(shape=(nObs,), fill_value=intercept)),
                   varsUni.iloc[:, range(0, 9)],
                   varsNormal.iloc[:, [0, 2]],
                   varsCorrNormal.iloc[:, [x for x in range(9, 15)]],
                   varsUni.iloc[:, 15:]], axis=1)

    X.columns = ["intercept"] + ["fried" + str(x + 1) for x in range(5)] + \
                ["bin1", "bin2"] + \
                ["cat1"] + \
                ["othUnNorm" + str(x + 1) for x in range(2)] + \
                ["othCorrNorm" + str(x + 1) for x in range(6)] + \
                ["othCorrUni" + str(x + 1) for x in range(6)]

    X.bin1 = np.where(X.bin1 >= 0.5, 1, 0)
    X.bin2 = np.where(X.bin2 >= 0.20, 1, 0)
    for cci, ccj in enumerate([0.2, 0.35, 0.6, 0.975, 1.0]):
        X.cat1 = np.where(X.cat1 <= ccj, cci + 2, X.cat1)
    X.cat1 = X.cat1 - 2
    X.cat1 = np.int8(X.cat1)
    for vci in ["bin1", "bin2", "cat1"]:
        print(X[vci].value_counts(dropna=False, normalize=True))

    dum = pd.get_dummies(X.cat1, drop_first=True, prefix="cat1")
    X = pd.concat([X, dum], sort=False, axis=1)

    # Friedman + binaries & categoricals, no noise:

    namesCoef = ["intercept", "f1f2", "f3", "f4", "f5", "b1", "b2", "c1_1", "c1_2", "c1_3", "c1_4"]
    namesVarsUsed = ["fried1", "fried2", "fried3", "fried4", "fried5", "bin1", "bin2", "cat1_1", "cat1_2", "cat1_3",
                     "cat1_4"]

    friedmanCoef = [10, 20, 10, 5]
    binCatCoef = [1.0, -1.5, 2.0, -2.5, 1.25, -3.0]
    coefficients = pd.Series([intercept] + friedmanCoef + binCatCoef, index=namesCoef)

    # Friedman-Only Variables:
    contrib = pd.DataFrame(np.full(shape=(len(X), len(namesCoef)), fill_value=np.nan), index=X.index, columns=namesCoef)
    contrib.intercept = intercept
    contrib.f1f2 = coefficients["f1f2"] * np.sin(np.pi * X.fried1 * X.fried2)
    contrib.f3 = coefficients["f3"] * (X.fried3 - 0.5) ** 2
    contrib.f4 = coefficients["f4"] * X.fried4
    contrib.f5 = coefficients["f5"] * X.fried5

    # Friedman-Only Variable Test:
    y = contrib.sum(axis=1)
    yDoubleCheck = intercept + 10 * np.sin(
        np.pi * np.array(X.fried1).reshape(-1, 1) * np.array(X.fried2).reshape(-1, 1)) + \
                   20 * (np.array(X.fried3).reshape(-1, 1) - 0.5) ** 2 + 10 * np.array(X.fried4).reshape(-1,
                                                                                                         1) + 5 * np.array(
        X.fried5).reshape(-1, 1)
    print(f"Double-Check Friedman Calculation: {np.all(np.isclose(np.array(y).reshape(-1, 1), yDoubleCheck))}")
    del yDoubleCheck, y

    # Add Variables that Augment the Friedman Set Here:
    contrib.b1 = coefficients["b1"] * X.bin1
    contrib.b2 = coefficients["b2"] * X.bin2
    contrib.c1_1 = coefficients["c1_1"] * X.cat1_1
    contrib.c1_2 = coefficients["c1_2"] * X.cat1_2
    contrib.c1_3 = coefficients["c1_3"] * X.cat1_3
    contrib.c1_4 = coefficients["c1_4"] * X.cat1_4

    stats_contrib = contrib.describe()

    y = contrib.sum(axis=1)
    y.hist()
    print(y.describe())

    # A lower score is preferable:
    depVar = pd.DataFrame(np.where(y < 0, 1, 0),
                          columns=["defaultNoNoise"],
                          index=y.index)
    depVar["latentNoNoise"] = y

    randIter = +1
    rnd = np.random.RandomState(randSeed[randIter])
    depVar["error"] = rnd.logistic(scale=3, size=len(depVar))
    depVar['latent'] = depVar["latentNoNoise"] + depVar["error"]
    depVar["default"] = pd.DataFrame(np.where(depVar["latent"] < 0, 1, 0))

    withErrorDiff = depVar["latent"] - depVar["latentNoNoise"]
    print(withErrorDiff.hist())
    print(pd.crosstab(depVar["default"], depVar["defaultNoNoise"], normalize=True))

    X["othBin1"] = np.where(X["othCorrNorm1"] >= X["othCorrNorm1"].quantile(.5), 0, 1)
    X["othBin1"].value_counts()
    X["othBin2"] = np.where(X["othCorrNorm2"] >= X["othCorrNorm2"].quantile(.4), 0, 1)
    X["othBin2"].value_counts()
    X["othCat1"] = np.where(X["othCorrUni6"] <= 0.15, 1,
                            np.where(X["othCorrUni6"] <= 0.55, 2,
                                     np.where(X["othCorrUni6"] <= 0.75, 3,
                                              np.where(X["othCorrUni6"] <= 0.975, 4, 0))))
    X["othCat2"] = np.where(X["othCorrUni5"] <= 0.15, 1,
                            np.where(X["othCorrUni5"] <= 0.55, 2,
                                     np.where(X["othCorrUni5"] <= 0.75, 3,
                                              np.where(X["othCorrUni5"] <= 0.975, 4, 0))))

    droppers = [x for x in X.columns if str.startswith(x, "othCorrNorm")
                or str.startswith(x, "cat1_")
                or str.startswith(x, "othUnNorm")]
    X.drop(inplace=True, columns=droppers + ["othCorrUni5", "othCorrUni6"])

    # Full true X-Y data:
    yX = pd.merge(depVar, X, left_index=True, right_index=True, how='inner')
    sample_yX = yX.sample(100)

    yX.to_csv(todaysDate + " Augmented Friedman Set.csv")



