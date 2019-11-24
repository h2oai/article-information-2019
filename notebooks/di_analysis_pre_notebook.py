
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
import os
import h2o
from h2o.estimators import H2OXGBoostEstimator


h2o.init()

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# Function to get Shapley contribs
def get_shapley(model, data, x):
    # calculate SHAP values using function predict_contributions
    contributions = model.predict_contributions(data[x])
    print("First 5 rows of contributions:\n")
    print(contributions.head(5))
    # convert the H2O Frame to use with shap's visualization functions
    contributions_matrix = contributions.as_data_frame().as_matrix()
    # shap values are calculated for all features
    shap_values = contributions_matrix[:, 0:len(x)]
    # expected values is the last returned column
    expected_value = contributions_matrix[:, len(x)-1].min()
    return shap_values, expected_value


if __name__ == '__main__':

    cur_directory = os.getcwd()
    if not str.endswith(cur_directory, "output"):
        os.chdir("./data/output")

    features = ['loan_amount', 'loan_to_value_ratio', 'no_intro_rate_period', 'intro_rate_period',
                'property_value', 'income', "term_360", "conforming", "debt_to_income_ratio"]
    label = "high_priced"
    fold_column = "cv_fold"
    pgcg_list = ['black', 'asian', 'amind', 'hipac', 'white',
                 'hispanic', 'non_hispanic',
                 'female', 'male',
                 'agegte62', 'agelt62']
    pg_list = ['black', 'asian', 'amind', 'hipac',
               'hispanic',
               'female',
               'agegte62']
    cg_list = ['white', 'white', 'white', 'white',
               'non_hispanic',
               'male',
               'agelt62']

    train = pd.read_csv('hmda_train.csv')
    test = pd.read_csv('hmda_test.csv')

    train_h2o = h2o.import_file('hmda_train.csv')
    test_h2o = h2o.import_file('hmda_test.csv')
    train_h2o[label] = train_h2o[label].asfactor()
    test_h2o[label] = test_h2o[label].asfactor()

    params = {'booster': "gbtree",
              'backend': 'gpu',
              'stopping_metric': 'auc',
              'distribution': 'bernoulli',
              'categorical_encoding': 'auto',
              }

    xgb = H2OXGBoostEstimator(**params)

    xgb.train(x=features, y=label, training_frame=train_h2o, validation_frame=test_h2o, fold_column=fold_column)
    print(xgb)
    print(xgb.gpu_id)

    pred_train = xgb.predict(train_h2o).as_data_frame().loc[:, ["predict", "p0"]]
    pred_train.index = train.index
    pred_train = pd.concat([train[[label] + pgcg_list], pred_train], axis=1)
    pred_train.loc[:, pgcg_list] = pred_train.loc[:, pgcg_list].fillna(value=0)
    # Because a lower score and an outcome of 0 is better (higher priced loans are unfavorable to the borrower):
    pred_train["true_favorable"] = 1 - pred_train[label]
    pred_train.rename(inplace=True, columns={'p0': 'prob_favorable',
                                             'predict': 'gets_unfavorable'})
    pred_train["gets_favorable"] = 1 - pred_train["gets_unfavorable"]
    pred_train.drop(inplace=True, columns=['label', "gets_unfavorable"])
    print(pred_train.columns)
    print(pred_train.head())

    favorable_treatment = np.array(pred_train["gets_favorable"]).reshape(-1, 1)
    score = np.array(pred_train["prob_favorable"]).reshape(-1, 1)
    gets_favorable = (favorable_treatment * pred_train[pgcg_list]).sum()
    gets_favorable.name = "Total Favorable"
    total = pred_train[pgcg_list].sum()
    total.name = "Total"
    percent_favorable = gets_favorable / total
    percent_favorable.name = "Percent Favorable"
    total_control_class = pd.Series(np.array(total[cg_list]),
                                    index=pg_list, name="Total Control")
    pct_favorable_control_class = pd.Series(np.array(percent_favorable[cg_list]),
                                            index=pg_list, name="Percent Favorable Control")
    di_table = pd.concat([total,
                          total_control_class,
                          gets_favorable,
                          percent_favorable,
                          pct_favorable_control_class,
                          ], axis=1, sort=False)
    di_table["Percent Difference"] = di_table["Percent Favorable Control"] - di_table["Percent Favorable Outcomes"]
    di_table["Adverse Impact Ratio"] =  di_table["Percent Favorable Outcomes"] / di_table["Percent Favorable Control"]
    print(di_table)

    pred_train["intercept"] = 1

    import statsmodels as sm
    from statsmodels.discrete.discrete_model import Logit
    from statsmodels.discrete.discrete_model import LogitResults

    race_logit = Logit(endog=pred_train["gets_favorable"],
                       exog=pred_train[pg_list[:4] + ["intercept"]],
                       )

    race_logit_fit = race_logit.fit(cov_type="HC0")
    race_logit_fit.summary()
    z = race_logit_fit.params / race_logit_fit.bse

    import statsmodels.api as sm
    from statsmodels import robust
    race_logit = sm.GLM(endog=pred_train["gets_favorable"],
                        exog=pred_train[pg_list[:4] + ["intercept"]],
                        family=sm.families.Binomial(),

                        )
    race_logit = sm.Logit(endog=pred_train["gets_favorable"],
                          exog=pred_train[pg_list[:4] + ["intercept"]],
                          )
    race_logit_fit = race_logit.fit(cov_type="HC3")
    print(race_logit_fit.summary())
    race_logit_fit.bse
    del race_logit_fit, race_logit

