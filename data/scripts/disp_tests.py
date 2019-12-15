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

import pandas as pd
import numpy as np
from data.scripts.di_testing import DisparityTesting
import os
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 200)

if __name__ == '__main__':

    pg_names = ["black", "native_american", "asian", "hispanic", "female"]
    cg_names = ["white", "white", "white", "non_hispanic", "male"]
    pgcg_names = ["black", "native_american", "asian", "white", "hispanic", "non_hispanic", "female", "male"]
    # pg_names, cg_names, pgcg_names = ["black", "female"], ["white", "male"], ["black", "white", "female", "male"]

    file_name = 'test_hmda_with_preds.csv'
    predicted = "high_priced_mgbm_pred"
    model_name = "HMDA Monotonic GBM"
    label = "high_priced"
    outcome = "decision"
    pred_prob_for_outcome = 0.30
    higher_score_favorable = False

    data = pd.read_csv(os.path.join('./data/output/', file_name))
    if "native_american" in pg_names:
        data.rename(inplace=True, columns={'amind': 'native_american'})
    data[outcome] = np.where(data[predicted] >= pred_prob_for_outcome, 1, 0)
    data[outcome].value_counts(normalize=True)

    disparity_tests = DisparityTesting(pg_names=pg_names, cg_names=cg_names, pgcg_names=pgcg_names,
                                       higher_score_favorable=higher_score_favorable)

    cat_outcomes = disparity_tests.categorical_disparity_measures(data=data, label=label, outcome=outcome)
    cont_outcomes = disparity_tests.continuous_disparity_measures(data=data, predicted=predicted)
    disparity_measures = DisparityTesting.create_combined_output(cat_outcomes=cat_outcomes, cont_outcomes=cont_outcomes)

    plt.title("Adverse Impact Ratios for the " + model_name + " Model")
    plt.xlabel('Protected Classes')
    plt.legend(loc="upper left")
    plt.xticks(rotation=-90)
    plt.bar(x=disparity_measures.loc[disparity_measures["Adverse Impact Ratio"].notna(), "Class"],
            height=disparity_measures.loc[disparity_measures["Adverse Impact Ratio"].notna(), "Adverse Impact Ratio"],
            color='blue',
            )
    plt.show()
