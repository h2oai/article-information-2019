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

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 200)

hmda = pd.read_csv('./data/output/test_hmda_with_preds.csv')

pg_names = ["black", "amind", "hispanic", "female"]
cg_names = ["white", "white", "non_hispanic", "male"]
pgcg_names = ["black", "amind", "white", "hispanic", "non_hispanic", "female", "male"]
predicted = "high_priced_mgbm_pred"
label = "high_priced"
outcome = "decision"
pred_prob_for_outcome = 0.30
higher_score_favorable = False

hmda[outcome] = np.where(hmda[predicted] >= pred_prob_for_outcome, 1, 0)
hmda[outcome].value_counts(normalize=True)

data = hmda[["Id", predicted, outcome, label] + pgcg_names]

disp_tests = DisparityTesting(pg_names=pg_names, cg_names=cg_names, pgcg_names=pgcg_names,
                              higher_score_favorable=higher_score_favorable)

cat_outcomes = disp_tests.categorical_disparity_measures(data=data, label=label, outcome=outcome)
print(cat_outcomes)
cont_outcomes = disp_tests.continuous_disparity_measures(data=data, predicted=predicted)



