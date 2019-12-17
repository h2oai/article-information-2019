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
from notebooks.scripts.disparity_measurement import DisparityTesting

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 200)

file_name = './data/output/test_hmda_with_preds.csv'

pg_names = ["black", "amind", "hispanic", "female"]
cg_names = ["white", "white", "non_hispanic", "male"]
pgcg_names = ["black", "amind", "white", "hispanic", "non_hispanic", "female", "male"]
# pg_names = ["black", "female"]
# cg_names = ["white", "male"]
# pgcg_names = ["black", "white", "female", "male"]

predicted = "high_priced_mgbm_pred"
predicted = "high_priced_gbm_pred"
label = "high_priced"
outcome = "decision"
probability_for_classification = 0.30
higher_score_favorable = False


data = pd.read_csv(file_name)
data[outcome] = np.where(data[predicted] >= probability_for_classification, 1, 0)
data[outcome].value_counts(normalize=True)

disp_tests = DisparityTesting(pg_names=pg_names, cg_names=cg_names, pgcg_names=pgcg_names,
                              higher_score_favorable=higher_score_favorable)

cat_outcomes = disp_tests.categorical_disparity_measures(data=data, label=label, outcome=outcome)
cont_outcomes = disp_tests.continuous_disparity_measures(data=data, predicted=predicted)
disparity_measures = DisparityTesting.create_combined_output(cat_outcomes=cat_outcomes, cont_outcomes=cont_outcomes)


