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


def disparity_tables(file_name,
                     pg_names,
                     cg_names,
                     pgcg_names,
                     predicted,
                     label,
                     probability_for_classification,
                     higher_score_favorable,
                     model_name,
                     ):
    data = pd.read_csv(file_name)
    data["classification_outcome"] = np.where(data[predicted] >= probability_for_classification, 1, 0)
    data["classification_outcome"].value_counts(normalize=True)

    disp_tests = DisparityTesting(pg_names=pg_names, cg_names=cg_names, pgcg_names=pgcg_names,
                                  higher_score_favorable=higher_score_favorable)

    cat_outcomes = disp_tests.categorical_disparity_measures(data=data, label=label, outcome="classification_outcome")
    cont_outcomes = disp_tests.continuous_disparity_measures(data=data, predicted=predicted)
    disparity_measures = DisparityTesting.create_combined_output(cat_outcomes=cat_outcomes, cont_outcomes=cont_outcomes)
    disparity_measures["Model Name"] = model_name
    disparity_measures = disparity_measures[["Model Name"] + [x for x in disparity_measures.columns if x != "Model Name"]]
    return disparity_measures


if __name__ == '__main__':

    hmda_static_params = {"pg_names": ["black", "female"],
                          "cg_names": ["white", "male"],
                          "pgcg_names": ["black", "white", "female", "male"],
                          # "pg_names": ["black", "amind", "hispanic", "female"],
                          # "cg_names": ["white", "white", "non_hispanic", "male"],
                          # "pgcg_names": ["black", "amind", "white", "hispanic", "non_hispanic", "female", "male"],
                          "higher_score_favorable": False,
                          "probability_for_classification": 0.20,
                          }

    hmda_mgbm = disparity_tables(**hmda_static_params,
                                 model_name="Monotonic GBM - HMDA Data",
                                 file_name='./data/output/test_hmda_with_preds.csv',
                                 predicted="high_priced_mgbm_pred",
                                 label="high_priced",
                                 )

    hmda_gbm = disparity_tables(**hmda_static_params,
                                model_name="Standard GBM - HMDA Data",
                                file_name='./data/output/test_hmda_with_preds.csv',
                                predicted="high_priced_gbm_pred",
                                label="high_priced",
                                )

    simu_static_params = {"pg_names": ["prot_class1", "prot_class2"],
                          "cg_names": ["ctrl_class1", "ctrl_class2"],
                          "pgcg_names": ["prot_class1", "ctrl_class1", "prot_class2", "ctrl_class2"],
                          "higher_score_favorable": True,
                          "probability_for_classification": 0.60,
                          }
    simu_mgbm = disparity_tables(**simu_static_params,
                                 model_name="Monotonic GBM - Simulated Data",
                                 file_name='./data/output/test_sim_with_preds.csv',
                                 predicted="outcome_mgbm_pred",
                                 label="outcome",
                                 )

    simu_gbm = disparity_tables(**simu_static_params,
                                model_name="Standard GBM - Simulated Data",
                                file_name='./data/output/test_sim_with_preds.csv',
                                predicted="outcome_gbm_pred",
                                label="outcome",
                                )

    disparity_results = pd.concat([hmda_mgbm, hmda_gbm,
                                   simu_mgbm, simu_gbm], axis=0)

    disparity_results.to_csv('./data/output/gbm_simu_and_hmda_disparity_results.csv')