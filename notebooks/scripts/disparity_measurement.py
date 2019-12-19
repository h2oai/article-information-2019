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
from typing import Union
from sklearn import metrics
from scipy import stats


class DisparityTesting(object):
    def __init__(self,
                 pg_names: Union[list, tuple],
                 cg_names: Union[list, tuple],
                 pgcg_names: Union[list, tuple],
                 higher_score_favorable: bool):
        self.pg_names = pg_names
        self.cg_names = cg_names
        self.pgcg_names = pgcg_names
        self.higher_score_favorable = higher_score_favorable
        """
        This class creates metrics used to test various metrics of fairness.  See each method for detail on the
        particular calculations.

        :param higher_score_favorable: Boolean specifying whether a higher outcome value being tested
        is considered favorable.  For example, a marketing offer would generally be associated with a score
        where a higher value is favorable.  For credit, generally a lower score is favorable, as the score
        is typically a measure of the probability of default.

        :param pg_names: list of names of the variables that indicate whether an individual is a member of a
        protected group.

        :param cg_names: list of names of the variables that indicate whether an individual is a member of a
        control groups. These must correspond to the protected group values in "pg_names."  If a control group is
        used for more than one comparison (e.g., Non-Hispanic Whites as compared to Blacks, Hispanics, or Asians),
        then that control group variable should be repeated for each instance of an associated protected group in
        "pg_names".  For example:
            pg_names = ("black", "hispanic", "asian)
            cg_names = ("white", "white", "white")

        :param pgcg_names: This is a unique list of the values contained in pg_names and cg_names.  It is only used
        as a way to make the output more readable: it specifies the order in which each group is displayed.
        """

    def categorical_disparity_measures(self, data: pd.DataFrame, label: str, outcome: str):
        """
        This method calculates various disparity measures commonly used to assess fairness or discrimination in
        processes that lead to categorical outcomes.

        @param data: This is the Pandas data frame that contains the protected and control group information for
        each observation being tested, along with the given categorical outcome and the true outcome
        @param label: name of the true outcome variable in the data frame
        @param outcome: name of the given categorical outcome in the data frame
        @return: Pandas data frame containing a number of disparity measures
        """
        res = pd.DataFrame({'class': self.pgcg_names}, index=self.pgcg_names)
        for pi, ci in zip(self.pg_names, self.cg_names):
            res.loc[res["class"] == pi, "control"] = ci

        for groupi in self.pgcg_names:
            data_cm = data[data[groupi].notna()]
            cm = metrics.confusion_matrix(y_true=data_cm[label], y_pred=data_cm[outcome], sample_weight=data_cm[groupi])
            tn, fp, fn, tp = cm.ravel()
            res.loc[res["class"] == groupi, "total"] = (fp + tp + fn + tn)
            res.loc[res["class"] == groupi, "selected"] = (tp + fp)
            res.loc[res["class"] == groupi, "true_positive"] = tp
            res.loc[res["class"] == groupi, "true_negative"] = tn
            res.loc[res["class"] == groupi, "false_positive"] = fp
            res.loc[res["class"] == groupi, "false_negative"] = fn
            res.loc[res["class"] == groupi, "false_positive_rate"] = fp / (fp + tn)
            res.loc[res["class"] == groupi, "false_negative_rate"] = fn / (fn + tp)
            res.loc[res["class"] == groupi, "accuracy"] = metrics.accuracy_score(y_true=data_cm[label],
                                                                                 y_pred=data_cm[outcome],
                                                                                 sample_weight=data_cm[groupi])
            res.loc[res["class"] == groupi, "accuracy"] = metrics.accuracy_score(y_true=data_cm[label],
                                                                                 y_pred=data_cm[outcome],
                                                                                 sample_weight=data_cm[groupi])

            res.loc[res["class"] == groupi, "precision"] = metrics.precision_score(y_true=data_cm[label],
                                                                                   y_pred=data_cm[outcome],
                                                                                   sample_weight=data_cm[groupi])
            res.loc[res["class"] == groupi, "recall"] = metrics.recall_score(y_true=data_cm[label],
                                                                             y_pred=data_cm[outcome],
                                                                             sample_weight=data_cm[groupi])

        res["favorable"] = res["selected"] if self.higher_score_favorable else res["total"] - res["selected"]
        res["percent_selected"] = res["selected"] / res["total"]
        res["percent_favorable"] = res["favorable"] / res["total"]

        res.loc[res["class"].isin(self.pg_names), "control_total"] = np.array(res["total"][self.cg_names])
        res.loc[res["class"].isin(self.pg_names), "control_false_positive_rate"] = \
            np.array(res["false_positive_rate"][self.cg_names])
        res.loc[res["class"].isin(self.pg_names), "control_false_negative_rate"] = \
            np.array(res["false_negative_rate"][self.cg_names])
        res.loc[res["class"].isin(self.pg_names), "control_selected"] = np.array(res["selected"][self.cg_names])
        res.loc[res["class"].isin(self.pg_names), "control_percent_favorable"] = \
            np.array(res["percent_favorable"][self.cg_names])

        res["marginal_effects"] = res["control_percent_favorable"] - res["percent_favorable"]
        res["adverse_impact_ratio"] = res["percent_favorable"] / res["control_percent_favorable"]
        res["relative_false_positive_rate"] = res["false_positive_rate"] / res["control_false_positive_rate"]
        res["relative_false_negative_rate"] = res["false_negative_rate"] / res["control_false_negative_rate"]

        for fishi in self.pg_names:
            fishers_values = stats.fisher_exact(np.array(
                res.loc[res["class"] == fishi, ["total", "selected",
                                                "control_total", "control_selected"]]).reshape(2, 2))
            res.loc[res["class"] == fishi, "fishers_exact"] = fishers_values[0]
            res.loc[res["class"] == fishi, "fishers_exact_p_value"] = fishers_values[1]
        return res

    def continuous_disparity_measures(self,
                                      data: pd.DataFrame,
                                      predicted: str):
        """
        This method calculates various disparity measures commonly used to assess fairness or discrimination in
        processes that lead to continuous outcomes (e.g., pricing of loans, or probability measures that are used
        in optimizations).

        @param data: This is the Pandas data frame that contains the protected and control group information for
        each observation being tested, along with the given categorical outcome and the true outcome
        @param predicted: name of the model's predicted outcome (typically, though not always, continuous) in the
        data frame.
        @return: Pandas data frame containing a number of disparity measures
        """
        res = pd.DataFrame({'class': self.pgcg_names}, index=self.pgcg_names)
        for pi, ci in zip(self.pg_names, self.cg_names):
            res.loc[res["class"] == pi, "control"] = ci

        score = np.array(data[predicted]).reshape(-1, 1)
        res["total"] = data[self.pgcg_names].sum(axis=0)
        res["average"] = (score * data[self.pgcg_names]).sum(axis=0) / data[self.pgcg_names].sum(axis=0)
        res.loc[res["class"].isin(self.pg_names), "control_average"] = np.array(res["average"][self.cg_names])
        res["average_difference"] = res["average"] - res["control_average"]
        res["standard_deviation"] = score.std()
        res["standardized_mean_difference"] = res["average_difference"] / res["standard_deviation"]

        for pgi, cgi in zip(self.pg_names, self.cg_names):
            t_test = stats.ttest_ind(data.loc[data[pgi] == 1, predicted], data.loc[data[cgi] == 1, predicted])
            res.loc[res["class"] == pgi, "t_statistic"] = t_test[0]
            res.loc[res["class"] == pgi, "t_statistic_p_value"] = t_test[1]
        return res
    
    @staticmethod
    def create_combined_output(cat_outcomes: pd.DataFrame,
                               cont_outcomes: pd.DataFrame,
                               cat_vars: Union[list, tuple] = ('class', 'control', 'total', 'false_positive_rate',
                                                               'relative_false_positive_rate',  'false_negative_rate',
                                                               'relative_false_negative_rate', 'accuracy',
                                                               'marginal_effects', 'adverse_impact_ratio', 
                                                               'fishers_exact',
                                                               'fishers_exact_p_value'),
                               cont_vars: Union[list, tuple] = ('class', 'control', 'standardized_mean_difference', 
                                                                't_statistic', 't_statistic_p_value'),
                               ):

        disp_outcomes = pd.merge(cat_outcomes[list(cat_vars)], cont_outcomes[list(cont_vars)], on=["class", "control"])
        disp_outcomes["class"] = disp_outcomes["class"].str.replace("_", "-").str.title()
        disp_outcomes["control"] = disp_outcomes["control"].str.replace("_", "-").str.title()
        disp_outcomes.loc[disp_outcomes["class"] == "Amind", "class"] = "Native American"

        disp_outcomes.columns = disp_outcomes.columns.str.title()
        for old, new in zip(["_", "P Value", "T Statistic"], [" ", "P-Value", "T-Statistic"]):
            disp_outcomes.columns = disp_outcomes.columns.str.replace(old, new)
        return disp_outcomes

