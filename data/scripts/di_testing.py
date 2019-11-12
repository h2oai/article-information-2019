
import h2o
import pandas as pd
import numpy as np
from typing import Union


class DisparateImpactTesting(object):
    def __init__(self, lower_value_favorable: bool,
                 pg_names: Union[list, tuple] = ('black', 'hispanic' 'asian', 'female', 'older'),
                 cg_names: Union[list, tuple] = ('white', 'white', 'white', 'male', 'younger'),
                 pgcg_names: Union[list, tuple] = ('black', 'hispanic', 'asian', 'white',
                                                   'female', 'male', 'older', 'younger'),
                 ):
        """
        This class creates metrics used to test disparate impact.  See each method for detail on the
        particular calculations.

        :param lower_value_favorable: Boolean specifying whether a lower outcome value being tested
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
        self.pg_names = pg_names
        self.cg_names = cg_names
        self.lower_value_favorable = lower_value_favorable
        self.pgcg_names = pgcg_names

    def adverse_impact_ratio(self,
                             data: Union[pd.DataFrame, h2o.H2OFrame],  # Not checked with h2o data frame
                             label: str,
                             ) -> pd.DataFrame:
        """
        This calculates the Adverse Impact Ratio (AIR), a standard measure of disparate impact used in
        U.S. legal and regulatory settings.  It is defined as:
            AIR = (% Protected Class Chosen) / (% Control Group Chosen)

        :param data: Data containing the label and protected and control group information for each person

        :param label: The string name of the label

        :return: A table containing the AIR and other summary statistics for each group.
        """
        di_cols = ["class", "Control Class", "Total", "Total Favorable", 
                   "Percent Favorable", "Adverse Impact Ratio"]
        di_table = pd.DataFrame(np.full(shape=(len(self.pgcg_names), len(di_cols)), fill_value=np.nan),
                                index=self.pgcg_names, columns=di_cols)
        di_table["class"] = self.pgcg_names
        di_table.loc[di_table["class"].isin(self.pg_names), "Control Class"] = self.cg_names

        if self.lower_value_favorable:
            outcome = data[label].to_numpy().reshape(-1, 1)
        else:
            outcome = (1 - data[label]).to_numpy().reshape(-1, 1)

        di_table["Total"] = data[self.pgcg_names].sum(axis=0)
        di_table["Total Favorable"] = np.sum(outcome * data[self.pgcg_names])
        di_table["Percent Favorable"] = di_table["Total Favorable"] / di_table["Total"]
        for cgi, pgi in zip(self.pg_names, self.cg_names):
            di_table.loc[di_table["class"] == cgi, "Adverse Impact Ratio"] = \
                di_table.loc[di_table["class"] == cgi, "Percent Favorable"].to_numpy() / \
                di_table.loc[di_table["class"] == pgi, "Percent Favorable"].to_numpy()

        di_table.index = range(len(di_table))
        return di_table
