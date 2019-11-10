
import h2o
import pandas as pd
import numpy as np
from typing import Union


class DisparateImpactTesting(object):
    def __init__(self, lower_score_better: bool,
                 pg_names: Union[list, tuple] = ('black', 'hispanic' 'asian', 'female', 'older'),
                 cg_names: Union[list, tuple] = ('white', 'white', 'white', 'male', 'younger'),
                 pgcg_names: Union[list, tuple] = ( 'black', 'hispanic', 'asian', 'white',
                                                    'female', 'male', 'older', 'younger'),
                 ):
        self.pg_names = pg_names
        self.cg_names = cg_names
        self.lower_score_better = lower_score_better
        self.pgcg_names = pgcg_names

    def adverse_impact_ratio(self,
                             data: Union[pd.DataFrame, h2o.H2OFrame],
                             label: str,
                             ):
        di_cols = ["class", "ctrl_class", "total", "gets_offer", "pct", "air"]
        di_table = pd.DataFrame(np.full(shape=(len(self.pgcg_names), len(di_cols)), fill_value=np.nan),
                                index=self.pgcg_names, columns=di_cols)
        di_table["class"] = self.pgcg_names
        di_table.loc[di_table["class"].isin(self.pg_names), "ctrl_class"] = self.cg_names
        di_table["gets_offer"] = (np.sum(data[label].to_numpy().reshape(-1, 1) * data[self.pgcg_names]))
        di_table["total"] = data[self.pgcg_names].sum(axis=0)
        di_table["pct"] = di_table["gets_offer"] / di_table["total"]
        for cgi, pgi in zip(self.pg_names, self.cg_names):
            di_table.loc[di_table["class"] == cgi, "air"] = di_table.loc[di_table["class"] == cgi, "pct"].to_numpy() / \
                                                            di_table.loc[di_table["class"] == pgi, "pct"].to_numpy()
        di_table.index = range(len(di_table))
        return di_table
