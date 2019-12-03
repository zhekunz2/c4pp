#!/usr/bin/env python

import pandas as pd
import sys

class Stan_CSV_1000:
    def __init__(self, csv_file):
        csv_df = pd.read_csv(csv_file,comment='#')
        csv_df.drop([xx for xx in list(csv_df) if not "__" in xx], axis=1, inplace=True)
        #csv_df = csv_df.loc[:, csv_df.mean().apply(np.isfinite)]
        cols = ["lp__","accept_stat__","stepsize__","treedepth__", "n_leapfrog__","divergent__","energy__"]
        csv_df = csv_df[cols]
        self.csv_df = csv_df[1000:]

    def get_runtime_info(self, iters=100):
        print(str(iters) + "," + str([ii for ii in self.csv_df[:iters].mean()])[1:-1])

    def get_all_runtime_info(self):
        for ii in range(10,101,10):
            self.get_runtime_info(ii)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sample_file = sys.argv[1]
    else:
        sample_file = "output_1000_1.gz"
    stan_csv = Stan_CSV_1000(sample_file)
    stan_csv.get_all_runtime_info()
