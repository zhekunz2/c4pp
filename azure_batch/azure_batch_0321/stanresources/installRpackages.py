#!/usr/bin/python

from rpy2.robjects.packages import importr
from rpy2.robjects import r

utils = importr('utils')
utils.chooseCRANmirror(ind=1)
utils.install_packages('FNN')
utils.install_packages('statip')
