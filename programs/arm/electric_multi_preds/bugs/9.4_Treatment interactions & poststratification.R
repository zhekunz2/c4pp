## Read the data
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/electric.company

# The R codes & data files should be saved in the same directory for
# the source command to work
library("R2WinBUGS")
source("9.3_Randomized experiments.R") # where data was cleaned

## Treatment interactions and poststratification

 # model with only treat. indicator



lm.333 <- lm (post.test ~ treatment + pre.test)
display (lm.333) 


