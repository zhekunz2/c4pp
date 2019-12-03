## Read the data
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/radon

# The R codes & data files should be saved in the same directory for
# the source command to work
library("R2OpenBUGS")
source("12.6_Group-level predictors.R") # where variables were defined
# close the Bugs window to proceed

## Multilevel model with no group-level predictors
print("M1")
M1 <- lmer (y ~ x + (1 | county))
display(M1)
summary(M1)