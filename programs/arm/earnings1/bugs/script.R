## Read the data
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/earnings

# The R codes & data files should be saved in the same directory for
# the source command to work
library("R2WinBUGS")
library("arm")
library("readstata13")
heights <- read.dta13 ("heights.dta")
attach.all (heights)
male <- 2 - sex
ok <- !is.na (earn+height+male)
heights.clean <- as.data.frame (cbind (earn, height, male)[ok,])

attach.all (heights.clean)

## Mixed discrete/continuous data

earn.pos <- ifelse (earn>0, 1, 0)
fit.1a <- glm (earn.pos ~ height + male, data=heights.clean, family=binomial(link="logit"))
display(fit.1a)
summary(fit.1a)
# log.earn <- log(earn)
# fit.1b <- lm (log.earn ~ height + male,  data=heights.clean, subset=earn>0)
# display(fit.1b)

