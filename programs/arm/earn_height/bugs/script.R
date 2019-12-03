## Read & clean the data
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/earnings
# attach.all <- function (x, overwrite = NA, name = "attach.all")  {
#     rem <- names(x) %in% ls(.GlobalEnv)
#     if (!any(rem)) overwrite <- FALSE
#     rem <- names(x)[rem]
#     if (is.na(overwrite)) {
#         question <- paste("The following objects in .GlobalEnv will mask\nobjects in the attached database:\n", paste(rem, collapse = ", "), "\nRemove these objects from .GlobalEnv?", sep = "")
#         if (interactive()) {
#             if (.Platform$OS.type == "windows")  overwrite <- "YES" == winDialog(type = "yesno",  question)
#             else overwrite <- 1 == menu(c("YES", "NO"), graphics = FALSE, title = question)
#         }
#         else overwrite <- FALSE
#     }
#     if (overwrite) remove(list = rem, envir = .GlobalEnv)
#     attach(x, name = name)
# }
library("R2WinBUGS")
library ("arm")
library("readstata13")
heights <- read.dta13 ("heights.dta")
attach.all (heights)

  # recode sex variable 
male <- 2 - sex

  # (for simplicity) remove cases with missing data
ok <- !is.na (earn+height+sex) & earn>0
heights.clean <- as.data.frame (cbind (earn, height, male)[ok,])
n <- nrow (heights.clean)
attach.all (heights.clean)
height.jitter.add <- runif (n, -.2, .2)

## Model fit
lm.earn <- lm (earn ~ height)
display (lm.earn)
sim.earn <- sim (lm.earn)
beta.hat <- coef(lm.earn)
summary(lm.earn)
## Figure 4.1 (left)
par (mar=c(6,6,4,2)+.1)
plot (height + height.jitter.add, earn, xlab="height", ylab="earnings", pch=20, mgp=c(4,2,0), yaxt="n", col="gray10",
     main="Fitted linear model")
axis (2, c(0,100000,200000), c("0","100000","200000"), mgp=c(4,1.1,0))
for (i in 1:20){
  curve (sim.earn$coef[i,1] + sim.earn$coef[i,2]*x, lwd=.5, col="gray", add=TRUE)}
curve (beta.hat[1] + beta.hat[2]*x, add=TRUE, col="red")

## Figure 4.1 (right) 
par (mar=c(6,6,4,2)+.1)
plot (height + height.jitter.add, earn, xlab="height", ylab="earnings", pch=20, mgp=c(4,2,0), yaxt="n", col="gray10",
     main="Fitted linear model",xlim=c(0,80),ylim=c(-200000,200000))
axis (2, c(-100000,0,100000), c("-100000","0","100000"), mgp=c(4,1.1,0))
for (i in 1:20){
  curve (sim.earn$coef[i,1] + sim.earn$coef[i,2]*x, lwd=.5, col="gray", add=TRUE)}
curve (beta.hat[1] + beta.hat[2]*x, add=TRUE, col="red")
