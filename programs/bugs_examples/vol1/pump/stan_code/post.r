library(coda) 
post <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#'); 

N <- 10;
poi <- post[, c("alpha", "beta", paste("theta.", 1:N, sep = ''))]   
summary(as.mcmc(poi)) 

# copied from JAGS 
"benchstats" <-
structure(c(0.0601627050740002, 0.100952696924200, 0.08932503773, 
0.116273870980000, 0.599003113000001, 0.6082596306, 0.894592907996003, 
0.876048990530005, 1.583160045, 1.9833574464, 0.7061573086, 0.94860922336, 
0.025231937224413, 0.07805500440382, 0.0378282229082668, 0.0304751902078451, 
0.314649498624114, 0.136960663769425, 0.73140082256967, 0.71030782476205, 
0.762548883078548, 0.424327039225723, 0.272109766290328, 0.548803924118205, 
0.000356833478277114, 0.00110386445838974, 0.000534971858773434, 
0.000430984273078343, 0.00444981588348116, 0.00193691628214342, 
0.0103435696280886, 0.0100452695923822, 0.0107840697242214, 0.00600089053754638, 
0.00384821321941954, 0.0077612595257154, 0.000364275961155716, 
0.00114230156859317, 0.000542330603585075, 0.000436139437203953, 
0.00451615813238448, 0.00188395836024800, 0.0112388833614326, 
0.0106720645433105, 0.0114205076290931, 0.0062359030992733, 0.00671822724764772, 
0.0128090520670023), .Dim = as.integer(c(12, 4)), .Dimnames = list(
    c("theta[1]", "theta[2]", "theta[3]", "theta[4]", "theta[5]", 
    "theta[6]", "theta[7]", "theta[8]", "theta[9]", "theta[10]", 
    "alpha", "beta"), c("Mean", "SD", "Naive SE", "Time-series SE"
    )))

print(benchstats) 
