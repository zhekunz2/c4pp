
parameters {
  real<lower=-1,upper=1> x; 
  real<lower = -sqrt(1 - square(x)), 
       upper =  sqrt(1 - square(x))> y; 
} 
model {
  
} 
