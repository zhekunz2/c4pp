#!/usr/bin/env Rscript
#usage: ./jsonify.R [inputfile] [outputfile]
library("jsonlite")
myfunc <- function(file,output) {
source(file, local=TRUE)
vars <- ls()
mylist <- list()

for(a in vars){
if(a != "file" && a != "output")
{
val <- get(a)
if(is.character(val))
next
if(is.matrix(val)){
	mylist[a] <- list(val)
}
else if(is.vector(val)){
     mylist[a] <- list(val)
}
else{
mylist[a] <- val
}
}
}

jj <- toJSON(mylist)
write(jj, output)
}

args = commandArgs(trailingOnly=TRUE)
myfunc(args[1], args[2])


