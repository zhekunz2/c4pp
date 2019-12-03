#!/usr/bin/python

import os
import shutil
import re

outputBase = 'bugs' # output.1.txt, output.2.txt, etc.
input_file = open('bpa-code.txt', 'r')
count = 500

def txtfile(string):
    return re.findall(r'[\w\.-]+\.txt', string)

def chapter_parse(string):
    match = re.search(r'^# \d+\.\d\.[\d\.]*', string)
    return match.group()[2:] if match else None

dest = None
code = ""
file_copy = []
chapter = "0.0."
for line in input_file:
    count = count - 1
    chapter_new = chapter_parse(line)
    if not chapter_new:
        code += line.replace("bugs.directory = bugs.dir,", "").replace("debug = TRUE,", "").replace("debug =TRUE,", "")
        file_copy.extend(txtfile(line))
        if "<- bugs(" in line:
            var = line.split()[0].replace(".", "")
        if "working.directory = getwd()" in line:
            code += "coda.data <- as.matrix(read.bugs(\"CODAchain1.txt\"))\nparam_file <- c()\nfor (out_param in colnames(as.matrix(coda.data))) {{\n    param_file <- cbind(param_file, paste(gsub(\"\\\\.\", \"_\", out_param), \",[\", paste(coda.data[, out_param], collapse = \" \"), \"],\", out$summary[,\"Rhat\"][out_param]))\n}}\nwrite.table(param_file, file = \"bugs_{0}.out\", row.names=FALSE, col.nam=FALSE, sep=\"\\n\", quote = FALSE)\n\n".format(var)

    else:
        if code != "":
            try:
                os.mkdir("bugs_bpa/{0}_{1}".format(outputBase, chapter))
            except:
                pass
            dest = open("bugs_bpa/{0}_{1}/{0}_{1}R".format(outputBase, chapter), 'w')
            code = "library(R2OpenBUGS)\nlibrary(coda)" + code
            dest.write(code)
            code = ""
            dest.close()
            for f in file_copy:
                try:
                    shutil.copy2(f, "bugs_bpa/{0}_{1}/".format(outputBase, chapter))
                except:
                    pass
            file_copy = []
        chapter = chapter_new
    #if (count <= 0):
    #    break

