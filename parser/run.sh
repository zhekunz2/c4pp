#!/bin/bash
if [ ! -e antlr-4.7.1-complete.jar ]; then
    wget http://www.antlr.org/download/antlr-4.7.1-complete.jar
fi
antlr4='java -Xmx500M -cp ".:./antlr-4.7.1-complete.jar:$CLASSPATH" org.antlr.v4.Tool'
grun='java org.antlr.v4.gui.TestRig'
#$antlr4 -package "main.java.tool.parser" -visitor Stan.g4
$antlr4 -package "tool.parser"  -Dlanguage=Python2 -visitor Stan.g4
#javac Stan*.java
#$grun Stan program -tree $1
