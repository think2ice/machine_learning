# Dataset available online at: 
# http://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data

# Description of the variables
#http://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.names

# Reading the dataset
yeast <- read.table(url("http://archive.ics.uci.edu/ml/machine-learning-databases/
                        yeast/yeast.data"),header = FALSE)
cols.names <- c("seq.name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc","class")
names(yeast) <- cols.names
# Basic analysis
dim(yeast)
summary(yeast)
str(yeast)
par(mfrow = c(3,3))
for (i in 2:(ncol(yeast)-1)) {
  hist(yeast[,i], main = names(yeast)[i])
}
barplot(table(yeast[,10]))
# Looking at the data density, perhaps a transformation could be applied to 
# variables erl and pox
summary(yeast[,6:7])
# pox is always almost 0
