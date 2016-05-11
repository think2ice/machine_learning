####################################################################
# Machine Learning - MIRI Master
# Lluís A. Belanche

# LAB 10: Kernels and kernel methods (Part 2)
# version of May 2016
####################################################################
setwd("/Users/manel/Documents/Universidad/MIRI/Q1B/ML/my_labs/machine_learning.git/lab9")

####################################################################
## Exercise 1: Playing with Kernel PCA with real data
####################################################################

## We want to perform a principal components analysis on the USArrests dataset, which
## contains Lawyers's ratings of 43 state judges in the US Superior Court.

## see USJudgeRatings {datasets}


summary(USJudgeRatings)
USJudgeRatings

## Have a look at the data; it seems that most ratings are highly correlated

require(graphics)
pairs(USJudgeRatings, panel = panel.smooth, main = "US Judge Ratings data")

require(psych)
describe (USJudgeRatings)

## the variances (sd²) of the variables do not vary much, but scaling is many times appropriate
## there is no need to do it beforehand, since prcomp() is able to handle it

pca.ratings <- prcomp(USJudgeRatings, scale = TRUE)
pca.ratings

summary(pca.ratings)
biplot(pca.ratings, cex=0.6)

## Rough interpretation: indeed all of the ratings are highly correlated, except CONT, which 
## is orthogonal to the rest; proximity of the judges in the 2D representation matches 
## proximity in the original space

## Do the following (in separate sections)

# 1. Do a PCA plot using the first 2 PCs, as you did in Part 1 for the Iris data
#    Add names to the plot; you may use:

plot(pca.ratings$x[,1], pca.ratings$x[,2])
text(pca$x[,1], pca$x[,2], rownames(USJudgeRatings), pos= 3, cex=0.6)

# 2. Do a kernel PCA plot using the first 2 PCs, as you did in Part 1 for the Iris data
#    You will have to modify the plotting function, for example to eliminate colors (there are no classes here)
#    Add state names to the plot; you may use:

plotting <-function (kernelfu, kerneln, iftext=FALSE)
{
  xpercent <- eig(kernelfu)[1]/sum(eig(kernelfu))*100
  ypercent <- eig(kernelfu)[2]/sum(eig(kernelfu))*100
  
  plot(rotated(kernelfu), col=as.integer(iris[,5]), 
       main=paste(paste("Kernel PCA (", kerneln, ")", format(xpercent+ypercent,digits=3)), "%"),
       xlab=paste("1st PC -", format(xpercent,digits=3), "%"),
       ylab=paste("2nd PC -", format(ypercent,digits=3), "%"))
  
  if (iftext) text(rotated(kernelfu)[,1], rotated(kernelfu)[,2], rownames(iris), pos= 3)
}
kpv <- kpca(~., data=USJudgeRatings, kernel="vanilladot", kpar=list(), features=2)
plotting (kpv, "linear")
text(rotated(kpv)[,1], rotated(kpv)[,2], rownames(USJudgeRatings), pos= 3, cex=0.6)

kpp <- kpca(~., data=USJudgeRatings, kernel="polydot", kpar=list(degree=3,offset=1), features=2)
plotting(kpp,"cubic")
text(rotated(kpp)[,1], rotated(kpp)[,2], rownames(USJudgeRatings), pos= 3, cex=0.6)
## 2. ------------------------------Polynomial Kernel (degree 3)-----------------
kpp <- kpca(~., data=USJudgeRatings, kernel="polydot", kpar=list(degree=3,offset=1), features=2)
plotting(kpp,"cubic")
text(rotated(kpp)[,1], rotated(kpp)[,2], rownames(USJudgeRatings), pos= 3, cex=0.6)

## 3. -------------------------------RBF Kernel-----------------------
kpc1 <- kpca(~., data=USJudgeRatings, kernel="rbfdot", kpar=list(sigma=0.6), features=2)
plotting(kpc1,"RBF - sigma 0.6")
text(rotated(kpc1)[,1], rotated(kpc1)[,2], rownames(USJudgeRatings), pos= 3, cex=0.6)

kpc2 <- kpca(~., data=USJudgeRatings, kernel="rbfdot", kpar=list(sigma=1), features=2)
plotting(kpc2,"RBF - sigma 1.0")
text(rotated(kpc2)[,1], rotated(kpc2)[,2], rownames(USJudgeRatings), pos= 3, cex=0.6)

## Note we could use our pre-defined 'rbf' kernel as well
kpc3 <- kpca(~., data=USJudgeRatings, kernel=rbf, features=2)
plotting(kpc3,"RBF - sigma 1.0")
text(rotated(kpc3)[,1], rotated(kpc3)[,2], rownames(USJudgeRatings), pos= 3, cex=0.6)

## The effect of sigma is a large one ...
kpc4 <- kpca(~., data=USJudgeRatings, kernel="rbfdot", kpar=list(sigma=10), features=2)
plotting(kpc4,"RBF - sigma 10")
text(rotated(kpc4)[,1], rotated(kpc4)[,2], rownames(USJudgeRatings), pos= 3, cex=0.6)

## The effect of sigma is a large one ...
kpc5 <- kpca(~., data=USJudgeRatings, kernel="rbfdot", kpar=list(sigma=0.01), features=2)
plotting(kpc5,"RBF - sigma 0.01")
text(rotated(kpc5)[,1], rotated(kpc5)[,2], rownames(USJudgeRatings), pos= 3, cex=0.6)

#    Use different kernels and kernel parameters
#    Compare the solutions one by one with the one given by standard PCA

#    It is a pity we do not have legal knowledge about these judges; we could then "judge" the results better
#    In particular, some clusters of judges emerge that could be identified (by k-means, for example)
#    Also, some trends emerge, as given by the new PCs, in which some judges are at opposite extremes

##############################################################################################
## Exercise 2: Playing with the SVM for classification with real data with a string kernel
##############################################################################################

## We are going to use a slightly-processed version of the famous
## Reuters news articles dataset.  All articles with no Topic
## annotations are dropped. The text of each article is converted to
## lowercase, whitespace is normalized to single-spaces.  Only the
## first term from the Topic annotation list is retained (some
## articles have several topics assigned).  

## The resulting dataset is a list of pairs (Topic, News content) We willl only use three topics
## for analysis: Crude Oil, Coffee and Grain-related news

## The resulting data frame contains 994 news items on crude oil,
## coffee and grain. The news text is the column "Content" and its
## category is the column "Topic". The goal is to create a classifier
## for the news articles.

## Note that we can directly read the compressed version (reuters.txt.gz). 
## There is no need to unpack the gz file; for local files R handles unpacking automagically

reuters <- read.table("reuters.txt.gz", header=T)

# We leave only three topics for analysis: Crude Oil, Coffee and Grain-related news
reuters <- reuters[reuters$Topic == "crude" | reuters$Topic == "grain" | reuters$Topic == "coffee",]

reuters$Content <- as.character(reuters$Content)    # R originally loads this as factor, so needs fixing
reuters$Topic <- factor(reuters$Topic)              # re-level the factor to have only three levels

levels(reuters$Topic)

length(reuters$Topic)

table(reuters$Topic)

## an example of a text about coffee
reuters[2,]

## an example of a text about grain
reuters[7,]

## an example of a text about crude oil
reuters[12,]

(N <- dim(reuters)[1])  # number of rows

# we shuffle the data first
set.seed(12)
reuters <- reuters[sample(1:N, N),]

# To deal with textual data we need to use a string kernel. Several such kernels are implemented in the
# "stringdot" method of the kernlab package. We shall use the simplest one: the p-spectrum
# kernel. The feature map represents the string as a multiset of its substrings of length p

# Example, for p=2 we have

# phi("ababc") = ("ab" -> 2, "ba" -> 1, "bc" --> 1, other -> 0)

# we can define a normalized 3-spectrum kernel (p is length)
k <- stringdot("spectrum", length=3, normalized=T)

# Let's see some examples:

k("I did it my way", "I did it my way")

k("He did it his way", "I did it my way")

k("I did it my way", "She did it her way")

k("I did it my way", "Let's get our way out")


## Do the following (in separate sections)

# 1. Do a PCA plot using the first 2 PCs, as you did in Part 1 for the Iris data
# You can add colors for the three classes (very much like the Iris data)

pca <- prcomp(~Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data=iris)


# 2. Do a kernel PCA plot using the first 2 PCs, as you did in Part 1 for the Iris data
# You can add colors for the three classes

# We now split the data into learning (2/3) and test (1/3) parts
ntrain <- round(N*2/3)     # number of training examples
tindex <- sample(N,ntrain) # indices of training examples


# 3. Create a kernel matrix using 'k' as kernel, something like this:

   ## WARNING: this may take a couple of minutes
   K <- kernelMatrix(k, reuters$Content)
   dim(K)

# 4. Train a SVM using this kernel matrix in the training set, something like this:

  svm1.train <- ksvm (K[tindex,tindex],reuters$Topic[tindex], type="C-svc", kernel='matrix')

  # No, I did not get fool, I am not supplying the full solution, just hints ...

# 5. Compute the error of this model in the test part

# 6. Now that you have a baseline code, you are asked to improve it to obtain a reliable solution:
#       - perform 10X10CV in the training set
#       - use it to tune the cost (C) parameter
#       - use it to tune the p parameter in the p-spectrum kernel

#    So you have to fit several svms with different p-kernels and evaluate their cross-validation error
#    Note that the cross-validation error is cross(), have a look at the help for ksvm()!
#    Choose your best model by playing a little bit with the C and p parameters
#    Note that changing p implies to re-compute K

# 7. Refit your best solution (C,p) using the whole learning part (no cross-validation)
# 8. Use this model to predict the test part and give a final prediction error

# Your code starts here ...
