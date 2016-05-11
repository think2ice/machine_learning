####################################################################
# Machine Learning - MIRI Master
# Lluís A. Belanche

# LAB 9: Kernel methods: the SVM (Part 1)
# version of April 2016
####################################################################


set.seed(6046)


####################################################################
# 1. Modelling artificial 2D sinusoidal data for two-class problems
####################################################################


## the SVM is located in two different packages: one of them is 'e1071'
library(e1071)

## First we create a simple two-class data set:

N <- 200

make.sinusoidals <- function(m,noise=0.2) 
{
  x1 <- c(1:2*m)
  x2 <- c(1:2*m)
  
  for (i in 1:m) {
    x1[i] <- (i/m) * pi
    x2[i] <- sin(x1[i]) + rnorm(1,0,noise)
  }
  
  for (j in 1:m) {
    x1[m+j] <- (j/m + 1/2) * pi
    x2[m+j] <- cos(x1[m+j]) + rnorm(1,0,noise)
  }
  
  target <- as.factor(c(rep(+1,m),rep(-1,m)))
  
  return(data.frame(x1,x2,target))
}

## let's generate the data
dataset <- make.sinusoidals (N)

## and have a look at it
summary(dataset)

plot(dataset$x1,dataset$x2,col=dataset$target)

## Now we wish to fit and visualize different SVM models

## model 1: LINEAR kernel, C=1 (cost parameter)
(model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=1, kernel="linear", scale = FALSE))

## Now we are going to visualize what we have done; since we have artificial data, instead of creating a random test set, we can create a grid of points as test

source("plot-prediction.R")

## make sure you understand the following results (one by one and their differences)

## plot the data, the OSH with margins, the support vectors, ...
plot.prediction (model, "linear kernel, C=1")

## model 2: linear kernel, C=0.1 (cost parameter)
(model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=0.1, kernel="linear", scale = FALSE))

plot.prediction (model, "linear kernel, C=0.1")

## the margin is wider (lower VC dimension), the number of support vectors is larger (more violations of the margin)

## model 3: linear kernel, C=25 (cost parameter)
(model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=25, kernel="linear", scale = FALSE))

plot.prediction (model, "linear kernel, C=25")

## the margin is narrower (higher VC dimension), number of support vectors is smaller (less violations of the margin)

## Let's put it together, for 6 values of C:

par(mfrow=c(2,3))

for (C in 10^seq(-2,3))
{
  model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=C, kernel="linear", scale = FALSE)
  plot.prediction (model, paste ("linear kernel (C=", C, ") ", model$tot.nSV, " Support Vectors", sep=""))
}


## Now we move to a QUADRATIC kernel (polynomial of degree 2); the kernel has the form:
## k(x,y) = (<x,y> + coef0)^degree

## quadratic kernel, C=1 (cost parameter)
(model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=1, kernel="polynomial", degree=2, coef0=1, scale = FALSE))

par(mfrow=c(1,1))

plot.prediction (model, "quadratic kernel, C=1")

## notice that neither the OSH or the margins are linear (they are quadratic); they are linear in the feature space
## in the previous linear kernel, both spaces coincide

## Let's put it together directly, for 6 values of C:

par(mfrow=c(2,3))

for (C in 10^seq(-2,3))
{
  model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=C, kernel="polynomial", degree=2, coef0=1, scale = FALSE)
  plot.prediction (model, paste ("quadratic kernel (C=", C, ") ", model$tot.nSV, " Support Vectors", sep=""))
}

## Now we move to a CUBIC kernel (polynomial of degree 3); the kernel has the form:
## k(x,y) = (<x,y> + coef0)^degree

## cubic kernel, C=1 (cost parameter)
(model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=1, kernel="polynomial", degree=3, coef0=1, scale = FALSE))

par(mfrow=c(1,1))

plot.prediction (model, "cubic kernel, C=1")

## notice that neither the OSH or the margins are linear (they are now cubic); they are linear in the feature space
## this choice seems much better, given the structure of the classes

## Let's put it together directly, for 6 values of C:

par(mfrow=c(2,3))

for (C in 10^seq(-2,3))
{
  model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=C, kernel="polynomial", degree=3, coef0=1, scale = FALSE)
  plot.prediction (model, paste ("cubic kernel (C=", C, ") ", model$tot.nSV, " Support Vectors", sep=""))
}

## Finally we use the Gaussian RBF kernel (polynomial of infinite degree; the kernel has the form:
## k(x,y) = exp(-gamma·||x - y||^2)

## RBF kernel, C=1 (cost parameter)
(model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=1, kernel="radial", scale = FALSE))

## the default value for gamma is 0.5

par(mfrow=c(1,1))

plot.prediction (model, "radial kernel, C=1, gamma=0.5")

## Let's put it together directly, for 6 values of C, holding gamma constant = 0.5:

par(mfrow=c(2,3))

for (C in 10^seq(-2,3))
{
  model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=C, kernel="radial", scale = FALSE)
  plot.prediction (model, paste ("RBF kernel (C=", C, ") ", model$tot.nSV, " Support Vectors", sep=""))
}

## Now for 8 values of gamma, holding C constant = 1:

par(mfrow=c(2,4))

for (g in 2^seq(-3,4))
{
  model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=1, kernel="radial", gamma=g, scale = FALSE)
  plot.prediction (model, paste ("RBF kernel (gamma=", g, ") ", model$tot.nSV, " Support Vectors", sep=""))
}

## In practice we should optimize both (C,gamma) at the same time

## How? Using cross-validation or trying to get "good" estimates analyzing the data

## Now we define a utility function for performing k-fold CV:

## the learning data is split into k equal sized parts
## every time, one part goes for validation and k-1 go for building the model (training)
## the final error is the mean prediction error in the validation parts
## Note k=N corresponds to LOOCV

## a typical choice is k=10
k <- 10 
folds <- sample(rep(1:k, length=N), N, replace=FALSE) 

valid.error <- rep(0,k)


## this function is not intended to be useful for general training purposes
## but it is useful for illustration
## in particular, it does not optimize the value of C (it requires it as parameter)

train.svm.kCV <- function (which.kernel, myC, kCV=10)
{
  for (i in 1:kCV) 
  {  
    train <- dataset[folds!=i,] # for building the model (training)
    valid <- dataset[folds==i,] # for prediction (validation)
    
    x_train <- train[,1:2]
    t_train <- train[,3]
    
    switch(which.kernel,
           linear={model <- svm(x_train, t_train, type="C-classification", cost=myC, kernel="linear", scale = FALSE)},
           poly.2={model <- svm(x_train, t_train, type="C-classification", cost=myC, kernel="polynomial", degree=2, coef0=1, scale = FALSE)},
           poly.3={model <- svm(x_train, t_train, type="C-classification", cost=myC, kernel="polynomial", degree=3, coef0=1, scale = FALSE)},
           RBF={model <- svm(x_train, t_train, type="C-classification", cost=myC, kernel="radial", scale = FALSE)},
           stop("Enter one of 'linear', 'poly.2', 'poly.3', 'radial'"))
    
    x_valid <- valid[,1:2]
    pred <- predict(model,x_valid)
    t_true <- valid[,3]
    
    # compute validation error for part 'i'
    valid.error[i] <- sum(pred != t_true)/length(t_true)
  }
  # return average validation error
  100*sum(valid.error)/length(valid.error)
}

# Fit an SVM with linear kernel

C <- 1

(VA.error.linear <- train.svm.kCV ("linear", myC=C))

## The procedure is to choose the model with the lowest CV error and then refit it with the whole learning data,
## then use it to predict the test set; we will do this at the end

## Fit an SVM with quadratic kernel 

(VA.error.poly.2 <- train.svm.kCV ("poly.2", myC=C))

## ## Fit an SVM with cubic kernel

(VA.error.poly.3 <- train.svm.kCV ("poly.3", myC=C))

## we get a series of decreasing CV errors ...

## and finally an RBF Gaussian kernel 

(VA.error.RBF <- train.svm.kCV ("RBF", myC=C))

## Now in a real scenario we should choose the model with the lowest CV error
## which in this case is the RBF (we get a very low CV error because this problem is easy for a SVM)

## so we choose RBF and C=1 and refit the model in the whole training set (no CV)
model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=C, kernel="radial", scale = FALSE)

## and make it predict a test set:

## let's generate the test data
dataset.test <- make.sinusoidals (1000)

## and have a look at it
summary(dataset.test)

par(mfrow=c(1,1))
plot(dataset.test$x1,dataset.test$x2,col=dataset.test$target)

pred <- predict(model,dataset.test[,1:2])
t_true <- dataset.test[,3]

table(pred,t_true)

# compute testing error (in %)
(sum(pred != t_true)/length(t_true))

## In a real setting we should also optimize the value of C, again
## with CV; all this can be done very conveniently using tune() to do
## automatic grid-search

## Other packages provide with heuristic methods to estimate the gamma in the RBF kernel (see below)

####################################################################
# 2. Playing with the SVM for regression and 1D data
####################################################################

## Now we do regression; we have an extra parameter: the 'epsilon', which controls the width of the epsilon-insensitive tube (in feature space)

A <- 20

## a really nice-looking function
x <- seq(-A,A,by=0.11)
y <- sin(x)/x + rnorm(x,sd=0.03)

plot(x,y,type="l")

## With this choice of the 'epsilon', 'gamma' and C parameters, the SVM underfits the data (blue line):

model1 <- svm (x,y,epsilon=0.01)
lines(x,predict(model1,x),col="blue")

## With this choice of the 'epsilon', 'gamma' and C parameters, the SVM overfits the data (green line):

model2 <- svm (x,y,epsilon=0.01,gamma=200, C=100)
lines(x,predict(model2,x),col="green")

## With this choice of the 'epsilon', 'gamma' and C parameters, the SVM has a very decent fit (red line):
model3 <- svm (x,y,epsilon=0.01,gamma=10)
lines(x,predict(model3,x),col="red")

## The other nice package where the SVM is located is {kernlab}

library(kernlab)

## the ksvm() method in this package has some nice features, as creation of user-defined kernels (not seen in this course), built-in cross-validation (via the 'cross' parameter) and automatic estimation of the gamma parameter

## Now we'll do standard (ridge) regression and kernel (ridge) regression, as seen in class

# 1. standard (ridge) regression

d <- data.frame(x,y)

linreg.1 <- lm (d)

plot(x,y,type="l")

abline(linreg.1, col="yellow")

## the result is obviously terrible, because our function is far from linear

## suppose we use a quadratic polynomial now:

linreg.2 <- lm (y ~ x + I(x^2), d)

plot(x,y,type="l")

points(x, predict(linreg.2), col="red", type="l")

## and keep increasing the degree ... in the end we would certainly do polynomial regression (say, degree 6):

linreg.6 <- lm (y ~ poly(x,6), d)

points(x, predict(linreg.6), col="green", type="l")

## and keep increasing the degree ... 

linreg.11 <- lm (y ~ poly(x,11), d)

points(x, predict(linreg.11), col="blue", type="l")

## we get something now ... but wait: instead of extracting new features manually (the higher-order monomials), it is much better to use a kernel function: let's do kernel (ridge) regression with the RBF kernel

## the reason for using regularization is that with the RBF kernel we extract monomials fo infinite degrees, so we need to explicitly control the complexity more than ever

###########################
## kernel ridge regression

## Let's start by computing the Gaussian RBF kernel manually

N <- length(x)
sigma <- 1
kk <- tcrossprod(x)
dd <- diag(kk)

## note that 'crossprod' and 'tcrossprod' are simply matrix multiplications (i.e., dot products)
## see help(crossprod) for details

## crossprod is a function of two arguments x,y; if only one is given, the second is taken to be the same as the first

## this computes the RBF kernel matrix rather quickly
myRBF.kernel <- exp(sigma*(-matrix(dd,N,N)-t(matrix(dd,N,N))+2*kk))

dim(myRBF.kernel)

## the first 5 entries (note diagonal is always 1)
myRBF.kernel[1:5,1:5]

## this is a good moment to review the class notes about kernel (ridge) regression

lambda <- 0.01

ident.N <- diag(rep(1,N))

alphas <- solve(myRBF.kernel + lambda*ident.N)

alphas <- alphas %*% y

lines(x,myRBF.kernel %*% alphas,col="magenta")

## not bad, a little bit wiggly, but essentially OK. The important point is that we have converted a linear technique into a non-linear one, by introducing the kernel

## if we add more regularization:

lambda <- 1

alphas <- solve(myRBF.kernel + lambda*ident.N)

alphas <- alphas %*% y

plot(x,y,type="l")
lines(x,myRBF.kernel %*% alphas,col="red")

## that is it!

## The advantage of the SVM for regression over kernel (ridge) regression is that in the former we may get sparser models:

(model2$tot.nSV/N)

# this is the fraction of points that are support vectors in the SVM for regression

# certainly not a very sparse model; this is because the function we want to model is quite complex, so most
# data points are important; the kernel (ridge) regression uses 100% of the data always.


####################################################################
## 3. Solving a difficult two-class problem with gene data
####################################################################

## In genetics, a promoter is a region of DNA that facilitates the transcription of a particular gene. 
## Promoters are located near the genes they regulate.

## Promoter Gene Data: data sample that contains DNA sequences, classified into 'promoters' and 'non-promoters'.
## 106 observations, 2 classes (+ and -)
## The 57 explanatory variables describing the DNA sequence have 4 possible values, represented 
## as the nucleotide at each position:
##    [A] adenine, [C] cytosine, [G] guanine, [T] thymine.

## The goal is to develop a predictive model (a classifier)

## data reading

dd <- read.csv2("promotergene.csv")

p <- ncol(dd)
n <- nrow(dd)

summary(dd)

## Since the data is categorical, we perform a Multiple Correspondence Analysis
## (the analog of PCA or Principal Components Analysis for numerical data)

## you have to source the auxiliary file
source ("acm.r")

X <- dd[,2:p]

mc <- acm(X)

# this is the projection of our data in the first two factors

plot(mc$rs[,1],mc$rs[,2],col=dd[,1])

# Histogram of eigenvalues (we have n-1)

barplot(mc$vaps)

# estimation of the number of dimensions to keep:

i <- 1

while (mc$vaps[i] > mean(mc$vaps)) i <- i+1

(nd <- i-1)

## Create a new dataframe 'Psi2' for convenience

Psi2 <- as.data.frame(cbind(mc$rs[,1:nd], dd[,1]))

names(Psi2)[43] <- "Class"
Psi2[,43] <- as.factor(Psi2[,43])
attach(Psi2)

## split data into learn (2/3) and test (1/3) sets

set.seed(2)
index <- 1:n
learn <- sample(1:n, round(0.67*n))

##### Support vector machine

## The implementation in {kernlab} is a bit more flexible than the one
## in {e1071}: we have the 'cross' parameter, for cross-validation (default is 0)

# we start with a linear kernel

## note how we are going to specify LOOCV (recommended for small datasets only)
mi.svm <- ksvm (Class~.,data=Psi2[learn,],kernel='polydot',C=1,cross=length(learn))

# note Number of Support Vectors, Training error and Cross validation error

mi.svm

# choose quadratic kernel now

cuad <- polydot(degree = 2, scale = 1, offset = 1)

mi.svm <- ksvm (Class~.,data=Psi2[learn,],kernel=cuad,C=1,cross=length(learn))

# note Number of Support Vectors, Training error and Cross validation error

mi.svm

# choose now the RBF kernel with automatic adjustment of the variance

mi.svm <- ksvm (Class~.,data=Psi2[learn,],C=1,cross=length(learn))

# note Number of Support Vectors, Training error and Cross validation error

mi.svm

# idem but changing the cost parameter C

mi.svm <- ksvm (Class~.,data=Psi2[learn,],C=5,cross=length(learn))

# note Number of Support Vectors, Training error and Cross validation error

mi.svm

# choose this latter model and use it to predict the test set

svmpred <- predict (mi.svm, Psi2[-learn,-43])

tt <- table(dd[-learn,1],svmpred)

error_rate.test <- 100*(1-sum(diag(tt))/sum(tt))
error_rate.test

# gives a prediction error of 14.3%


####################################################################
## 4. One-class SVM for novelty detection 
####################################################################

# Let is implement a very basic 2D novelty detector 
# We are going to use the same data as in example 1 and {e1071}

plot(dataset$x1,dataset$x2,col=dataset$target)

## we set nu = 0.1; this means that everything that has a probability lower than 10% is 'novel/outlying'

detector <- svm(dataset[,1:2], dataset$target,type="one-classification",
                kernel="radial", nu=0.1, scale = FALSE)

## recall that nu is a lower bound on the fraction of Support Vectors:

detector$tot.nSV/nrow(dataset)

# let's create density data for the 'image' plot below
resol <- 200
rng <- apply(dataset[,1:2],2,range)
tx <- seq(rng[1,1],rng[2,1],length=resol)
ty <- seq(rng[1,2],rng[2,2],length=resol)
pnts <- matrix(nrow=length(tx)*length(ty),ncol=2)
k <- 1
for(j in 1:length(ty)){
  for(i in 1:length(tx)){
    pnts[k,] <- c(tx[i],ty[j])
    k <- k+1
  }
}

# SVM predictition for density plot
pred <- predict(detector,pnts, decision.values=TRUE)
z <- matrix(attr(pred,"decision.values"),nrow=length(tx),ncol=length(ty))

image (tx,ty,z,xlab="",ylab="",axes=FALSE,
      xlim=c(rng[1,1],rng[2,1]),ylim=c(rng[1,2],rng[2,2]),
      col = rainbow(1000, start=0, end=.25))

# this is the learned surface
contour(tx,ty,z,add=TRUE, drawlabels=TRUE, level=0, lwd=3)

# plot points from training set as black points
points(dataset[,1:2],col="black",cex=1)

# plot highlighted SV points
sv <- dataset[c(detector$index),]
points(sv,pch=13,col="black",cex=2)
