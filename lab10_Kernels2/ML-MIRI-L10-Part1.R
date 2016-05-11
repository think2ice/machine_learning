####################################################################
# Machine Learning - MIRI Master
# Lluís A. Belanche

# LAB 10: Kernels and kernel methods (Part 1)
# version of May 2016
####################################################################

# The first part of this file is based on J.P. Vert's excelllent teaching material:

# Original file is available at
#   http://cbio.ensmp.fr/~jvert/svn/tutorials/practical/makekernel/makekernel.R
# The corresponding note is available at
#   http://cbio.ensmp.fr/~jvert/svn/tutorials/practical/makekernel/makekernel_notes.pdf

## We are going to use the {kernlab} package
## Personally, in order to get the maximum of this nice package, I recommend that you have a look
## at the manual http://cran.r-project.org/web/packages/kernlab/kernlab.pdf
## Do it when you have some spare time (not now)

set.seed(6046)


####################################################################
# 1. Modelling artificial 2D Gaussian data with hand-made kernels
####################################################################

## First we create a simple two-class data set:

N <- 200 # number of data points
d <- 2   # dimension
sigma <- 2  # variance of the distribution
meanpos <- 0 # centre of the distribution of positive examples
meanneg <- 3 # centre of the distribution of negative examples
npos <- round(N/2) # number of positive examples
nneg <- N-npos # number of negative examples

## Generate the positive and negative examples
xpos <- matrix(rnorm(npos*d,mean=meanpos,sd=sigma),npos,d)
xneg <- matrix(rnorm(nneg*d,mean=meanneg,sd=sigma),npos,d)
x <- rbind(xpos,xneg)

## Generate the class labels
t <- matrix(c(rep(1,npos),rep(-1,nneg)))

## Visualize the data
plot(x,col=ifelse(t>0,1,2))
legend("topleft",c('Pos','Neg'),col=seq(2),pch=1,text.col=seq(2))

## load the library
library(kernlab)

## Now let's train a SVM with the standard (built-in) RBF kernel
## see help(kernels) for definition of the RBF and other built-in kernels

## a) Let's start by computing the Gaussian RBF kernel manually

sigma <- 1
kk <- tcrossprod(x)
dd <- diag(kk)

## note that 'crossprod' and 'tcrossprod' are simply matrix multiplications (i.e., dot products)
## see help(crossprod) for details
## it is a function of two arguments x,y; if only one is given, the second is taken to be the same as the first

## This experssion computes the RBF kernel rather quickly
myRBF.kernel <- exp(sigma*(-matrix(dd,N,N)-t(matrix(dd,N,N))+2*kk))

dim(myRBF.kernel)

## the first 5 entries (note diagonal is always 1)
myRBF.kernel[1:5,1:5]

## Now we would like to train a SVM with our precomputed kernel

## We basically have two options in {kernlab}:

## either we explicitly convert myRBF.kernel to a 'kernelMatrix' object, and then ksvm() understands it:

svm1 <- ksvm (as.kernelMatrix(myRBF.kernel), t, type="C-svc")

## or we keep it as a regular matrix and we add the kernel='matrix' argument:

svm2 <- ksvm(myRBF.kernel,t, type="C-svc", kernel='matrix')

## b) This is how we would do it "the classical way" (since RBF is a built-in kernel)

## note the list 'kpar' is the way to pass parameters to a kernel
## WARNING! the ksvm() method scales the data by default; to prevent it, use scale=c()

svm3 <- ksvm(x,t, type="C-svc", kernel='rbf', kpar=list(sigma=1), scale=c())

# (notice the difference in speed)

## Now we compare the 3 formulations, they *should* be exactly the same
## However, for obscure implementation reasons unknown to me, this is not always the case
## (this happens for the built-in version)

## The RBF built-in version is much faster (it is written in C code)

svm1
svm2
svm3

## Now we are going to make predictions with our hand-computed kernel

## First we split the data into training set and test set
ntrain <- round(N*0.8)     # number of training examples
tindex <- sample(N,ntrain) # indices of training samples

## Then we train the svm with our kernel matrix over the training points:

svm1.train <- ksvm (myRBF.kernel[tindex,tindex],t[tindex], type="C-svc",kernel='matrix')

## Let's call SV the set of obtained support vectors

## Then it becomes tricky. We must compute the test-vs-SV kernel matrix,
## which we do in two phases:

# First the test-vs-train matrix
testK <- myRBF.kernel[-tindex,tindex]

# then we extract the SVs from the train
testK <- testK[,SVindex(svm1.train),drop=FALSE]

dim(testK)

# Now we can predict the test data
# Warning: here we MUST convert the matrix testK to a 'kernelMatrix'
y1 <- predict(svm1.train,as.kernelMatrix(testK))

# Do the same with the usual built-in kernel formulation
svm2.train <- ksvm(x[tindex,],t[tindex], type='C-svc', kernel='rbf', kpar=list(sigma=1), scale=c())

y2 <- predict(svm2.train,x[-tindex,])

# Check that the predictions are the same
table(y1,y2)

# Check the real performance
table(Truth=t[-tindex], Pred=y1)
cat('Error rate = ',100*sum(y1!=t[-tindex])/length(y1),'%')


## Now we are going to understand better the class 'kernel' in kernlab

## An object of class 'kernel' is simply a function (this was expectable)
## with additional slot for kernel parameters, named 'kpar'

## We can start by looking at two built-in kernels to see how they were created
vanilladot
rbfdot

## Let us create a RBF kernel and look at its attributes
rbf <- rbfdot(sigma=1)
rbf
rbf@.Data # the kernel function itself
rbf@kpar  # the kernel paramters
rbf@class # the R class

## Once we have a kernel object, we can do several things, eg:

## 1) Compute the kernel between two vectors
rbf(x[1,],x[2,])

## 2) Compute a kernel matrix between two sets of vectors
K <- kernelMatrix(rbf,x[1:5,],x[6:20,])
dim(K)

## or between a set of vectors with itself (this is the typical use: it gives a square kernel matrix)
K <- kernelMatrix(rbf,x)
dim(K)

## 3) Obviously we can train a SVM
m <- ksvm (x,t, kernel=rbf, type="C-svc", scale=c())

###########################################################################
## Now we are going to make our own kernels and "integrate" them in kernlab

## To make things simple, we start with our "own version" of the linear kernel:

kval <- function(x, y = NULL) 
{
  if (is.null(y)) {
    crossprod(x)
  } else {
    crossprod(x,y)
  }
}

## We then create the kernel object.
## Since this kernel has no parameters, we specify kpar=list(), an empty list

mylinearK <- new("kernel",.Data=kval,kpar=list())

## this is what we did
str(mylinearK)

## compare with
str(myRBF.kernel)

## which was the data only (the kernel matrix)

## Now we can call different functions of kernlab right away

mylinearK (x[1,],x[2,])

kernelMatrix (mylinearK,x[1:5,])

m1 <- ksvm(x,t, kernel=mylinearK, type="C-svc")

# Check that we get the same results as the normal vanilla kernel (linear kernel)
linearK <- vanilladot()
linearK(x[1,],x[2,])
kernelMatrix(linearK,x[1:5,])

m2 <- ksvm(x,t, kernel=linearK, type="C-svc")

m1
m2

## Creating a more complex kernel function -with parameters- is a bit tricky; the easiest way
## is to hack a kernel function directly:

superkernel <- function (sigma1 = 1, sigma2 = 1)
{
  kval <- function (x,y)
  {
    (sum(sqrt(sigma1)*x*y) + 1)*exp(-sigma2*sum((x-y)^2))
  }
  return(new("kernel", .Data=kval, kpar=list(sigma1,sigma2)))
}

superk11 <- superkernel() # will use the default values sigma1 = 1, sigma2 = 1

## 1) Compute the kernel between two vectors
superk11(x[1,],x[2,])

## 2) Compute a kernel matrix between two sets of vectors
K <- kernelMatrix(superk11,x[1:5,],x[5:20,])
dim(K)

## or a kernel matrix
K <- kernelMatrix(superk11,x)
dim(K)

# Now we use our superkernel with diferent parameters:
svp <- ksvm(x,t,type="C-svc",C = 10, kernel=superkernel(2,1),scaled=c())

svp
plot(svp)

## As a final example, we build a kernel that evaluates a precomputed kernel
## This is particularly useful when the kernel is very costly to evaluate,
## so we do it once and store in a external file, for example

## The way we do it is to design a new kernel whose "parameter" is a precomputed kernel matrix K
## The kernel function is then a function of integers i,j such that preK(i,j)=K[i,j]

mypreK <- function (preK=matrix())
{
  rval <- function(i, j = NULL) {
    ## i, j are just indices to be evaluated
    if (is.null(j)) 
    {
      preK[i,i]
    } else 
    {
      preK[i,j]
    }
  }
  return(new("kernel", .Data=rval, kpar=list(preK = preK)))
}

## To simplify matters, suppose we already loaded the kernel matrix from disk into
## our matrix 'myRBF.kernel' (the one we created at the start)

## We create it
myprecomputed.kernel <- mypreK(myRBF.kernel)
str(myprecomputed.kernel)

## We check that it works

myRBF.kernel[seq(5),seq(5)]                 # original matrix (seen just as a matrix)
kernelMatrix(myprecomputed.kernel,seq(5))   # our kernel

## We can of course use it to train SVMs

svm.pre <- ksvm(seq(N),t, type="C-svc", kernel=myprecomputed.kernel, scale=c())
svm.pre

## which should be equal to our initial 'svm1'
svm1

## compare the predictions are equal
p1 <- predict (svm.pre, seq(N))
p2 <- predict (svm1)[1:N]
table(p1,p2)

####################################################################
# 2. PCA and KPCA for the Iris flowers data set
####################################################################

## We illustrate now a kernelized algorithm: kernel PCA

data(iris)

## Have a look at the data

pairs(iris[,-5], panel = panel.smooth, main = "Iris data")

## First we perform good old standard PCA

pca <- prcomp(~Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data=iris)
pca

## This a PCA biplot, a standard visualization tool in multivariate data analysis
## (not covered in this course)
## It allows information on both examples and variables of a data matrix to be displayed simultaneously

## This is a nice visualization tool; if you are interested in meaning, have a look at:

# http://forrest.psych.unc.edu/research/vista-frames/help/lecturenotes/lecture13/biplot.html

biplot(pca)

## these are the returned singular values of the data matrix
## (the square roots of the eigenvalues of the covariance/correlation matrix)
pca$sdev

eigenval <- pca$sdev^2
xpercent <- eigenval[1]/sum(eigenval)*100   # proportion of variance explained by the first PC
ypercent <- eigenval[2]/sum(eigenval)*100   # proportion of variance explained by the second PC

plot (pca$x[,1], pca$x[,2], col=as.integer(iris[,5]),
      main=paste(paste("PCA -", format(xpercent+ypercent, digits=3)), "% explained variance"),
      xlab=paste("1st PC (", format(xpercent, digits=2), "%)"),
      ylab=paste("2nd PC (", format(ypercent, digits=2), "%)"))

## We see that PCA does a wonderful job here in representing/separating the three flower species
## in a lower dimensional space (from 4 to 2 dimensions)
## This is not always the case, given that PCA is an unsupervised and linear technique

## The unsupervised character cannot be changed, but we can capture non-linear PCs with kernel PCA

## first we create a plotting function 

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

## 1. -------------------------------Linear Kernel---------------------
kpv <- kpca(~., data=iris[,1:4], kernel="vanilladot", kpar=list(), features=2)
plotting (kpv, "linear")

## 2. ------------------------------Polynomial Kernel (degree 3)-----------------

kpp <- kpca(~., data=iris[,1:4], kernel="polydot", kpar=list(degree=3,offset=1), features=2)
plotting(kpp,"cubic")

## 3. -------------------------------RBF Kernel-----------------------

kpc1 <- kpca(~., data=iris[,1:4], kernel="rbfdot", kpar=list(sigma=0.6), features=2)
plotting(kpc1,"RBF - sigma 0.6")

kpc2 <- kpca(~., data=iris[,1:4], kernel="rbfdot", kpar=list(sigma=1), features=2)
plotting(kpc2,"RBF - sigma 1.0")

## Note we could use our pre-defined 'rbf' kernel as well
kpc3 <- kpca(~., data=iris[,1:4], kernel=rbf, features=2)
plotting(kpc3,"RBF - sigma 1.0")

## The effect of sigma is a large one ...
kpc4 <- kpca(~., data=iris[,1:4], kernel="rbfdot", kpar=list(sigma=10), features=2)
plotting(kpc4,"RBF - sigma 10")

## The effect of sigma is a large one ...
kpc5 <- kpca(~., data=iris[,1:4], kernel="rbfdot", kpar=list(sigma=0.01), features=2)
plotting(kpc5,"RBF - sigma 0.01")

## This is a drawback of visualization methods, which is aggravated when we go non-linear
## because we now have a tunable parameter; notice that, in this case, upon changing the value
## of sigma, the first two PCs play different roles. As we make sigma grow, the role of the first 
## two PCs tends to get roughly equal; as we make it go to zero, the first PC does the most part

