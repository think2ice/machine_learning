####################################################################
# Machine Learning - MIRI Master
# Lluís A. Belanche

# LAB 8: Resampling methods
# version of April 2016
####################################################################


####################################################################
## 1. A gentle introduction (no data yet!)
####################################################################

## Suppose you fit a model to a data sample.

## How "good" is this solution in reality?

## By this we mean several important questions that should worry you very much if you
## are interested in data analysis/mining, machine learning, statistics ... the right way:

## 1) How good is this solution on average? That is, what systematic error we get if we insist 
## in fitting a specific classifier (e.g. a linear one) to data from our problem?

## WE STUDIED THIS FOR THE REGRESSION CASE ALREADY: IT IS THE BIAS

## 2) How much would this solution change should we change the specific data sample we use?

## WE STUDIED THIS FOR THE REGRESSION CASE ALREADY: IT IS THE VARIANCE

## 3) What methods do we have to estimate the true error of the model?

## In this session we will focus on the last question. We know from previous lectures that computing the 
## error on the same data used for fitting the model is in general wrong, because we can make the
## error optimistically small (overfitting); this phenomenon is aggravated when we have
## a medium-sized data set, and is very serious for small-sized samples, to the point that is can 
## completely falsify results.

## Instead of a fitting error, we need a *prediction* error

## In practice we have only ONE data sample and we need to do three different tasks,
## which require different (actually, independent) data samples. These tasks are:

## 1) fit models to the data ("calculation of the model's coefficients")
## 2) if we have more than one candidate model, select one ("model selection")
## 3) estimate the error of the selected model as honestly as possible ("error estimation")

## We begin by the last question. There is only one universal way:

## Use a separate data sample (called a TEST SAMPLE)

## For solving 1)+2), there are two basic families of methods:

## A) Use a heuristic that combines the training errors obtained with
## some measure of the complexity of the model; one finds methods such as AIC and BIC
## the first is used with linear or logistic regression; the second with E-M for clustering

## There are many drawbacks: it is unclear how they behave for non-linear models, they do not
## provide with an estimation of the error (just an abstract quantity) and are only crude approximations
## ... so we won't pursue them further; however, they find their place in some cases

## B) Use a resampling strategy: divide the data into parts for fitting the models (TRAINING)
## and parts for making them predict (VALIDATION). The general form is called
## CROSS-VALIDATION (explanation on the blackboard), of which LOOCV is a particular case.

## The AVERAGE CROSS-VALIDATION error is used for 2) above ("model selection")

## When we have selected a model, we refit it on the full data available for LEARNING (=TRAINING+VALIDATION)
## and use the final model to predict the test set. It is then possible to derive a confidence interval

## Now we'll be using artificial data so we get some nice numbers to better understand the concepts


####################################################################
## 2. Practice with a classification problem
####################################################################

## the MASS library contains the multivariate gaussian
library(MASS)

## first we fix the seed so we all get the same results
set.seed(1234567)

## We are going to design an artificial 2-class problem, where each class is a multivariate gaussian

## Let us first generate a symmetric positive-definite (PD) matrix that will 
## constitute the covariance matrix of our Gaussians

## We do this by sampling from a Wishart distribution
## for more details, see rWishart {stats}

S <- toeplitz((2:1)/2)

# this S matrix is our reference matrix (the parameter of the Wishart) and is PD
S

## now we sample from the Wishart distribution

R <- rWishart(1, 4, S)

## we get a feasible random 2x2 covariance matrix
dim(R[,,1])
R[,,1]

## now we generate the data for the 2 classes; both classes will share the same
## covariance matrix R that we just generated

## we set 2/3 and 1/3 for the two class priors and (-1,0), (1,1/2) for the two means

## total desired size of the data set (both classes together)
N <- 200

## set class sizes according to priors
prior.1 <- 2/3
prior.2 <- 1/3

N1 <- round(prior.1*N)
N2 <- N - N1

## sample from the gaussians
data1 <- mvrnorm(N1, mu=c(-1,0), Sigma=R[,,1])
data2 <- mvrnorm(N2, mu=c(1,1/2), Sigma=R[,,1])

## Now we create a dataframe with all data stacked by rows, plus the target (the class)

data <- data.frame (rbind(data1,data2),target=as.factor(c(rep('1',N1),rep('2',N2))))

summary(data)

## these are the two classes
plot(data$X1,data$X2,col=data$target)

## If you recall bayesian theory on classification (yes, one of our lectures!)
## The optimal classifier is a linear one (because the two classes share the same covariance matrix)

## Obviously we could compute the correct model because we know the data generation mechanism 
## (in plain words, we do not need the data!)... but this is not realistic, so we won't do it. 

## However to illustrate the ideas we are going to generate as many data as needed

## If we compute this solution using the sample 'data' of size N that we have, we will
## get ONE solution model that depends on the sample used, and so does every quantity that we measure 
## (in particular, he error of the fitted models)

## there you are ... (we already did this in a previous lab)

my.lda <- lda(target ~ X1 + X2, data = data)

## Compute resubstitution (aka apparent, aka training) error

(ct <- table(data$target, predict(my.lda)$class))

# total error (in percentage)
(lda.tr.error <- 100*(1-sum(diag(prop.table(ct)))))

## What if we try now qda()?

my.qda <- qda(target ~ X1 + X2, data = data)

## Compute resubstitution (aka apparent, aka training) error

(ct <- table(data$target, predict(my.qda)$class))

# total error (in percentage)
(qda.tr.error <- 100*(1-sum(diag(prop.table(ct)))))

## So, according to this, qda (a quadratic model) is better (which we know is wrong)

## We need to resort to cross-validation to get more meaningful numbers

library(TunePareto) # for generateCVRuns()

k <- 10

CV.folds <- generateCVRuns(data$target, ntimes=1, nfold=k, stratified=TRUE)

## prepare the structure to store the partial results

cv.results <- matrix (rep(0,4*k),nrow=k)
colnames (cv.results) <- c("k","fold","TR error","VA error")

cv.results[,"TR error"] <- 0
cv.results[,"VA error"] <- 0
cv.results[,"k"] <- k

## let us first compute the 10-fold CV errors

priors <- c(prior.1,prior.2) # for clarity

for (j in 1:k)
{
  # get VA data
  va <- unlist(CV.folds[[1]][[j]])

  # train on TR data
  my.lda.TR <- lda(target ~ X1 + X2, data = data[-va,], prior=priors, CV=FALSE)
  
  # predict TR data
  pred.va <- predict (my.lda.TR)$class
  
  tab <- table(data[-va,]$target, pred.va)
  cv.results[j,"TR error"] <- 1-sum(tab[row(tab)==col(tab)])/sum(tab)
  
  # predict VA data
  pred.va <- predict (my.lda.TR, newdata=data[va,])$class
  
  tab <- table(data[va,]$target, pred.va)
  cv.results[j,"VA error"] <- 1-sum(tab[row(tab)==col(tab)])/sum(tab)
  
  cv.results[j,"fold"] <- j
}
  
## have a look at the results ...
cv.results

## Now we see the large variability across the different VA data

## What one really uses is the average of the last column
(VA.error <- mean(cv.results[,"VA error"]))

## You may think that instead the average is quite good, but this need not be the case,
## as we shall see in a minute

## Now let us change the number of folds/experiments k

## To do this, we embed the previous code into a function; we also prepare it for either
## LDA or QDA

## The function returns the AVERAGE CROSS-VALIDATION error

DA.CV <- function (k, method)
{
  CV.folds <- generateCVRuns(data$target, ntimes=1, nfold=k, stratified=TRUE)
  
  cv.results <- matrix (rep(0,4*k),nrow=k)
  colnames (cv.results) <- c("k","fold","TR error","VA error")
  
  cv.results[,"TR error"] <- 0
  cv.results[,"VA error"] <- 0
  cv.results[,"k"] <- k
  
  priors <- c(prior.1,prior.2) # for clarity
  
  for (j in 1:k)
  {
    # get VA data
    va <- unlist(CV.folds[[1]][[j]])
    
    # train on TR data
    if (method == "LDA") { my.da.TR <- lda(target ~ X1 + X2, data = data[-va,], prior=priors, CV=FALSE) }
    else if (method == "QDA") { my.da.TR <- qda(target ~ X1 + X2, data = data[-va,], prior=priors, CV=FALSE) }
    else stop("Wrong method")
    
    # predict TR data
    pred.va <- predict (my.da.TR)$class
    
    tab <- table(data[-va,]$target, pred.va)
    cv.results[j,"TR error"] <- 1-sum(tab[row(tab)==col(tab)])/sum(tab)
    
    # predict VA data
    pred.va <- predict (my.da.TR, newdata=data[va,])$class
    
    tab <- table(data[va,]$target, pred.va)
    cv.results[j,"VA error"] <- 1-sum(tab[row(tab)==col(tab)])/sum(tab)
    
    cv.results[j,"fold"] <- j
  }
  mean(cv.results[,"VA error"])
}

## Armed with this function, we are going to produce a plot by changing k

the.Ks <- 2:20
res <- vector("numeric",length(the.Ks)+1)
for (k in the.Ks) res[k] <- DA.CV(k,"LDA")

## let us see the results
plot(res[-1],type="b",xlab="Value of k",ylab="average CV error", ylim=c(0.22,0.3))

## Now I'll reveal you the truth ...

# The following function computes the true probability of error for two-class normally distributed
# features with equal covariance matrices and arbitrary means and priors
# the formula is well-known and can be found in pattern recognition textbooks

prob.error <- function (Pw1, Pw2, Sigma, Mu1, Mu2)
{
  # Numerically correct way for t(x) %*% solve(M) %*% (x), i.e., for x^T M^{-1} x
  quad.form.inv <- function (M, x)
  {
    drop(crossprod(x, solve(M, x)))
  }
  
  stopifnot (Pw2+Pw1==1,Pw2>0,Pw1>0,Pw2<1,Pw1<1)
  alpha <- log(Pw2/Pw1)
  D <- quad.form.inv (Sigma, Mu1-Mu2)
  A1 <- (alpha-D/2)/sqrt(D)
  A2 <- (alpha+D/2)/sqrt(D)
  Pw1*pnorm(A1)+Pw2*(1-pnorm(A2))
}

## In this case we get:

(pe <- prob.error (prior.1,prior.2,R[,,1],c(-1,0),c(1,1/2)))


## add it to the plot (it may be that if falls outside the plot, in the bottom)
abline(pe,0)

## We can see that no estimation gives you the correct result, most (if not all) of them over-estimate it
## (there could be some under-estimation too); and the thing stabilizes gently to a slight over-estimate as k increases

## So there is no "best" value for k to be used in k-CV; it depends largely on the amount of data
## more data allows one to decrease k; less and less data forces to use a large k

## Aha! but these numbers are random numbers, because they depend on the specific partition that
## k-CV uses; correct: so a good idea if your computational resources permit is to iterate the process ...

iters <- 20

res <- vector("numeric",length(the.Ks)+1)
for (k in the.Ks) res[k] <- mean(replicate(iters,DA.CV(k,"LDA")))

## let us see the results
plot(res[-1],type="b",xlab="Value of k", ylim=c(0.24,0.3))

# in blue the true error
abline(pe,0,col="blue")

# in red the training error
abline(lda.tr.error/100, 0, col="red")

# in green the LOOCV error (notice this one cannot be iterated)
lda.LOOCV <- lda(target ~ X1 + X2, data = data, prior=priors, CV=TRUE)

(ct <- table(data$target, lda.LOOCV$class))
(lda.LOOCV.error <- 1-sum(diag(prop.table(ct))))
abline(lda.LOOCV.error, 0, col="green")

legend("topright", legend=c("average k-CV error", "true error", "training error", "LOOCV error"),    
       pch=c(1,1), col=c("black", "blue","red","green"))

# in this case both training error and LOOCV errors coincide; in general the former will be larger

## In comparison to the previous results, we can see that now all estimations are over-estimates, and
## the thing has stabilized quite a lot; and the training error is an optimistic value

## Ask yourself why the CV error is consistently an over-estimate of the true error (that is, is a pessimistic value)

## Now we'll use these errors for model selection (you will see that this is possible, because they will have lower
## values for better models); however, you can realize that if, in addition to model selection, you want to give a more
## correct estimation of the true error of a model, the CV error is not correct. That is why you need a separate TEST sample

## There is no optimal k, but you have to choose one: very often this is a very imprecise function of data sample size
## and computational resources (larger values of k entail a heavier computational burden)

## Standard choices are 5CV or 10CV, 10x10 CV and LOOCV

## Let us choose 10x10 CV ... and use it to choose among LDA and QDA for this problem
## Yes ... we know LDA is the optimal, but this we don't know in practice

(lda.10x10.CV <- mean(replicate(10,DA.CV(10,"LDA"))))

(qda.10x10.CV <- mean(replicate(10,DA.CV(10,"QDA"))))

## It is a close shave but the result is correct; now we will follow the standard procedure:

## We need a test set to estimate the true error of our LDA model
## We make it rather large to have a more significant estimation

N.test <- 10000

N1 <- round(prior.1*N.test)
N2 <- N.test - N1

data1 <- mvrnorm(N1, mu=c(-1,0), Sigma=R[,,1])
data2 <- mvrnorm(N2, mu=c(1,1/2), Sigma=R[,,1])

## Now we create a dataframe with all data stacked by rows

data.test <- data.frame (rbind(data1,data2),target=as.factor(c(rep('1',N1),rep('2',N2))))

## First we refit our LDA model to the full sample used for learning

my.lda <- lda(target ~ X1 + X2, data = data)

## And now we make it predict the test set

(ct <- table(data.test$target, predict(my.lda, newdata=data.test)$class))

## total error (in percentage)
(pe.hat <- 1-sum(diag(prop.table(ct))))

## Now a 95% CI around it

dev <- sqrt(pe.hat*(1-pe.hat)/N.test)*1.967

sprintf("(%f,%f)", pe.hat-dev,pe.hat+dev)

## true error of our model was
pe

## so the method works

