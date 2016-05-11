####################################################################
# Machine Learning - MIRI Master
# Lluís A. Belanche

# LAB 11: Multilayer Perceptrons (Part 1)
# version of May 2016
####################################################################

library(MASS)
library(nnet)

####################################################################
## Multilayer Perceptron Example 1: Admission into graduate school data
####################################################################

## We recover a dataset from a previous lab session
## Suppose we are interested in how variables, such as 

## GRE (Graduate Record Exam scores)
## GPA (Grade Point Average) and 
## rank (prestige of the undergraduate institution),
## affect admission into graduate school.

## The target variable, admit/don't admit, is a binary variable, which we want to characterize
## and, if possible, to predict (a model)


Admis <- read.csv("Admissions.csv")

## view the first few rows of the data
head(Admis)

## We will treat the variables 'gre' and 'gpa' as continuous. 
## The variable 'rank' takes on the values 1 through 4, so we can fairly treat it as numerical
## (although, in rigour, it is ordinal)

Admis$admit <- factor(Admis$admit, labels=c("No","Yes"))

summary(Admis)

N <- nrow(Admis)

## We first split the available data into learning and test sets, selecting randomly 2/3 and 1/3 of the data
## We do this for a honest estimation of prediction performance

set.seed(43)

learn <- sample(1:N, round(2*N/3))

nlearn <- length(learn)
ntest <- N - nlearn

## Maybe you recall that, using logistic regression (a linear classifier),
## we got a prediction error which was quite high (around 30%) in the test part

## The nnet() function is quite powerful and very reliable from the optimization
## point fo view. From the computational point fo view, it has two limitations:

## 1- it does not have a built-in mechanism for multiple runs or cross-validation
## 2- it only admits networks of one hidden layer (of 'size' hidden neurons)

## Please a quick look at ?nnet before going any further

## The basic parameters are 'size', 'decay' (the regularization constant, lambda)
## As usual, R detects it is a classification problem because 'admit' is a factor
## It buils a MLP with one output neuron (just two classes), with the logistic function
## and uses the cross-entropy as error function

## Let's start by standardizing the inputs; this is important to avoid network stagnation (premature convergence)

Admis$gpa <- scale(Admis$gpa)
Admis$gre <- scale(Admis$gre)
Admis$rank <- scale(Admis$rank)

## To illustrate the first results, we just fit a MLP with 2 hidden neurons

model.nnet <- nnet(admit ~., data = Admis, subset=learn, size=2, maxit=200, decay=0)

## Take your time to understand the output
model.nnet 

## In particular, why the total number of weights is 11, what 'initial  value' and 'final  value' are
## and what does 'converged' mean

# This is the fitting criterion (aka error function)
model.nnet$value

#  fitted values for the training data
model.nnet$fitted.values

# and the residuals
model.nnet$residuals

## Now look at the weights

model.nnet$wts

## I think this way is clearer:

summary(model.nnet)

## i1,i2,i3 are the 3 inputs, h1, h2 are the two hidden neurons, b is the bias (offset)

## As you can see, some weights are large (two orders of magnitude larger then others)
## This is no good, since it makes the model unstable (ie, small changes in some inputs may
## entail significant changes in the network, because of the large weights)

## One way to avoid this is by regularizing the learning process:

model.nnet <- nnet(admit ~., data = Admis, subset=learn, size=2, maxit=200, decay=0.01)

## notice the big difference
summary(model.nnet)

# Now let's compute the training error

p1 <- as.factor(predict (model.nnet, type="class"))

t1 <- table(p1,Admis$admit[learn])
(error_rate.learn <- 100*(1-sum(diag(t1))/nlearn))

# And the corresponding test error

p2 <- as.factor(predict (model.nnet, newdata=Admis[-learn,], type="class"))

t2 <- table(p2,Admis$admit[-learn])
(error_rate.test <- 100*(1-sum(diag(t2))/ntest))

## We get 26.32%, so it seems that the MLP helps a little bit; however, we need to work harder

## We are going to do the modelling in a principled way now. Using 10x10 CV to select the best
## combination of 'size' and 'decay'

## Just by curiosity, let me show how with a non-linear model we can fit almost any dataset (in the sense of reducing the training error):

model.nnet <- nnet(admit ~., data = Admis, subset=learn, size=20, maxit=200)

# Now let's compute the training error

p1 <- as.factor(predict (model.nnet, type="class"))

t1 <- table(p1,Admis$admit[learn])
(error_rate.learn <- 100*(1-sum(diag(t1))/nlearn))

# And the corresponding test error

p2 <- as.factor(predict (model.nnet, newdata=Admis[-learn,], type="class"))

t2 <- table(p2,Admis$admit[-learn])
(error_rate.test <- 100*(1-sum(diag(t2))/ntest))

## That's it: we got a training error around 6% (four times lower than the previous one), but it is 
## illusory ... the test error is larger than before (around 40%); 
## Actually the relevant comparison is between 6% and 40%, this large gap is an indication of overfitting


## {caret} is an excellent package for training control, once you know what all these concepts are

## WARNING: if the package is not installed in your computer, installation takes quite a while
library(caret)

## For a specific model, in our case the neural network, the function train() in {caret} uses a "grid" of model parameters
## and trains using a given resampling method (in our case we will be using 10x10 CV). All combinations are evaluated, and 
## the best one (according to 10x10 CV) is chosen and used to construct a final model, which is refit using the whole training set

## Thus train() returns the constructed model (exactly as a direct call to nnet() would)

## In order to find the best network architecture, we are going to explore two methods:

## a) Explore different numbers of hidden units in one hidden layer, with no regularization
## b) Fix a large number of hidden units in one hidden layer, and explore different regularization values (recommended)

## doing both (explore different numbers of hidden units AND regularization values) is usually a waste of computing resources (but notice that train() would admit it)

## Let's start with a)

## set desired sizes

(sizes <- 2*seq(1,10,by=1))

## specify 10x10 CV
trc <- trainControl (method="repeatedcv", number=10, repeats=10)

model.10x10CV <- train (admit ~., data = Admis, subset=learn, method='nnet', maxit = 500, trace = FALSE,
                        tuneGrid = expand.grid(.size=sizes,.decay=0), trControl=trc)

## We can inspect the full results
model.10x10CV$results

## and the best model found
model.10x10CV$bestTune

## The results are quite disappointing ...

## Now method b)

(decays <- 10^seq(-3,0,by=0.1))

## WARNING: this takes a few minutes
model.10x10CV <- train (admit ~., data = Admis, subset=learn, method='nnet', maxit = 500, trace = FALSE,
                        tuneGrid = expand.grid(.size=20,.decay=decays), trControl=trc)

## We can inspect the full results
model.10x10CV$results

## and the best model found
model.10x10CV$bestTune

## The results are a bit better; we should choose the model with the lowest 10x10CV error overall,
## in this case it corresponds to 20 hidden neurons, with a decay of 0.7943282

## So what remains is to predict the test set with our final model

p2 <- as.factor(predict (model.10x10CV, newdata=Admis[-learn,], type="raw"))

t2 <- table(pred=p2,truth=Admis$admit[-learn])
(error_rate.test <- 100*(1-sum(diag(t2))/ntest))

## We get 27.82% after all this work; it seems that the information in this dataset is not enough
## to accurately predict admittance. Note that ...

## ... upon looking at the confusion matrix for the predictions ...
t2

## it clearly suggests that quite a lot of people is getting accepted when they should not, given their gre, gpa and rank
## It is very likely that other (subjective?) factors are being taken into account, that are not in the dataset

## A different approach is to do the same thing but using the square error instead of the cross-entropy
## This is conceptually different, because we are now treating the class labels as numbers
## Although less principled, many people do it, and in practice it often delivers similar results


####################################################################
## Multilayer Perceptron Example 2: circular artificial 2D data
####################################################################

set.seed(3)

p <- 2
N <- 200

x <- matrix(rnorm(N*p),ncol=p)
y <- as.numeric((x[,1]^2+x[,2]^2) > 1.4)
mydata <- data.frame(x=x,y=y)
plot(x,col=c('black','green')[y+1],pch=19,asp=1)

## Let's use one hidden layer, 3 hidden units, no regularization and the error function "cross-entropy"
## In this case it is not necessary to standardize because they variables already are
## (they have been generated from a distribution with mean 0 and standard deviation 1).

nn1 <- nnet(y~x.1+x.2,data=mydata,entropy=T,size=3,decay=0,maxit=2000,trace=T)

yhat <- as.numeric(predict(nn1,type='class'))
par(mfrow=c(1,2))
plot(x,pch=19,col=c('red','blue')[y+1],main='actual labels',asp=1)
plot(x,col=c('red','blue')[(yhat>0.5)+1],pch=19,main='predicted labels',asp=1)
table(actual=y,predicted=predict(nn1,type='class'))

## Excellent, indeed

## Let's execute it again, this time wth a different random seed

set.seed(4)

nn1 <- nnet(y~x.1+x.2,data=mydata,entropy=T,size=3,decay=0,maxit=2000,trace=T)
yhat <- as.numeric(predict(nn1,type='class'))
par(mfrow=c(1,2))
plot(x,pch=19,col=c('red','blue')[y+1],main='actual labels',asp=1)
plot(x,col=c('red','blue')[(yhat>0.5)+1],pch=19,main='predicted labels',asp=1)
table(actual=y,predicted=predict(nn1,type='class'))

## we see that the optimizer does not always find a good solution

## How many hidden units do we need?

par(mfrow=c(2,2))
for (i in 1:4)
{
  set.seed(3)
  nn1 <- nnet(y~x.1+x.2,data=mydata,entropy=T,size=i,decay=0,maxit=2000,trace=T)
  yhat <- as.numeric(predict(nn1,type='class'))
  plot(x,pch=20,col=c('black','green')[yhat+1])
  title(main=paste('nnet with',i,'hidden unit(s)'))
}


## Let's find out which function has been learned exactly, with 3 units

set.seed(3)
nn1 <- nnet(y~x.1+x.2,data=mydata,entropy=T,size=3,decay=0,maxit=2000,trace=T)

## create a grid of values
x1grid <- seq(-3,3,l=200)
x2grid <- seq(-3,3,l=220)
xg <- expand.grid(x1grid,x2grid)
xg <- as.matrix(cbind(1,xg))

## input them to the hidden units, and get their outputs
h1 <- xg%*%matrix(coef(nn1)[1:3],ncol=1)
h2 <- xg%*%matrix(coef(nn1)[4:6],ncol=1)
h3 <- xg%*%matrix(coef(nn1)[7:9],ncol=1)

## the hidden units compute the tanh() function, so we cut the output value at 0; we get a decision line

par(mfrow=c(2,2))
contour(x1grid,x2grid,matrix(h1,200,220),levels=0)
contour(x1grid,x2grid,matrix(h2,200,220),levels=0,add=T)
contour(x1grid,x2grid,matrix(h3,200,220),levels=0,add=T)
title(main='net input = 0\n in the hidden units')

## this is the logistic function, used by nnet() for the hidden neurons, and 
## for the output neurons in two-class classification problems
logistic <- function(x) {1/(1+exp(-x))}

z <- coef(nn1)[10] + coef(nn1)[11]*logistic(h1) + coef(nn1)[12]*logistic(h2) + coef(nn1)[13]*logistic(h3)

contour(x1grid,x2grid,matrix(z,200,220))
title('hidden outputs = logistic of the net inputs\n and their weighted sum')
contour(x1grid,x2grid,matrix(logistic(z),200,220),levels=0.5)
title('logistic of the previous sum')
contour(x1grid,x2grid,matrix(logistic(z),200,220),levels=0.5)
points(x,pch=20,col=c('black','green')[y+1])
title('same with training data points')

####################################################################
## Multilayer Perceptron Example 3: mixture of distributions artificial 2D data
####################################################################

## Thanks to Thomas Stibor for the excellent plotting code

set.seed (4)

mixture.distributions <- function (N,N2=N)
{
  z<- c(rbinom(1,N,1/3))
  z[2]<-rbinom(1,N-z[1],1/2)
  v<- c(rbinom(1,N,1/3))
  v[2]<-rbinom(1,N-v[1],1/2)
  x <- c(rnorm(z[1])/2+2, rnorm(z[2])+6, rnorm(N-z[1]-z[2])/2+3, rnorm(v[1])/1+4, rnorm(v[2])/3+1,
         rnorm(N-v[1]-v[2])+8)
  y <- c(rnorm(z[1])/2+1, rnorm(z[2])/3+1, rnorm(N-z[1]-z[2])/2+1.5, rnorm(v[1])/3+2, rnorm(v[2])/3+2,
         rnorm(N-v[1]-v[2])/3+1.5)
  target <- c(rep("red",N),rep("blue",N2))
  return(data.frame(x,y,target))
}

N <- 400
data <- mixture.distributions (N)

## Let's see what we have done 
## A 2D classification problem

par(mfrow=c(1,1))
plot(data$x,data$y, col='green')

## where the class borders are not clear ... a realistic dataset
## (or a 2D projection of a realistic dataset)

plot(data$x,data$y, col=as.character(data$target))

## Let's plot the real data generation mechanism

red.density <- function(x) 
{
  1/3*(dnorm(x[1],2,.5)*dnorm(x[2],1,.5)+
       dnorm(x[1],6,1)*dnorm(x[2],1,1/3)+
       dnorm(x[1],3,0.5)*dnorm(x[2],1.5,.5))
}

blue.density <- function(x) 
{
  1/3*(dnorm(x[1],4,1)*dnorm(x[2],2,1/3)+
       dnorm(x[1],1,1/3)*dnorm(x[2],2,1/3)+
       dnorm(x[1],8,1)*dnorm(x[2],1.5,1/3))
}

## determine min-max range
rng <- apply(cbind(data$x,data$y),2,range)

## create tx times ty grid 
resol <- 200
tx <- seq(rng[1,1],rng[2,1],length=resol)
ty <- seq(rng[1,2],rng[2,2],length=resol)
pnts <- matrix(nrow=length(tx)*length(ty),ncol=2)
k <- 1
for(j in 1:length(ty))
{
  for(i in 1:length(tx))
  {
    pnts[k,] <- c(tx[i],ty[j])
    k <- k+1
  }
}

## set up coordinate system
plot(data$x, data$y, type = "n", asp=0, xlab = "x", ylab = "y",
     xlim=c(rng[1,1],rng[2,1]), ylim=c(rng[1,2],rng[2,2]), cex.axis = 1.2, axes=TRUE)

## plot points
points(data$x,data$y,col=as.vector(data$target),pch=19,cex=0.5)

## plot density contour for both classes
blue.class <- matrix(apply(pnts,1,blue.density),nrow=length(tx),ncol=length(ty))
contour(tx, ty, blue.class, add = TRUE, col="blue",nlevels=8)

red.class <- matrix(apply(pnts,1,red.density),nrow=length(tx),ncol=length(ty))
contour(tx, ty, red.class, add = TRUE, col="red", nlevels=15)

## plot optimal Bayes separation curve
bayes.sep <- matrix((apply(pnts,1,blue.density)-apply(pnts,1,red.density)),nrow=length(tx),ncol=length(ty))
contour(tx, ty, bayes.sep, add = TRUE, col="black", levels=0, 
        lwd=5, drawlabels=TRUE, labels="Bayes optimal",labcex=1, method="edge")


## Now let's train a MLP of different sizes (2,4,8,16) and we'll plot the learned classifier
## against the truth to see what the network is really doing

## To do this, we embed the previous code into a function and add the MLP training part

train.plot <- function (N.hidden)
{
  ## set up coordinate system
  plot(data$x, data$y, type = "n", asp=0, xlab = "", ylab = "", main = paste(N.hidden, "hidden neurons"),
       xlim=c(rng[1,1],rng[2,1]), ylim=c(rng[1,2],rng[2,2]), cex.axis = 1.2, axes=TRUE)
  
  ## plot points
  points(data$x,data$y,col=as.vector(data$target),pch=19,cex=1.5)
  
  ## plot density contour for both classes
  blue.class <- matrix(apply(pnts,1,blue.density),nrow=length(tx),ncol=length(ty))
  contour(tx, ty, blue.class, add = TRUE, col="blue",nlevels=8)
  
  red.class <- matrix(apply(pnts,1,red.density),nrow=length(tx),ncol=length(ty))
  contour(tx, ty, red.class, add = TRUE, col="red", nlevels=15)
  
  ## plot optimal Bayes separation curve
  bayes.sep <- matrix((apply(pnts,1,blue.density)-apply(pnts,1,red.density)),nrow=length(tx),ncol=length(ty))
  contour(tx, ty, bayes.sep, add = TRUE, col="black", levels=0, 
          lwd=5, drawlabels=TRUE, labels="Bayes optimal",labcex=1, method="edge")
  
  ## train the MLP
  
  learned.nnet  <- nnet(target ~ x+y, data, size=N.hidden, maxit=1000)

  prediction.nnet  <- predict(learned.nnet, data.frame(x=pnts[,1],y=pnts[,2]))

  z <- matrix(prediction.nnet,nrow=length(tx),ncol=length(ty))

  ## plot neural network decision function

  contour(tx,ty,z,add=T,levels=0.5,lwd=5,col="green", drawlabels=TRUE, labcex=1, method="edge",labels="MLP solution")
}

## this takes a while but it's worth the wait
par(mfrow=c(2,2))

train.plot (2)
train.plot (4)
train.plot (8)
train.plot (16)

## The solutions with 2 and 4 neurons clearly underfit the data, while that with 16 neurons
## clearly overfits the data; the one with 8 neurons seems a nice fit
## We should use regularization instead and guide the process by the 10x10 CV error on training data (not visually!)
## as we did for Example 1, but this was an exercise about visualization of the process

## To be frank, this is what you are going to do in Part 2, among other tasks.

