####################################################################
# Machine Learning - MIRI Master
# Lluís A. Belanche

# LAB 12: Radial Basis Function Network (Part 2)
# version of May 2016
####################################################################


####################################################################
## Exercise

## We continue with Example 1: regression of a 1D function

## We are interested in studying the influence of sample size on the fit.
## The idea is that you embed the code in Part 1 into a couple of handy functions and leave
## the learning sample size (N) as a parameter.

## These are the learning sample sizes you are going to study

Ns <- c(20,50,100,200,500)

## You are asked to report the chosen lambda and the final test error (on the same test set),
## plot the learned function against the data and the true genearting function and see if the fit
## is better/worse/equal as a function of N and in what sense it is better/worse/equal

# Your code starts here ...
set.seed (4)
myf <- function (x) { (1 + x - 2*x^2) * exp(-x^2) }
a <- -5
b <- 5
domain <- c(a,b)

myf.data <- function (N, a, b) 
{
  x <- runif(N, a, b)
  t <- myf(x) + rnorm(N, sd=0.2)
  dd <- data.frame(x,t)
  names(dd) <- c("x", "t")
  return (dd)
}

N <- 200
d <- myf.data (N, a , b)

summary(d)

## The black points are the data, the blue line is the true underlying function

plot (d)
curve (myf, a, b, col='blue', add=TRUE)
