####################################################################
# Machine Learning - MIRI Master
# Author: 
# think_2ice <- Manel Alba

#  LAB 7: Clustering (k-means and E-M) (Part2)
# version of April 2016
####################################################################


####################################################################
## Exercise 1:  Clustering artificial 2-D CIRCLE DATA

library(mlbench)

# We generate 2D data: each of the clusters is a 2-D Gaussian. The centers are equally spaced 
# on a circle around the origin with radius r
# The covariance matrices are of the form sigma^2 I (sd^2 parameter in  mlbench.2dnormals())

N <- 1000
K <- 6

# the clusters
data.1 <- mlbench.2dnormals (N,K)
plot(data.1)

# the raw data (what the clustering method will receive)

plot(x=data.1$x[,1], y=data.1$x[,2])

## Do the following (in separate sections)

# 1. Decide beforehand which clustering method will work best and with which settings.
#    Hint: have a look at the way the data is generated: ?mlbench.2dnormals

# The data is a 2 dimension gaussian problem. K-means is a form of Gaussian mixture so 
# it should work pretty good

# 2. Apply k-means a number of times with fixed K=6 and observe the results

library (cclust)
K <- 6

kmeans.6 <- cclust (data.1$x,K,iter.max=100,method="kmeans",dist="euclidean")
## plot and paint the clusters (according to the computed assignments)
plot(data.1$x[,1],data.1$x[,2],col=(kmeans.6$cluster+1))
## plot the cluster centers
points(kmeans.6$centers,col=seq(1:kmeans.6$ncenters)+1,cex=2,pch=19)


# 3. Apply k-means with a choice of K values of your own and monitor the CH index; which K looks better?

# Function that execute a kmeans and outputs its Calinski-Harabasz index
do.kmeans <- function (whatK)
{
  r <- cclust (data.1$x,whatK,iter.max=100,method="kmeans",dist="euclidean")
  (clustIndex(r,data.1$x, index="calinski"))
}
CHs <- 2:10
for (i in 2:10){
  K <- i
  # Replicate the kmeans to work with an average value
  # It could happen that in only one realization I found a better
  # value than the optimal because this is a heuristic procedure
  CHs[i-1] <- max (r <- replicate (100, do.kmeans(K)))
}
plot(2:10,CHs, type = "b")
cat("The optimum number of clusters is ", which.max(CHs)+1)

# 4. Apply E-M with K=6 and observe the results (means, coefficients and covariances)

library(Rmixmod)

data.1.model <- mixmodGaussianModel (family="all", equal.proportions=FALSE)
z <- mixmodCluster (data.frame(data.1$x),models = data.1.model, nbCluster = 6)
summary(z)
z@bestResult
# hard assignments
found.clusters <- z@bestResult@partition
# the final centers
means <- z@bestResult@parameters@mean

# 5. Check the results against tour expectations (#1.)

par(mfrow = c(1,3))
# Truth clusters
plot(data.1, main = "Truth vs")
# EM centers and clusters
plot(data.1$x[,1],data.1$x[,2],col=(found.clusters+1), main = "E-M vs")
points(means,col=seq(1:5)+1,cex=2,pch=19)
K=6
CH = 0
for (i in 1:100){
  aux <- cclust (data.1$x,K,iter.max=100,method="kmeans",dist="euclidean")
  if (CH < clustIndex(aux,data.1$x, index="calinski")){
    kmeans.6 <- aux
  }
}
# Kmeans centers and clusters
plot(data.1$x[,1],data.1$x[,2],col=(kmeans.6$cluster+1), main = "Kmeans")
points(kmeans.6$centers,col=seq(1:kmeans.3$ncenters)+1,cex=2,pch=19)

####################################################################
## Exercise 2:  Clustering real 2-D data

## This exercise involves the use of the 'Geyser' data set, which contains data from the ‘Old Faithful’ geyser 
## in Yellowstone National Park, Wyoming. 

## the MASS library seems to contain the best version of the data
library(MASS)
help(geyser, package="MASS")
summary(geyser)
str(geyser)
plot(geyser)
## with ggplot2, maybe we get better plots:
library(ggplot2)
qplot(waiting, duration, data=geyser)

# 6. Decide beforehand which clustering method will work best and with which settings.
#    No hint this time, this is a real dataset

par(mfrow = c(1,2))
plot(density(geyser$duration), main = "Duration density")
plot(density(geyser$waiting), main = "Waiting density")
# Mixture of gaussians in both variables looking at their densities?
# From my point of view, observing the data seems to be 3 groups or clusters. 
# So I can follow 2 paths, using E-M or using k-means with k=3
K <- 3
kmeans.3 <- cclust (data.matrix(geyser),K,iter.max=100,method="kmeans",dist="euclidean")
CH3 <- (clustIndex(kmeans.3, data.matrix(geyser), index="calinski"))
plot(geyser, main="Geyser data")
plot(as.matrix(geyser),col=(kmeans.3$cluster+1), main = "Kmeans with k = 3")



# 7. Apply k-means with different values of K and observe the results

for (k in 2:15) {
  K <- k
  kmeans <- cclust (data.matrix(geyser),K,iter.max=100,method="kmeans",dist="euclidean")
  CH <- (clustIndex(kmeans,data.matrix(geyser), index="calinski"))
  cat("Number of clusters ", kmeans$ncenters,"CH: ",CH, "\n")
}


# 8. Apply k-means 100 times, get averages of the CH index, and decide the best value of K. Does it work?

par(mfrow = c(1,1))
gey <- data.matrix(geyser)
do.kmeans <- function (whatK)
{
  r <- cclust (gey, whatK, iter.max=100, method="kmeans", dist="euclidean")
  (clustIndex(r, gey, index="calinski"))
}
CHs <- 2:20
x <- CHs
for (i in 2:20){
  K <- i
  CHs[i-1] <- max (r <- replicate (100, do.kmeans(K)))
}
plot(x ,CHs, type = "b")
cat("The optimum number of clusters is ", which.max(CHs)+1)

# 9. Apply E-M with a family of your choice ("spherical", "diagonal", etc), with the best value fo K delivered by k-means

# Best k 
K <- 10
model.famili = c("general","diagonal","spherical","all")
for (f in model.famili) {
  em.model <- mixmodGaussianModel (family="all", equal.proportions=FALSE)
  z <- mixmodCluster (data.frame(geyser),models = em.model, nbCluster = K)
  summary(z)
}

z@bestResult
# hard assignments
found.clusters <- z@bestResult@partition
# the final centers
means <- z@bestResult@parameters@mean

# 10. Choose the model and number of clusters with the largest BIC



# 11. Apply E-M again with a family of your choice ("spherical", "diagonal", etc), this time letting BIC decide the best number of clusters
#     The easiest way to inspect the final results is with summary() of your mixmodCluster() call



# 12. Once you're done, try and plot the results; just plot() the result of mixmodCluster()



####################################################################
## Exercise 3:  Clustering real multi-dimensional data

## This exercise involves the use of the 'Auto' data set, which we introduced in a previous lab session

# 13. Get the Auto data, redo the preprocessing



# 14. Apply E-M again with a family of your choice ("spherical", "diagonal", etc), letting BIC decide the best
#     number of clusters



# 15. Inspect and report the results of your clustering
#     Warning: do not directly plot() the results, it takes a long time



# 16. Use the clusplot() function in {cluster}
#     Like this:
#        library(cluster)
#        clusplot(Auto, z@bestResult@partition, color=TRUE, shade=TRUE, labels=2, lines=0)
#     please do consult ?clusplot.default



