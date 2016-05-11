plot.prediction <- function (model, model.name, resol=200)
  # the grid has a (resol x resol) resolution
{
  x <- cbind(dataset$x1,dataset$x2)
  rng <- apply(x,2,range);
  tx <- seq(rng[1,1],rng[2,1],length=resol);
  ty <- seq(rng[1,2],rng[2,2],length=resol);
  pnts <- matrix(nrow=length(tx)*length(ty),ncol=2);
  k <- 1
  for(j in 1:length(ty))
  {
    for(i in 1:length(tx))
    {
      pnts[k,] <- c(tx[i],ty[j])
      k <- k+1
    } 
  }
  
  # we calculate the predictions on the grid
  
  pred <- predict(model, pnts, decision.values = TRUE)
  
  z <- matrix(attr(pred,"decision.values"),nrow=length(tx),ncol=length(ty))
  
  # and plot them
  
  image(tx,ty,z,xlab=model.name,ylab="",axes=FALSE,
        xlim=c(rng[1,1],rng[2,1]),ylim=c(rng[1,2],rng[2,2]),
        col = rainbow(200, start=0, end=.25))
  
  # then we draw the optimal separation and its margins
  
  contour(tx,ty,z,add=TRUE, drawlabels=TRUE, level=0, lwd=3)
  contour(tx,ty,z,add=TRUE, drawlabels=TRUE, level=1, lty=1, lwd=1, col="grey")
  contour(tx,ty,z,add=TRUE, drawlabels=TRUE, level=-1, lty=1, lwd=1, col="grey")
  
  # then we plot the input data from the two classes
  
  points(dataset[dataset$target==1,1:2],pch=21,col=1,cex=1)
  points(dataset[dataset$target==-1,1:2],pch=19,col=4,cex=1)
  
  # finally we add the SVs
  
  sv <- dataset[c(model$index),];
  sv1 <- sv[sv$target==1,];
  sv2 <- sv[sv$target==-1,];
  points(sv1[,1:2],pch=13,col=1,cex=2)
  points(sv2[,1:2],pch=13,col=4,cex=2)
}