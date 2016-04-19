C.H <- function (clres, x)
{
    gss <- function(x, clsize, withins) 
    {
        n <- sum(clsize)
        k <- length(clsize)
        allmean <- apply(x, 2, mean)
        dmean <- sweep(x, 2, allmean, "-")
        allmeandist <- sum(dmean^2)
        wgss <- sum(withins)
        bgss <- allmeandist - wgss
        list(wgss = wgss, bgss = bgss)
       }
    
    cal.har <- function(zgss, clsize) 
    {
        n <- sum(clsize)
        k <- length(clsize)
        vrc <- (zgss$bgss/(k - 1))/(zgss$wgss/(n - k))
        return(vrc = vrc)
    }
    
    cal.har (gss(x, clres$size, clres$withins), clres$size)
}