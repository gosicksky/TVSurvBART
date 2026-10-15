predict.wbart <- function(object, target_time, newdata, mc.cores=1, openmp=(mc.cores.openmp()>0), ...) {
  
  ##if(class(newdata) != "matrix") stop("newdata must be a matrix")
  
  #new data: the first two columns should be id and time
  
  p <- length(object$treedraws$cutpoints)
  
  if(p!=ncol(as.data.frame(bartModelMatrix(newdata[,3:ncol(newdata)]))))
    stop(paste0('The number of columns in newdata must be equal to ', p))
  
  call <- pwbart
  
  return(call(target_time, newdata, object, mc.cores=mc.cores, ...))
}