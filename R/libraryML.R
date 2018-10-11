library(Metrics)
library(caret)
library(data.table) # data manipulation
#library(dplyr) # data manipulation
library(reshape2) # data manipulation
library(lubridate) # dates manipulation
library(dummies) # data manipulation
library(doParallel) # parallelization
library(AUC) # visualisation
library(ggplot2) # visualisation
library(ggpubr) # visualisation
library(plotly) # visualisation
library(GGally) # visualisation
library(caret) # modelling 
library(xgboost) # modelling
library(tensorflow) # modelling
# library(kknn) # modelling
library(FNN) # modelling
library(measurements)
library(ggExtra)
library(DataExplorer)

# Error metrics ---------------------------------------------------------------------------------
# Wrapper function of caret::confusionMatrix -----------------------------------------------------
best.auc <- function(x){
stopifnot(class(x)[2] =='roc')
tt <- -pi/4
R <- matrix(c(cos(tt),-sin(tt),sin(tt),cos(tt)),nrow=2)
x2 <- matrix(c(x$fpr,x$tpr),nrow=length(x$tpr))%*%R
x$cutoffs[which.max(x2[,2])]
}

get.cutoff <- function(x,y,z = 'tpr'){
  if(missing(x)) cat('x <- AUC::roc(pred,data)')
  # x <- AUC::roc(pred,data)
  stopifnot(class(x)[2] =='roc')
  stopifnot(z %in% c('tpr','tnr'))
  if(z == 'tnr'){
    z <- 'fpr'
    y <- 1 - y
  }
  out <- x$cutoffs[x[[z]] > y]
 ifelse(z == 'tpr', max(out),min(out) )
 max(out)
}

get.cutoff2 <- function(A,...){
  stopifnot(c('pred','y') %in% names(A))
  A$y <- factor(A$y)
  x <- AUC::roc(A$pred,A$y)
  get.cutoff(x,...)
}

# Ggplots Sensitivity vs Specificity and Sensitivity vs Precision
ggroc <- function(...,OnlyAUC = F,OnlyPr = F){
  A <- do.call(rbind,list(...))
  thereis.group <- 'group' %in% names(A)
  if(!thereis.group) A$group <- 'a'
  stopifnot(c('pred','y') %in% names(A))
  stopifnot(A$y %in% c('X0','X1'))
  
  A <- A[order(pred,decreasing = T)]
  A <-   A[,
  .(pred,y,
    TP = cumsum(y == 'X1'),
    FP = cumsum(y == 'X0'),
    P = 1:.N,
    N = .N:1,
    T = sum(y == 'X1'),
    F = sum(y == 'X0')),
    group]
  
  A$TPR <- A$TP/A$T
  A$FPR <- A$FP/A$F
  A$PPV <- A$TP/A$P
  A <- A[,.SD[sample(.N,min(.N,10^4))],group] # We don't need all the rows for the plot (time consuming)
  
  g1 <- ggplot(A,aes(FPR,TPR,col = group)) + geom_line() + geom_abline(slope=1) + 
    xlab('1 - Specificity') + ylab('Sensitivity') + theme(legend.title = element_blank())
  g2 <- ggplot(A,aes(PPV,TPR,col = group)) + geom_line() + 
    xlab('Precision') + ylab('Sensitivity') 
  if(thereis.group) {
  g3 <- ggarrange(g1,g2,align='v',legend='right',common.legend = T)
  } else {
  g3 <-  ggarrange(g1,g2,align='v',legend='none',common.legend = F)
  }
  invisible(list(g1,g2,g3))
  }

confusionMatrix2 <- function(pred,data,positive = 'X1', print = T) {
  stopifnot(positive=='X1')
  if(typeof(pred) == 'logical') pred <- ifelse(pred,'X1','X0')
  A <- confusionMatrix(pred,data,positive)
  if(print){
  print(paste0('Positive = ',positive))
  print(A$table)
  print(A$byClass[c('Sensitivity','Specificity','Precision')])}
  invisible(A)
}

# Wrapper function of caret::confusionMatrix with AUC
confusionMatrix3 = function(pred,data, group = NULL,sinkfilename = NULL){
  if(!is.null(group)) {
    if(is.null(sinkfilename))
      sinkfilename = paste0('./output/', filenameout, 'confusionmatrix_',group,'.txt')
    sink(sinkfilename, split = T)
    }
  if(class(data) %in% c('numeric','integer')) data = data >= 1
  if(class(data) == 'logical') data = make.names(as.numeric(data))
  data <- factor(data)
  rocRR <- AUC::roc(pred,data)
  cat('auc = ',AUC::auc(rocRR))
  bestroc <- best.auc(rocRR)
  
  cat('\n')
  cat('Prob = ',bestroc[1], '(Best ROC)\n')
  confusionMatrix2(pred > bestroc, data)
  cat('\n')
  cat('Prob = 0.5\n')
  confusionMatrix2(pred > 0.5, data)
  cat('\n')
  if(!is.null(group)) {
    sink()
    file.copy.out = file.copy(sinkfilename,add.Sys.time2(sinkfilename))
  }
}

confusionRegression = function(data,pred) {
  out = caret::postResample( data, pred)
  out = c(out,RRSE = Metrics::rrse( data, pred))
  options( warn = -1 )
  out = c(out,cor.test( data, pred, method = 'spearman')$estimate)
  options( warn = 0 )
  out = c(out, MAE = Metrics::mae( data, pred))
  #print(out)
  out
}

# Wrapper of scale such that only scales certain selected features & returns output as data.table ---------
scale2 <- function(x,NO_scale = 'y',...){
  out <- data.table(
    scale(x[,-NO_scale,with=F],...),
    x[,NO_scale,with=F])
  out[,names(x),with=F]
  }

# Wrapper to get probability of the class of the FNN output
FNN.prob <- function(x) {
  stopifnot(levels(x) %in% c('X0','X1'))
  delta <- as.numeric(x == 'X1')
  delta + (-2*delta + 1) * attributes(x)$prob
}



# xgb data preparation ========================================================================
# A = data.table(a = letters[1:4],b= as.character(1:4),c=1:4,d=Sys.Date() + 1:4, e = Sys.time() + 1:4)
# cclass = sapply(A,class)
# cclass = grep('character',cclass,value = T)
# cclass = names(cclass)
# nonames = xgb.char.nonames(A[,cclass,with = F])
# if(length(nonames) > 0) A = A[,-nonames,with=F]
# dictionary = create.dictionary(A[,cclass,with = F])
# lapply2(A,as.numeric,names(A)[!names(A) %in% cclass])
# A = xgb.as.numeric(A,dictionary)

create.dictionary = function(A, nn = 300, sampling  =T) {
  ff = function(out) {
    out = unique(out)
    if(length(out) > nn ) out = NA
    if(sampling) out = sample(out)
    out
  }
  lapply(A,ff)
}
#dictionary = create.dictionary(A)


xgb.as.numeric = function(A, dictionary) {
  nnames = names(A)
  no_nnames = !names(A) %in% names(dictionary)
  ff = function(n, x) {
        as.numeric(factor(x, levels = unlist(dictionary[n])))
  }
  out = mapply(ff, names(A)[!no_nnames], A[,!no_nnames,with=F], SIMPLIFY=FALSE)
  out = as.data.table(out)
  if(any(no_nnames)) out = cbind(out,A[,no_nnames,with = F])
  out = out[,nnames,with = F]
  out
}

xgb.char.nonames = function(A, nn = 300) {
  ff = function(x) length(unique(x))
  out = sapply(A,ff)
  names(out)[out > nn | out == 0]
}

xgb.as.character = function(A, dictionary) {
  nnames = names(A)
  no_nnames = !names(A) %in% names(dictionary)
  ff = function(n, x) {
        unlist(dictionary[n])[x]
  }
  out = mapply(ff, names(A)[!no_nnames], A[,!no_nnames,with=F], SIMPLIFY=FALSE)
  out = as.data.table(out)
  if(any(no_nnames)) out = cbind(out,A[,no_nnames,with = F])
  out = out[,nnames,with = F]
  out
}
# A = data.table(a = letters[1:4],b= as.character(1:4),c=1:4,d=Sys.Date() + 1:4, e = Sys.time() + 1:4)
# cclass = sapply(A,class)
# cclass = grep('character',cclass,value = T)
# cclass = names(cclass)
# nonames = xgb.char.nonames(A[,cclass,with = F])
# if(length(nonames) > 0) A = A[,-nonames,with=F]
# dictionary = create.dictionary(A[,cclass,with = F])
# lapply2(A,as.numeric,names(A)[!names(A) %in% cclass])
# A = xgb.as.numeric(A,dictionary)


# ==================================================================================================
cartesian = function(lat,lon) {
  lat = conv_unit(lat,'degree','radian')
  lon = conv_unit(lon,'degree','radian')
x = cos(lat) * cos(lon)
y = cos(lat) * sin(lon)
z = sin(lat) 
return(list(x = x, y = y, z = z))
}

sampling = function(pp, logical = F) {
ff = function(pp2) sample(0:1,size = 1, replace = T, prob = c(1- pp2,pp2))
out = sapply(pp - trunc(pp),ff) + trunc(pp)
if(logical) out = as.logical(out)
out
}

# Save ---------------------------------------------------------------------------------------

save2 = function(...,file) {
  suffix = last(strsplit(file,'.',fixed=T)[[1]])
  suffix = paste0('.',suffix)
  suffix.tmp = paste0(Sys.time2(),suffix)
  file.tmp = gsub(suffix,suffix.tmp,file,fixed=T)
  save(...,file = file.tmp)
  file.copy(file.tmp,file,T)
}

xgb.save2 = function(model,fname) {
  suffix = last(strsplit(fname,'.',fixed=T)[[1]])
  suffix = paste0('.',suffix)
  suffix.tmp = paste0(Sys.time2(),suffix)
  fname.tmp = gsub(suffix,suffix.tmp,fname,fixed=T)
  xgb.save(model, fname.tmp)
  file.copy(fname.tmp,fname,T)
 
}

saveRDS2 = function(object, file,... ){
  suffix = last(strsplit(file,'.',fixed=T)[[1]])
  suffix = paste0('.',suffix)
  suffix.tmp = paste0(Sys.time2(),suffix)
  file.tmp = gsub(suffix,suffix.tmp,file,fixed=T)
  saveRDS(object,file = file.tmp,...)
  file.copy(file.tmp,file,T)
}

fwrite2 = function(x,file,...) {
  suffix = get.fileextension(file)
  suffix.tmp = paste0(Sys.time2(),suffix)
  file.tmp = gsub(suffix,suffix.tmp,file,fixed=T)
  fwrite(x, file.tmp, ...)
  file.copy(file.tmp,file,T)
}

ggsave2 = function(plot,filename,...) {
  suffix = get.fileextension(filename)
  suffix.tmp = paste0(Sys.time2(),suffix)
  filename.tmp = gsub(suffix,suffix.tmp,filename,fixed=T)
  ggsave(filename.tmp, plot, ...)
  file.copy(filename.tmp,filename,T)
 }

# wrapper for xgb.plot.importance
xgb.importance.plot = function(feature_names = names_xgb, model = model_xgb,
  nn_col = Inf, filename.plot = './output/xgb_importance_plot.png' ,show.plot = T, save.matrix = T,
  filename.matrix ) {
  imp_matrix = xgb.importance(feature_names = feature_names, model = model)[order(Gain,decreasing = T)]
  g1 = ggplot(imp_matrix[1:min(.N,nn_col)],aes(reorder(Feature, Gain, FUN = max), Gain, fill = Feature)) +
  geom_col() + coord_flip() +
  theme(legend.position = "none",text = element_text(size=25)) + labs(x = "Features", y = "Importance")
  ggsave(filename.plot,g1)
  if(show.plot) browseURL(filename.plot)
  if(save.matrix) {
    if(missing(filename.matrix)) {
      filename.matrix = gsub(get.fileextension(filename.plot),
        '_matrix.csv',
        filename.plot) }
    fwrite(imp_matrix,filename.matrix)
  }
}
