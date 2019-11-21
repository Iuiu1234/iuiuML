require(Metrics)
require(caret)
require(data.table) # data manipulation
#require(dplyr) # data manipulation
require(reshape2) # data manipulation
require(lubridate) # dates manipulation
require(dummies) # data manipulation
require(doParallel) # parallelization
require(AUC) # visualisation
require(ggplot2) # visualisation
require(ggpubr) # visualisation
require(plotly) # visualisation
require(GGally) # visualisation
require(caret) # modelling
require(xgboost) # modelling
require(tensorflow) # modelling
require(e1071) # modelling
require(yaml)
# require(kknn) # modelling
require(FNN) # modelling
require(measurements)
require(ggExtra)
require(DataExplorer)

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
  if(typeof(pred) == 'logical') 
    pred = ifelse(pred,'X1','X0')
  
  pred = factor(pred)
  A <- caret::confusionMatrix(pred,data,positive)
  if(print){
  print(paste0('Positive = ',positive))
  print(A$table)
  print(A$byClass[c('Sensitivity','Specificity','Precision')])}
  invisible(A)
}

# Wrapper function of caret::confusionMatrix with AUC
confusionMatrix3 = function(pred,data, show.best.roc = T){
  if(class(data) %in% c('numeric','integer')) data = data >= 1
  if(class(data) == 'logical') data = make.names(as.numeric(data))
  data <- factor(data)
  rocRR <- AUC::roc(pred,data)
  cat('auc = ',AUC::auc(rocRR))
  bestroc <- best.auc(rocRR)
  if(show.best.roc) {
  cat('\n')
  cat('Prob = ',bestroc[1], '(Best ROC)\n')
  confusionMatrix2(pred > bestroc, data)
  }
  cat('\n')
  cat('Prob = 0.5\n')
  confusionMatrix2(pred > 0.5, data)
  cat('\n')
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


confusionTable = function(pred, data, type = 'Classification', ...) {
  stopifnot(type %in% c('Classification', 'Regression'))
  if(type == 'Classification') {
    confusionMatrix3(pred,data,...)
  } else if(type == 'Regression') {
    confusionRegression(pred,data, ...)
  } else {
    stop ('type not found')
  }
  
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

create.dictionary = function(A, nn = 300, sampling  =T, set.seed_nn) {
  if(!missing(set.seed_nn)) set.seed(set.seed_nn)
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
  ff = function(x) length(unique(na.omit(x)))
  out = sapply(A,ff)
  names(out)[out > nn | out == 1]
}

xgb.num.nonames = function(A) xgb.char.nonames(A,Inf)

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
  suffix.tmp = paste0('_',Sys.time2(),'_tmp',suffix)
  file.tmp = gsub(suffix,suffix.tmp,file,fixed=T)
  save(...,file = file.tmp)
  file.copy(file.tmp,file,T)
  invisible()
}

xgb.save2 = function(model,fname) {
  suffix = last(strsplit(fname,'.',fixed=T)[[1]])
  suffix = paste0('.',suffix)
  suffix.tmp = paste0('_',Sys.time2(),'_tmp',suffix)
  fname.tmp = gsub(suffix,suffix.tmp,fname,fixed=T)
  xgb.save(model, fname.tmp)
  file.copy(fname.tmp,fname,T)
  invisible()

}

saveRDS2 = function(object, file,... ){
  suffix = last(strsplit(file,'.',fixed=T)[[1]])
  suffix = paste0('.',suffix)
  suffix.tmp = paste0('_',Sys.time2(),'_tmp',suffix)
  file.tmp = gsub(suffix,suffix.tmp,file,fixed=T)
  saveRDS(object,file = file.tmp,...)
  file.copy(file.tmp,file,T)
  invisible()
}

fwrite2 = function(x,file,...) {
  suffix = get.fileextension(file)
  suffix.tmp = paste0('_',Sys.time2(),'_tmp',suffix)
  file.tmp = gsub(suffix,suffix.tmp,file,fixed=T)
  fwrite(x, file.tmp, ...)
  file.copy(file.tmp,file,T)
  invisible()
}

ggsave2 = function(plot,filename,...) {
  suffix = get.fileextension(filename)
  suffix.tmp = paste0('_',Sys.time2(),'_tmp',suffix)
  filename.tmp = gsub(suffix,suffix.tmp,filename,fixed=T)
  ggsave(filename.tmp, plot, ...)
  file.copy(filename.tmp,filename,T)
  if(interactive()) browseURL(filename)
  print(plot)
  invisible()
 }

# wrapper for xgb.plot.importance
xgb.importance.plot = function(feature_names = names_xgb, model = model_xgb,
  nn_col = Inf, filename.plot = './output/xgb_importance_plot.png' ,show.plot = T, save.matrix = T,
  filename.matrix, width = 6, height = 5 ) {
  imp_matrix = xgb.importance(feature_names = feature_names, model = model)[order(Gain,decreasing = T)]
  g1 = ggplot(imp_matrix[1:min(.N,nn_col)],aes(reorder(Feature, Gain, FUN = max), Gain, fill = Feature)) +
  geom_col() + coord_flip() +
  theme(legend.position = "none",text = element_text(size=25)) + labs(x = "Features", y = "Importance")
  ggsave2(g1,filename.plot, width = width, height = height)
  #if(show.plot) if(interactive()) browseURL(filename.plot)
  if(save.matrix) {
    if(missing(filename.matrix)) {
      filename.matrix = gsub(get.fileextension(filename.plot),
        '_matrix.csv',
        filename.plot) }
    fwrite(imp_matrix,filename.matrix)
  }
}

# xgb cv wfor random search
xgb.cv.rs = function(train_xgb, params_xgb_ff, sign_error = -1, nn_iterations = 10,
  best_error = -Inf, ...) {

stopifnot(class(params_xgb_ff()) == 'list')
  stopifnot(class(train_xgb) == "xgb.DMatrix")
for( i in 1:nn_iterations){
seed_number = sample.int(1000,1)[[1]]
set.seed(seed_number)


#watchlist <- list(train=dtest)
params_xgb = params_xgb_ff()
# cv
cv_xgb = xgb.cv(params = params_xgb,
                   data = train_xgb,
                    ...)
                   #watchlist = watchlist,


  test_error_name = grep2(c('test','mean'),names(cv_xgb$evaluation_log))

    # sign is used is here (so max_aux has already the correct sign)
  max_error = max(sign_error*cv_xgb$evaluation_log[, test_error_name,with=F])
  max_error_index = cv_xgb$best_iteration
  test_error_name = gsub('mean','std',test_error_name)
  max_error_std = max(cv_xgb$evaluation_log[max_error_index, test_error_name, with = F])

  if (max_error > best_error){
    best_error = max_error
    best_error.index = max_error_index
    best_error.std = max_error_std
    best_seed = seed_number
    best_params = params_xgb
  }
  cat(paste0('cv ' ,i,', best ',best_params$eval_metric,' ', best_error,
    ' (', round(1.96*best_error.std,4), ') \n'))
  print(cv_xgb$evaluation_log[max_error_index])
}
  list(best_error = best_error, best_error.index = best_error.index,
    best_seed = best_seed, best_params = best_params, best_error.std = best_error.std)
}
# xgb.cv.rs(train_xgb,params_xgb,
#                   nn_iterations = 2,
#                   nrounds = 200,
#                   nfold = 5,
#                   early_stopping_rounds = 20,
#                    print_every_n = 20,
#                   verbose = F)

xgb.cv.yml = function(data_xgb, args_cv_xgb,  ...) {
  stopifnot(class(data_xgb) == "xgb.DMatrix")
  stopifnot(class(args_cv_xgb) == "list")
  data_xgb = list(data = data_xgb); gc_p = gc()
  # args_cv_xgb = read_yaml(params_xgb_file)
  params_xgb_ff = function(args) { 
    out = lapply(args$expresions, function(x) eval(parse(text=x)))
    out = c(out,args$constants)
    out
  }
  params_xgb = params_xgb_ff(args_cv_xgb)
  
  nn_iterations = params_xgb$nn_iterations
  best_error = -Inf
  for( i in 1:nn_iterations){
    seed_number = sample.int(1000,1)[[1]]
    set.seed(seed_number)
    
    #watchlist <- list(train=dtest)
    params_xgb = params_xgb_ff(args_cv_xgb)
    args_xgb.cv = names(as.list(args(xgb.cv)))
    args_xgb.cv2 = c( list(params = params_xgb), 
                     params_xgb[names(params_xgb) %in% args_xgb.cv], ...)
    data_xgb = c(data_xgb,args_xgb.cv2); gc_p = gc()
    cv_xgb = do.call(xgb.cv, data_xgb)
    data_xgb = data_xgb[1]; gc_p = gc()
    
    
    test_error_name = grep2(c('test','mean'),names(cv_xgb$evaluation_log))
    
    # sign is used is here (so max_aux has already the correct sign)
    max_error = max(params_xgb$sign_error*cv_xgb$evaluation_log[, test_error_name,with=F])
    max_error_index = cv_xgb$best_iteration
    test_error_name = gsub('mean','std',test_error_name)
    max_error_std = max(cv_xgb$evaluation_log[max_error_index, test_error_name, with = F])
    
    if (max_error > best_error){
      best_error = max_error
      best_error.index = max_error_index
      best_error.std = max_error_std
      best_seed = seed_number
      best_params = params_xgb
    }
    cat(paste0('cv ' ,i,', best ',best_params$eval_metric,' ', best_error,
               ' (', round(1.96*best_error.std,4), ') \n'))
    print(cv_xgb$evaluation_log[max_error_index])
    gc_p = gc()
    
  }
  list(best_error = best_error, best_error.index = best_error.index,
       best_seed = best_seed, best_params = best_params, best_error.std = best_error.std
  )
}


createDataPartition2 = function(x, k = 2, prob = 0.5, outnames) {
  if(length(prob) == 1) prob = c(prob, 1- prob)
  stopifnot(sum(prob) == 1)
  table_x = table(x)
  out = x
  for(i in 1:length(table_x)) {
    out[x == names(table_x)[i]] =
      sample(k,table_x[i],replace = T, prob = prob)
  }
  if(missing(outnames))
    out else
      outnames[out]
}

#
