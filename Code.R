source("stylometryfunctions.R")

# Loading data
M <- loadCorpus("Data/FunctionWords/", "frequentwords")
authors <- M$authornames
features <- M$features

# Combining all AI data

human_index <- which(authors == "Human")

human_data <- features[[human_index]]

llm_data <- NULL
for (i in 1:length(features)) {
  if (i != human_index) {
    llm_data <- rbind(llm_data, features[[i]])
  }
}
features_bin <- list(human_data, llm_data)

# Randomly split to train data(80%) and test data(20%)

split_idx <- lapply(features_bin, function(mat) {
  sample(1:nrow(mat), size = floor(0.8 * nrow(mat)))
})

features_train <- mapply(function(mat, idx) {
  mat[idx, ]
}, features_bin, split_idx, SIMPLIFY = FALSE)

features_test <- mapply(function(mat, idx) {
  mat[-idx, ]
}, features_bin, split_idx, SIMPLIFY = FALSE)

# DA

features_test_m <- do.call(rbind, features_test)
pred <- discriminantCorpus(features_train, features_test_m)
truth <- rep(1:length(features_test), sapply(features_test, nrow))
test_acc <- sum(pred  == truth) / length(truth)

# LOOCV for DA

predictions_da <- NULL
truth <- NULL

for (i in 1:length(features_train)) {
  for (j in 1:nrow(features_train[[i]])) {
    
    testdata <- matrix(features_train[[i]][j,], nrow=1)
    
    traindata <- features_train
    traindata[[i]] <- traindata[[i]][-j,,drop=FALSE]
    
    pred <- discriminantCorpus(traindata, testdata)
    predictions_da <- c(predictions_da, pred)
    
    truth <- c(truth, i)
  }
}

acc_da_cv  <- sum(predictions_da  == truth) / length(truth)

# KNN

KNN <- function(traindata, testdata, k=1) {
  train <- NULL
  for (i in 1:length(traindata)) {
    train <- rbind(train, apply(traindata[[i]],2,sum))
  }
  
  for (i in 1:nrow(train)) {
    train[i,] <- train[i,]/sum(train[i,])
  }
  for (i in 1:nrow(testdata)) {
    testdata[i,] <- testdata[i,]/sum(testdata[i,])
  }
  trainlabels <- 1:nrow(train)
  myKNN(train, testdata, trainlabels,k=k)
}

# LOOCV analysis for k

k_anakysis <- NULL

for (k in 1:10){
  predictions_knn <- NULL
  truth <- NULL
  for (i in 1:length(features_train)) {
    for (j in 1:nrow(features_train[[i]])) {
      
      testdata <- matrix(features_train[[i]][j,], nrow=1)
      
      traindata <- features_train
      traindata[[i]] <- traindata[[i]][-j,,drop=FALSE]
      
      pred <- KNN(traindata, testdata, k)
      predictions_knn <- c(predictions_knn, pred)
      
      truth <- c(truth, i)
    }
  }
  acc_knn_cv <- sum(predictions_knn == truth) / length(truth)
  k_anakysis <- c(k_anakysis, acc_knn_cv)
}

k_anakysis

# 10-fold CV analysis for k

Kmax <- 10
k_analysis1 <- numeric(Kmax)

nfold <- 10

# create fold indices INSIDE each class
folds <- lapply(features_train, function(mat) {
  sample(rep(1:nfold, length.out = nrow(mat)))
})

for (k in 1:Kmax) {
  
  acc_fold <- numeric(nfold)
  
  for (f in 1:nfold) {
    
    # build training and testing lists (same structure)
    train_list <- vector("list", length(features_train))
    test_list  <- vector("list", length(features_train))
    
    for (i in 1:length(features_train)) {
      train_list[[i]] <- features_train[[i]][folds[[i]] != f, , drop = FALSE]
      test_list[[i]]  <- features_train[[i]][folds[[i]] == f, , drop = FALSE]
    }
    
    # predict all test samples in this fold
    predictions <- NULL
    truth <- NULL
    
    for (i in 1:length(test_list)) {
      for (j in 1:nrow(test_list[[i]])) {
        
        testdata <- matrix(test_list[[i]][j, ], nrow = 1)
        
        pred <- KNN(train_list, testdata, k)
        
        predictions <- c(predictions, pred)
        truth <- c(truth, i)
      }
    }
    
    acc_fold[f] <- mean(predictions == truth)
  }
  
  k_analysis1[k] <- mean(acc_fold)
}

k_analysis1




predictions_knn <- NULL
truth <- NULL
for (i in 1:length(features_train)) {
  for (j in 1:nrow(features_train[[i]])) {
    
    testdata <- matrix(features_train[[i]][j,], nrow=1)
    
    traindata <- features_train
    traindata[[i]] <- traindata[[i]][-j,,drop=FALSE]
    
    pred <- KNN(traindata, testdata, 1)
    predictions_knn <- c(predictions_knn, pred)
    
    truth <- c(truth, i)
  }
}
acc_knn_cv <- sum(predictions_knn == truth) / length(truth)
acc_knn_cv
