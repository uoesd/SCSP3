set.seed(0)
library(caret)
library(dplyr)
library(ggplot2)
# Loading functions
source("stylometryfunctions.R")

build_dataset <- function(features_list) {
  X <- do.call(rbind, features_list)
  y <- rep(seq_along(features_list), sapply(features_list, nrow))
  list(X = X, y = y)
}

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

# Test

features_test_m <- do.call(rbind, features_test)
pred_da_t <- discriminantCorpus(features_train, features_test_m)
truth_da_t <- rep(1:length(features_test), sapply(features_test, nrow))
acc_da_t <- sum(pred_da_t  == truth_da_t) / length(truth_da_t)
acc_da_t
print(confusionMatrix(as.factor(pred_da_t), as.factor(truth_da_t)))

# LOOCV for DA

predictions_da_cv <- NULL
truth_da_cv <- NULL

for (i in 1:length(features_train)) {
  for (j in 1:nrow(features_train[[i]])) {
    
    testdata_da_cv <- matrix(features_train[[i]][j,], nrow=1)
    
    traindata_da_cv <- features_train
    traindata_da_cv[[i]] <- traindata_da_cv[[i]][-j,,drop=FALSE]
    
    pred_da_cv <- discriminantCorpus(traindata_da_cv, testdata_da_cv)
    predictions_da_cv <- c(predictions_da_cv, pred_da_cv)
    
    truth_da_cv <- c(truth_da_cv, i)
  }
}

acc_da_cv  <- sum(predictions_da_cv  == truth_da_cv) / length(truth_da_cv)
acc_da_cv

print(confusionMatrix(as.factor(predictions_da_cv), as.factor(truth_da_cv)))

# KNN

# 10-fold CV analysis for k

dataset <- build_dataset(features_train)
X <- dataset$X
y <- dataset$y

K <- 10
fold_id <- sample(rep(1:K, length.out = nrow(X)))

k_analysis <- numeric(20)

for (k in 1:20) {
  
  preds <- character(length(y))
  
  for (fold in 1:K) {
    
    test_idx  <- which(fold_id == fold)
    train_idx <- which(fold_id != fold)
    
    train_X <- X[train_idx, , drop = FALSE]
    train_y <- y[train_idx]
    test_X  <- X[test_idx, , drop = FALSE]
    
    # predict the whole fold at once
    preds[test_idx] <- myKNN(train_X, test_X, train_y, k)
  }
  
  k_analysis[k] <- mean(preds == y)
}

k_analysis

# Test

train_ds <- build_dataset(features_train)
train_X <- train_ds$X
train_y <- train_ds$y
test_X <- do.call(rbind, features_test)
test_pred_knn <- myKNN(train_X, test_X, train_y, k = 2)
test_y_knn <- rep(seq_along(features_test), sapply(features_test, nrow))
acc_knn_t <- mean(test_pred_knn == test_y_knn)
acc_knn_t
cm_knn <- confusionMatrix(as.factor(test_pred_knn), as.factor(test_y_knn))

knitr::kable(as.data.frame(cm_knn$table), caption = "Confusion Matrix")

confusionMatrix(as.factor(test_pred_knn), as.factor(test_y_knn))


# 
# train_data <- features_train
# test_data <- do.call(rbind, features_test)
# test_labels <- rep(seq_along(features_test), sapply(features_test, nrow))
# test_pred_knn_corpus <- KNNCorpus(train_data, test_data)
# library(caret)
# 
# cm_knn_corpus <- confusionMatrix(
#   as.factor(test_pred_knn_corpus),
#   as.factor(test_labels)
# )
# 
# cm_knn_corpus





# LOOCV for KNN
dataset <- build_dataset(features_train)
X <- dataset$X
y <- dataset$y

preds <- character(nrow(X))   # <-- FIXED TYPE

for (i in seq_len(nrow(X))) {
  
  train_X <- X[-i, , drop = FALSE]
  train_y <- y[-i]
  test_X  <- X[i, , drop = FALSE]
  
  preds[i] <- as.character(myKNN(train_X, test_X, train_y, k = 2))
}

loo_accuracy <- mean(preds == as.character(y))

## Report

# ---- confusion matrix object ----
cm_knn_loo <- confusionMatrix(as.factor(preds), as.factor(y))

# ---- extract key metrics into a table ----
knn_loo_table <- data.frame(
  Metric = c("Accuracy", "Kappa", "Sensitivity (Human recall)",
             "Specificity (AI recall)", "Balanced Accuracy"),
  Value = c(
    cm_knn_loo$overall["Accuracy"],
    cm_knn_loo$overall["Kappa"],
    cm_knn_loo$byClass["Sensitivity"],
    cm_knn_loo$byClass["Specificity"],
    cm_knn_loo$byClass["Balanced Accuracy"]
  )
)

knn_loo_table
knn_loo_confusion <- as.data.frame(cm_knn_loo$table)
knn_loo_confusion

