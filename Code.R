set.seed(0)
library(caret)
library(dplyr)
library(ggplot2)
library(tidyr)
library(gridExtra)
library(grid)
# Loading functions
source("stylometryfunctions.R")

build_dataset <- function(features_list) {
  X <- do.call(rbind, features_list)
  y <- rep(seq_along(features_list), sapply(features_list, nrow))
  list(X = X, y = y)
}


loadCorpus <- function(filedir,featureset="functionwords",maxauthors=Inf) {
  authornames <- list.files(filedir)
  booknames <- list()
  features <- list()
  count <- 0
  
  for (i in 1:length(authornames)) {
    #print(i)
    if (count >= maxauthors) {break}
    files <- list.files(sprintf("%s%s/",filedir,authornames[i]))
    if (length(files)==0) {next}
    
    firstbook <- FALSE
    booknames[[i]] <- character()
    for (j in 1:length(files)) {
      path <- sprintf("%s%s/%s",filedir,authornames[i],files[j])
      
      fields <- strsplit(files[j],split=' --- ')[[1]]  
      
      if (sprintf("%s.txt",featureset) == fields[2]) {
        booknames[[i]] <- c(booknames[[i]], fields[1])
        count <- count+1
        M <- as.matrix(read.csv(path,sep=',',header=FALSE))  
        if (firstbook == FALSE) {
          firstbook <- TRUE
          features[[i]] <- M
        } else {
          features[[i]]  <- rbind(features[[i]],M)
        }
        
      }
    }
  }
  return(list(features=features,booknames=booknames,authornames=authornames))
}

discriminantCorpus <- function(traindata, testdata) {
  thetas <- NULL
  preds <- NULL
  
  #first learn thea model for each aauthor
  for (i in 1:length(traindata)) {
    words <- apply(traindata[[i]],2,sum)
    
    #some words might never occur. This will be a problem since it will mean the theta for this word is 0, which means the likelihood will be 0 if this word occurs in the training set. So, we force each word to occur at leats once
    inds <- which(words==0) 
    if (length(inds) > 0) {words[inds] <- 1}
    thetas <- rbind(thetas, words/sum(words))
  }
  
  #now classify
  for (i in 1:nrow(testdata)) {
    probs <- NULL
    for (j in 1:nrow(thetas)) {
      probs <- c(probs, dmultinom(testdata[i,],prob=thetas[j,],log=TRUE))
    }
    preds <- c(preds, which.max(probs))
  }
  return(preds)
}


myKNN <- function(traindata, testdata, trainlabels, k=1) {
  if (mode(traindata) == 'numeric' && !is.matrix(traindata)) {
    traindata <- matrix(traindata,nrow=1)
  }
  if (mode(testdata) == 'numeric' && !is.matrix(testdata)) {
    testdata <- matrix(testdata,nrow=1)
  }
  
  mus <- apply(traindata,2,mean) 
  sigmas <- apply(traindata,2,sd)
  
  for (i in 1:ncol(traindata)) {
    traindata[,i] <- (traindata[,i] - mus[i])/sigmas[i]
  }
  
  for (i in 1:ncol(testdata)) {
    testdata[,i] <- (testdata[,i]-mus[i])/sigmas[i]
  }
  
  preds <- knn(traindata, testdata, trainlabels, k)
  return(preds)
}

discriminantCorpus <- function(traindata, testdata) {
  thetas <- NULL
  preds <- NULL
  
  #first learn thea model for each aauthor
  for (i in 1:length(traindata)) {
    words <- apply(traindata[[i]],2,sum)
    
    #some words might never occur. This will be a problem since it will mean the theta for this word is 0, which means the likelihood will be 0 if this word occurs in the training set. So, we force each word to occur at leats once
    inds <- which(words==0) 
    if (length(inds) > 0) {words[inds] <- 1}
    thetas <- rbind(thetas, words/sum(words))
  }
  
  #now classify
  for (i in 1:nrow(testdata)) {
    probs <- NULL
    for (j in 1:nrow(thetas)) {
      probs <- c(probs, dmultinom(testdata[i,],prob=thetas[j,],log=TRUE))
    }
    preds <- c(preds, which.max(probs))
  }
  return(preds)
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

# normalization (train)
normalize_rows <- function(mat) mat / rowSums(mat)
features_train_norm <- lapply(features_train, normalize_rows)

# Basic Information of The Training Data
eda_X <- do.call(rbind, features_train_norm)
eda_y <- rep(c("Human", "LLM"), sapply(features_train_norm, nrow))

df <- data.frame("Number of Texts" = c(nrow(features_train_norm[[1]]), nrow(features_train_norm[[2]])),
                 "Function Words" = c(ncol(eda_X)-1, ncol(eda_X)-1), 
                 "Feature for other words" = c(1,1),
                 row.names = c("Human", "LLM"),
                 check.names = FALSE)

T1 <- knitr::kable(df, caption = "Table 1: Basic Information of the Training Data")

# Top 10 Mean Difference in Function Words Between LLM and Human Texts
human_mean <- colMeans(features_train_norm[[1]])
llm_mean   <- colMeans(features_train_norm[[2]])

mean_diff <- llm_mean - human_mean
top <- order(abs(mean_diff), decreasing = TRUE)[1:10]

barplot(mean_diff[top], ylab = "Mean Difference")

# MDS Visualisation of Human and LLM Texts
dist_mat <- dist(eda_X)
mds_res  <- cmdscale(dist_mat)

plot(mds_res[,1], mds_res[,2],
     col = ifelse(eda_y == "Human", "blue", "red"),
     pch = 16,
     cex = 0.6,
     xlab = "MDS1",
     ylab = "MDS2")

human_cent <- colMeans(mds_res[eda_y == "Human", ])
llm_cent   <- colMeans(mds_res[eda_y == "LLM", ])

points(human_cent[1], human_cent[2], pch = 4, cex = 2, lwd = 2, col = "blue")
points(llm_cent[1], llm_cent[2], pch = 4, cex = 2, lwd = 2, col = "Black")

legend("topright",
       legend = c("Human texts", "LLM texts", "Human centroid", "LLM centroid"),
       col = c("blue", "red", "blue", "Black"),
       pch = c(16, 16, 4, 4),
       pt.cex = c(0.8, 0.8, 2, 2),
       bty = "n")

# Randomly split the whole dataset to train data(80%) and test data(20%)
split_idx <- lapply(features, function(mat) {
  sample(1:nrow(mat), size = floor(0.8 * nrow(mat)))
})

features_train_Authors <- mapply(function(mat, idx) {
  mat[idx, ]
}, features, split_idx, SIMPLIFY = FALSE)

features_test_Authors <- mapply(function(mat, idx) {
  mat[-idx, ]
}, features, split_idx, SIMPLIFY = FALSE)

# MDS Visualisation of Authors
x <- NULL
for (i in 1:length(features_train_Authors)) {
  x <- rbind(x, apply(features_train_Authors[[i]], 2, sum))
}

for (i in 1:nrow(x)) {
  x[i,] <- x[i,] / sum(x[i,])
}

for (j in 1:ncol(x)) {
  x[,j] <- (x[,j]- mean(x[,j]))/sd(x[,j])
}

d <- dist(x)
pts <- cmdscale(d)

plot(pts, 
     type="n", 
     xlim = c(min(pts[,1]) - 2, max(pts[,1]) + 1), 
     ylim = c(min(pts[,2]) - 1, max(pts[,2]) + 1))

text(pts[,1], pts[,2], labels=authors)

# DA

# Test
# Make test data matrix and test
features_test_matrix <- do.call(rbind, features_test)
pred_da_t <- discriminantCorpus(features_train, features_test_matrix)

# Calculate accuracy and make confusion matrix
truth_da_t <- rep(1:length(features_test), sapply(features_test, nrow))
acc_da_t <- sum(pred_da_t  == truth_da_t) / length(truth_da_t)
cm_da_t <- confusionMatrix(as.factor(pred_da_t), as.factor(truth_da_t))

# Reporting table
da_test_table <- data.frame(
  Value = c(
    cm_da_t$overall["Accuracy"],
    cm_da_t$overall["Kappa"],
    cm_da_t$byClass["Sensitivity"],
    cm_da_t$byClass["Specificity"],
    cm_da_t$byClass["Balanced Accuracy"],
    cm_da_t$overall["AccuracyLower"],
    cm_da_t$overall["AccuracyUpper"]
  )
)

# LOOCV
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

# Calculate accuracy and make confusion matrix
acc_da_cv  <- sum(predictions_da_cv  == truth_da_cv) / length(truth_da_cv)
cm_da_cv  <- confusionMatrix(as.factor(predictions_da_cv), as.factor(truth_da_cv))

# Reporting table
da_loo_table  <- data.frame(
  Value = c(
    cm_da_cv$overall["Accuracy"],
    cm_da_cv$overall["Kappa"],
    cm_da_cv$byClass["Sensitivity"],
    cm_da_cv$byClass["Specificity"],
    cm_da_cv$byClass["Balanced Accuracy"]
  )
)

# KNN

# 10-fold CV analysis for k

# Take the value of train points from list (myKNN needs matrix form) 
dataset <- build_dataset(features_train)
X <- dataset$X
y <- dataset$y

# Do 10-fold CV to find best k based on accuracy, balanced accuracy, sensitivity.
K <- 10
fold_id <- sample(rep(1:K, length.out = nrow(X)))

acc_k_analysis <- numeric(20)
balacc_k_analysis <- numeric(20)
sen_k_analysis <- numeric(20)

for (k in 1:20) {
  
  preds_k_analysis <- character(length(y))
  
  for (fold in 1:K) {
    
    test_idx  <- which(fold_id == fold)
    train_idx <- which(fold_id != fold)
    
    train_X <- X[train_idx, , drop = FALSE]
    train_y <- y[train_idx]
    test_X  <- X[test_idx, , drop = FALSE]
    
    preds_k_analysis[test_idx] <- myKNN(train_X, test_X, train_y, k)
  }
  
  acc_k_analysis[k] <- mean(preds_k_analysis == y)
  cm_k_analysis <- confusionMatrix(as.factor(preds_k_analysis), as.factor(y))
  balacc_k_analysis[k] <- cm_k_analysis$byClass["Balanced Accuracy"]
  sen_k_analysis[k] <- cm_k_analysis$byClass["Sensitivity"]
}

## LOOCV with k=2

preds_knn_loo <- character(nrow(X))   # <-- FIXED TYPE

for (i in seq_len(nrow(X))) {
  
  train_X <- X[-i, , drop = FALSE]
  train_y <- y[-i]
  test_X  <- X[i, , drop = FALSE]
  
  preds_knn_loo[i] <- as.character(myKNN(train_X, test_X, train_y, k = 2))
}

acc_knn_loo <- mean(preds_knn_loo == as.character(y))
cm_knn_loo <- confusionMatrix(as.factor(preds_knn_loo), as.factor(y))

# Reporting table
knn_loo_table <- data.frame( 
  Value = c(
    cm_knn_loo$overall["Accuracy"],
    cm_knn_loo$overall["Kappa"],
    cm_knn_loo$byClass["Sensitivity"],
    cm_knn_loo$byClass["Specificity"],
    cm_knn_loo$byClass["Balanced Accuracy"]
  )
)

# Reporting plot for k analysis

k_vals <- 1:20

# Dataframe for plotting
plot_df <- data.frame(
  k = k_vals,
  Accuracy = acc_k_analysis,
  Balanced_Accuracy = balacc_k_analysis,
  Sensitivity = sen_k_analysis)

# Name the variables
plot_long <- plot_df %>%
  pivot_longer(-k, names_to = "Metric", values_to = "Score")

knn_cv_plot <- ggplot(plot_long, aes(x = k, y = Score,
                                     shape = Metric, linetype = Metric)) +
  geom_line(size = 0.8) +
  geom_point(size = 2) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "k (number of neighbors)",
       y = "Score") +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "right",
    legend.title = element_blank(),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )






### Bonus

M <- loadCorpus("Data/FunctionWords/", "frequentwords")
authors <- M$authornames
features <- M$features

# find observations
human_index  <- which(authors == "Human")
gpt_index    <- which(authors == "GPT")
gemini_index <- which(authors == "Gemini")

# train human and gpt data
train_data <- list(
  features[[human_index]],
  features[[gpt_index]]
)


# find parameters

estimate_theta <- function(mat){
  word_counts <- colSums(mat)
  word_counts[word_counts==0] <- 1
  theta <- word_counts / sum(word_counts)
  return(theta)
}

theta_human <- estimate_theta(train_data[[1]])
theta_gpt   <- estimate_theta(train_data[[2]])


# test Gemini data
testdata <- features[[gemini_index]]


# modelling with multivariate distributions

log_h <- apply(testdata, 1, function(row) dmultinom(row, prob = theta_human, log = TRUE))
log_g <- apply(testdata, 1, function(row) dmultinom(row, prob = theta_gpt,   log = TRUE))

log_diff <- log_h - log_g

#plot
summary(log_diff)
hist(log_diff, main="Gemini log-likelihood difference", xlab="log p(Human) - log p(GPT)")
abline(v=0)






#######
Gemini <- features[[1]]
GPT <- features[[2]]
Human <- features[[3]]
Llama <- features[[4]]

# Human vs GPT: Test on Gemini
pred_g_h_ge <- discriminantCorpus(list(GPT, Human), Gemini)
table(pred_g_h_ge)

# Human vs GPT: Test on Llama
pred_g_h_l <- discriminantCorpus(list(GPT, Human), Llama)
table(pred_g_h_l)

# Human vs Gemini: Test on GPT
pred_ge_h_g <- discriminantCorpus(list(Gemini, Human), GPT)
table(pred_ge_h_g)

# Human vs Gemini: Test on Llama
pred_ge_h_l <- discriminantCorpus(list(Gemini, Human), Llama)
table(pred_ge_h_l)

# Human vs Llama: Test on GPT
pred_l_h_g <- discriminantCorpus(list(Llama, Human), GPT)
table(pred_l_h_g)

# Human vs Llama: Test on Gemini
pred_l_h_ge <- discriminantCorpus(list(Llama, Human), Gemini)
table(pred_l_h_ge)



summary_table <- data.frame(
  Train = c("GPT vs Human",
            "GPT vs Human",
            "Gemini vs Human",
            "Gemini vs Human",
            "Llama vs Human",
            "Llama vs Human"),
  Test = c("Gemini",
           "Llama",
           "GPT",
           "Llama",
           "GPT",
           "Gemini"),
  Prop_Class_AI = c(
    prop.table(table(pred_g_h_ge))[1],
    prop.table(table(pred_g_h_l))[1],
    prop.table(table(pred_ge_h_g))[1],
    prop.table(table(pred_ge_h_l))[1],
    prop.table(table(pred_l_h_g))[1],
    prop.table(table(pred_l_h_ge))[1]
  )
)

summary_table


#
library(ggplot2)

ggplot(summary_table,
       aes(x=Test, y=Prop_Class_AI, group=Train, color=Train)) +
  geom_line(size=1.2) +(base_size=14)
  geom_point(size=3) +
  ylim(0,1) +
  labs(title="Classification Proportion under Different Training Models",
       x="Tested AI",
       y="Proportion classified as AI class") +
  theme_minimal(base_size = 14)
