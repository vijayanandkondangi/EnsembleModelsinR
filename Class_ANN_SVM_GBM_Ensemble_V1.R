#------------------------------------------------------------------------------------
# Purpose:
# 1. The objective of this script is to illustrate the iterative process of model 
# building and evaluation for achieving continuous improvement of accuracy/AUC.
# 2. The concept of ensembling of models for achieving maximum possible accuracy, is 
# also illustrated.
# 3. In the process, in the models, grid search and cross validation are also 
# used to select the best possible value of the model's hyper-parameter.
#------------------------------------------------------------------------------------

# Load the data
setwd("D:/Working Directory/DataScience/UniversalBank")
df = read.csv("UniversalBank1.csv")

# Examine the data frame
str(df)

# Remove the ID column
df = df[,2:14]

# Convert the outcome column to a factor
df$PersonalLoan = as.factor(df$PersonalLoan)

# Study the ratio of 1s and 0s in the outcome column
table(df$PersonalLoan) # 9.6 of 1, 90.4 of 0

# Partition the data in to train, test & validation data sets
train_index = sample(1:nrow(df), 0.6 * nrow(df))
testvald_index = setdiff(1:nrow(df), train_index)
train = df[train_index, ]
testvald  = df[testvald_index, ]
test_index = sample(1:nrow(testvald), 0.5 * nrow(testvald))
vald_index = setdiff(1:nrow(testvald), test_index)
test  = testvald[test_index, ]
vald  = testvald[vald_index, ]

#------------------------------------------------------------------------------------
# Baseline Model (Majority)
table(train$PersonalLoan) # Baseline Accuracy of predicting zero for all: 0.8993333
table(vald$PersonalLoan) # Baseline Accuracy of predicting zero for all: 0.912
table(test$PersonalLoan) # Baseline Accuracy of predicting zero for all: 0.91

# Overall, this is a very sparse data set with the ratio of 1 being ~10%

#------------------------------------------------------------------------------------
# Model 1: ANN Model

# Grid search based on cross validation

grid1 =  expand.grid(size = seq(1, 10, 1), decay = seq(0.1, 1, 0.1)) 
numfolds1 = trainControl(method = "cv", number = 10)

anngrid1 = train(PersonalLoan ~ ., data = train, 
                 method = "nnet", metric = "Accuracy",
                 trControl = numfolds1, tuneGrid = grid1)

anngrid1
# Max accuracy = 0.9483387 for size = 3 & decay = 0.5

annmodel1 = nnet(PersonalLoan ~ ., data = train,
                      size = 3, decay = 0.5,
                      linear.output = FALSE)

pred1 = predict(annmodel1, newdata = vald)

confusionMatrix(as.factor(ifelse(pred1>0.5, 1, 0)), vald$PersonalLoan)$overall["Accuracy"]
# Accuracy: 0.965, with a threshold of 0.5

confusionMatrix(as.factor(ifelse(pred1>0.45, 1, 0)), vald$PersonalLoan)$overall["Accuracy"]
# Accuracy: 0.963, with a threshold of 0.45

confusionMatrix(as.factor(ifelse(pred1>0.425, 1, 0)), vald$PersonalLoan)$overall["Accuracy"]
# Accuracy: 0.963, with a threshold of 0.425

print(auc(roc(vald$PersonalLoan, pred1)))
# AUC: 0.9809

# Overall, this model is a significant improvement on the baseline accuracy.

#------------------------------------------------------------------------------------
# Model 2: SVM Model

# Grid search based on cross validation

svmgrid2 = tune(svm, PersonalLoan ~ ., data = train,  
                kernel = "radial", 
                ranges=list(cost=seq(1, 10, 1), gamma=seq(0.1, 1, 0.1)))
svmgrid2
# Best parameter values are cost = 9 & gamma = 0.1

svmmodel2 = svm(PersonalLoan ~ ., data = train,
                kernel = "radial", probability = TRUE,
                cost = 9, gamma = 0.1)

pred2 = predict(svmmodel2, newdata = vald, probability = TRUE)

confusionMatrix(as.factor(ifelse((attr(pred2, "probabilities")[,2])>0.5, 1, 0)), vald$PersonalLoan)$overall["Accuracy"]
# Accuracy: 0.982, with a threshold of 0.5

confusionMatrix(as.factor(ifelse((attr(pred2, "probabilities")[,2])>0.45, 1, 0)), vald$PersonalLoan)$overall["Accuracy"]
# Accuracy: 0.982, with a threshold of 0.45

confusionMatrix(as.factor(ifelse((attr(pred2, "probabilities")[,2])>0.425, 1, 0)), vald$PersonalLoan)$overall["Accuracy"]
# Accuracy: 0.983, with a threshold of 0.425

print(auc(roc(vald$PersonalLoan, (attr(pred2, "probabilities")[,2]))))
# AUC: 0.9939

# Overall, this model is a significant improvement on the baseline accuracy.

#------------------------------------------------------------------------------------
# Model 3: GBM Model

# Grid search based on cross validation
grid3 =  expand.grid(interaction.depth = c(2, 4, 6, 8, 10),
                        n.trees = 100, 
                        shrinkage = seq(.01, 0.05,.01),
                        n.minobsinnode = 10) 

numfolds3 = trainControl(method = "repeatedcv",
                           repeats = 5,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

train_ada = train
test_ada = test
vald_ada = vald

train_ada$PersonalLoan = ifelse(train_ada$PersonalLoan == 1, "Accepted", "Rejected")
test_ada$PersonalLoan = ifelse(test_ada$PersonalLoan == 1, "Accepted", "Rejected")
vald_ada$PersonalLoan = ifelse(vald_ada$PersonalLoan == 1, "Accepted", "Rejected")


gbmgrid3 = train(PersonalLoan ~ ., data = train_ada,
                    distribution = "adaboost",
                    method = "gbm", bag.fraction = 0.5,
                    trControl = numfolds3,
                    verbose = TRUE,
                    tuneGrid = grid3,
                    metric = "ROC")

gbmgrid3

# Best performance n.trees = 100, interaction.depth = 10, 
# shrinkage = 0.04 and n.minobsinnode = 10.

pred3 = predict(gbmgrid3, newdata = vald_ada, probability = TRUE)

confusionMatrix(pred3, as.factor(vald_ada$PersonalLoan))
# Accuracy: 0.987

# Overall, this model is a significant improvement on the baseline accuracy.

#------------------------------------------------------------------------------------
# Model 4: Ensemble of the 3 preceding models (1, 2, 3)
# Identification of best threshold values for models 1 & 2
# Evaluation of performance on unseen data (i.e. test data set)

pred1E = predict(annmodel1, newdata = vald)
pred2E = predict(svmmodel2, newdata = vald, probability = TRUE)
pred3E = predict(gbmgrid3, newdata = vald_ada)

maxaccuracy = 0
bestthreshold1 = 0
bestthreshold2 = 0

for (threshold1 in seq(0, 0.6, 0.01)) {
  for (threshold2 in seq(0, 0.6, 0.01)) {
    pred4 = data.frame(ifelse(pred1E>threshold1, 1, 0), 
                       ifelse(attr(pred2E, "probabilities")[,2]>threshold2, 1, 0), 
                       ifelse(pred3E=="Accepted", 1, 0))
    
    colnames(pred4) = c("pred1E", "pred2E", "pred3E")
    pred4$pred1E = as.integer(pred4$pred1E)
    pred4$pred2E = as.integer(pred4$pred2E)
    pred4$pred3E = as.integer(pred4$pred3E)
    
    for (i in seq(1:nrow(pred4))) {
      i_classes = c(pred4[i, "pred1E"], pred4[i, "pred2E"], pred4[i, "pred3E"])
      pred4[i, "finalresult"] = i_classes[which.max(tabulate(match(i_classes, unique(i_classes))))]
    }
    accuracy = confusionMatrix(as.factor(pred4$finalresult), as.factor(vald$PersonalLoan))$overall["Accuracy"]
    if (accuracy > maxaccuracy) {
      maxaccuracy = accuracy
      bestthreshold1 = threshold1
      bestthreshold2 = threshold2
    }
  }
}

print(maxaccuracy) # 0.983
print(bestthreshold1) # 0.09
print(bestthreshold2) # 0.29

# Evaluation of performance of the ensemble on test data set
# based on thresholds identified at the previous step

pred1ET = predict(annmodel1, newdata = test)
pred2ET = predict(svmmodel2, newdata = test, probability = TRUE)
pred3ET = predict(gbmgrid3, newdata = test_ada)

pred5 = data.frame(ifelse(pred1ET>0.09, 1, 0), 
                   ifelse(attr(pred2ET, "probabilities")[,2]>0.29, 1, 0), 
                   ifelse(pred3ET=="Accepted", 1, 0))

colnames(pred5) = c("pred1ET", "pred2ET", "pred3ET")
pred5$pred1E = as.integer(pred5$pred1ET)
pred5$pred2E = as.integer(pred5$pred2ET)
pred5$pred3E = as.integer(pred5$pred3ET)


for (i in seq(1, nrow(pred5), 1)) {
  i_classes = c(pred5[i, "pred1ET"], pred5[i, "pred2ET"], pred5[i, "pred3ET"])
  pred5[i, "finalresult"] = i_classes[which.max(tabulate(match(i_classes, unique(i_classes))))]
}


confusionMatrix(as.factor(pred5$finalresult), as.factor(test$PersonalLoan))$overall["Accuracy"]
# Accuracy: 0.982

table(pred5$finalresult, test$PersonalLoan)
#     0   1
# 0 883  11
# 1   7  99

# On unseen data (test data set), 
# the ensemble model has performed very well; an Accuracy of 0.982 was obtained.

#------------------------------------------------------------------------------------


