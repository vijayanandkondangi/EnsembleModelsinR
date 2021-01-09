#------------------------------------------------------------------------------------
# Purpose:
# 1. The objective of this script is to illustrate the iterative process of model 
# building and evaluation for achieving continuous improvement of accuracy/AUC.
# 2. The concept of ensembling of models for achieving maximum possible accuracy, is 
# also illustrated.
# 3. In the process, in one of the models, grid search and cross validation are also 
# used to select the best possible value of the model's hyper-parameter.
#------------------------------------------------------------------------------------

# Load the data
setwd("D:/Working Directory/DataScience/UniversalBank")
df = read.csv("UniversalBank.csv")

# Examine the data frame
str(df)

# Remove the ID column
df = df[,2:14]

# Convert the outcome column to a factor
df$Personal.Loan = as.factor(df$Personal.Loan)

# Study the ratio of 1s and 0s in the outcome column
table(df$Personal.Loan) # 9.6 of 1, 90.4 of 0

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
table(train$Personal.Loan) # Baseline Accuracy of predicting zero for all: 0.8976667
table(vald$Personal.Loan) # Baseline Accuracy of predicting zero for all: 0.924
table(test$Personal.Loan) # Baseline Accuracy of predicting zero for all: 0.903

# Overall, this is a very sparse data set with the ratio of 1 being ~10%

#------------------------------------------------------------------------------------
# Model 1: Logistic Regression including all predictors, family = binomial

logreg1 = glm(Personal.Loan ~., data = train, family = binomial)
summary(logreg1)

pred1 = predict(logreg1, newdata = vald, type="response")

confusionMatrix(as.factor(ifelse(pred1>0.5, 1, 0)), vald$Personal.Loan)$overall["Accuracy"]
# Accuracy: 0.959, with a threshold of 0.5

confusionMatrix(as.factor(ifelse(pred1>0.45, 1, 0)), vald$Personal.Loan)$overall["Accuracy"]
# Accuracy: 0.959, with a threshold of 0.45

confusionMatrix(as.factor(ifelse(pred1>0.425, 1, 0)), vald$Personal.Loan)$overall["Accuracy"]
# Accuracy: 0.96, with a threshold of 0.425

print(auc(roc(vald$Personal.Loan, pred1)))
# AUC: 0.9597

# Overall, this is in itself, an improvement on the baseline accuracy.

#------------------------------------------------------------------------------------
# Model 2: Random Forest

RF2 = randomForest(Personal.Loan ~ ., 
                   data = train, ntree = 500, nodesize = 5, 
                   mtry=4)

pred2 = predict(RF2, newdata = vald, type="prob")

confusionMatrix(as.factor(ifelse(pred2[,2]>0.5, 1, 0)), vald$Personal.Loan)$overall["Accuracy"]
# Accuracy: 0.989, with a threshold of 0.5

confusionMatrix(as.factor(ifelse(pred2[,2]>0.45, 1, 0)), vald$Personal.Loan)$overall["Accuracy"]
# Accuracy: 0.987, with a threshold of 0.45

confusionMatrix(as.factor(ifelse(pred2[,2]>0.425, 1, 0)), vald$Personal.Loan)$overall["Accuracy"]
# Accuracy: 0.987, with a threshold of 0.425

print(auc(roc(vald$Personal.Loan, pred2[,2])))
# AUC: 0.9967

# Overall, there is an improvement in accuracy and AUC.

#------------------------------------------------------------------------------------
# Model 3: Random Forest with Cross Validation

Grid3 = expand.grid(mtry = seq(1, 10, 1))
NumFolds3 = trainControl(method = "cv", number = 10)
RFGrid = train(Personal.Loan ~ ., data = train, method = "rf", trControl = NumFolds3, tuneGrid = Grid3)
RFGrid

#mtry = 6 returns the highest accuracy of 0.9833299

# Fitting a random forest model based on the identified mtry value (6)
RF3 = randomForest(Personal.Loan ~ ., 
                   data = train, ntree = 500, nodesize = 5, 
                   mtry=6)

pred3 = predict(RF3, newdata = vald, type="prob")

confusionMatrix(as.factor(ifelse(pred3[,2]>0.5, 1, 0)), vald$Personal.Loan)$overall["Accuracy"]
# Accuracy: 0.988, with a threshold of 0.5

confusionMatrix(as.factor(ifelse(pred3[,2]>0.45, 1, 0)), vald$Personal.Loan)$overall["Accuracy"]
# Accuracy: 0.987, with a threshold of 0.45

confusionMatrix(as.factor(ifelse(pred3[,2]>0.425, 1, 0)), vald$Personal.Loan)$overall["Accuracy"]
# Accuracy: 0.987, with a threshold of 0.425

print(auc(roc(vald$Personal.Loan, pred3[,2])))
# AUC: 0.9967

# Overall, there is NO improvement in accuracy and AUC.

#------------------------------------------------------------------------------------
# Model 4: Boosting

boost4 = boosting(Personal.Loan ~ ., train)
pred4 = predict(boost4, vald)

confusionMatrix(as.factor(ifelse(pred4$prob[,2]>0.5, 1, 0)), vald$Personal.Loan)$overall["Accuracy"]
# Accuracy: 0.986, with a threshold of 0.5

confusionMatrix(as.factor(ifelse(pred4$prob[,2]>0.45, 1, 0)), vald$Personal.Loan)$overall["Accuracy"]
# Accuracy: 0.988, with a threshold of 0.45

confusionMatrix(as.factor(ifelse(pred4$prob[,2]>0.425, 1, 0)), vald$Personal.Loan)$overall["Accuracy"]
# Accuracy: 0.99, with a threshold of 0.425

print(auc(roc(vald$Personal.Loan, pred4$prob[,2])))
# AUC: 0.9983

# Overall, there is an improvement in accuracy and AUC

#------------------------------------------------------------------------------------
# Model 5: Ensemble of the 3 preceding models (2, 3, 4)
# Evaluation of performance on unseen data (i.e. test data set)

threshold = 0.425
pred2E = predict(RF2, newdata = test, type="prob")
pred3E = predict(RF3, newdata = test, type="prob")
pred4E = predict(boost4, test)

pred5 = data.frame(ifelse(pred2E[,2]>threshold, 1, 0), ifelse(pred3E[,2]>threshold, 1, 0), ifelse(pred4E$prob[,2]>threshold, 1, 0))
colnames(pred5) = c("pred2E", "pred3E", "pred4E")

for (i in seq(1, nrow(pred5), 1)) {
  i_classes = c(pred5[i, "pred2E"], pred5[i, "pred3E"], pred5[i, "pred4E"])
  pred5[i, "finalresult"] = i_classes[which.max(tabulate(match(i_classes, unique(i_classes))))]
}


confusionMatrix(as.factor(pred5$finalresult), as.factor(test$Personal.Loan))$overall["Accuracy"]
# Accuracy: 0.988

table(as.factor(pred5$finalresult), test$Personal.Loan)
#     0   1
# 0 889   6
# 1   6  99

# On unseen data (test data set), for a threshold of 0.425, 
# the model has performed very well; an Accuracy of 0.993 was obtained.

#------------------------------------------------------------------------------------
# Model 6: Finetuning the threshold for maximizing accuracy


pred2E = predict(RF2, newdata = vald, type="prob")
pred3E = predict(RF3, newdata = vald, type="prob")
pred4E = predict(boost4, vald)
threshold = 0
accuracy = 0
maxaccuracy = 0
bestthreshold = 0

for (threshold in seq(0, 0.6, 0.0001)) {
  pred5 = data.frame(ifelse(pred2E[,2]>threshold, 1, 0), ifelse(pred3E[,2]>threshold, 1, 0), ifelse(pred4E$prob[,2]>threshold, 1, 0))
  colnames(pred5) = c("pred2E", "pred3E", "pred4E")
  for (i in seq(1, nrow(pred5), 1)) {
    i_classes = c(pred5[i, "pred2E"], pred5[i, "pred3E"], pred5[i, "pred4E"])
    pred5[i, "finalresult"] = i_classes[which.max(tabulate(match(i_classes, unique(i_classes))))]
  }
  accuracy = confusionMatrix(as.factor(pred5$finalresult), vald$Personal.Loan)$overall["Accuracy"]
  if (accuracy > maxaccuracy) {
    maxaccuracy = accuracy
    bestthreshold = threshold
  }
}

print(bestthreshold) # 0.544
print(maxaccuracy) # 0.992
  
# Evaluation of performance on unseen data (i.e. test data set)
# with the accuracy arrived at the previous step

threshold = 0.544
pred2E = predict(RF2, newdata = test, type="prob")
pred3E = predict(RF3, newdata = test, type="prob")
pred4E = predict(boost4, test)

pred5 = data.frame(ifelse(pred2E[,2]>threshold, 1, 0), ifelse(pred3E[,2]>threshold, 1, 0), ifelse(pred4E$prob[,2]>threshold, 1, 0))
colnames(pred5) = c("pred2E", "pred3E", "pred4E")


for (i in seq(1, nrow(pred5), 1)) {
  i_classes = c(pred5[i, "pred2E"], pred5[i, "pred3E"], pred5[i, "pred4E"])
  pred5[i, "finalresult"] = i_classes[which.max(tabulate(match(i_classes, unique(i_classes))))]
}


confusionMatrix(as.factor(pred5$finalresult), as.factor(test$Personal.Loan))$overall["Accuracy"]
# Accuracy: 0.983

table(as.factor(pred5$finalresult), test$Personal.Loan)
#     0   1
# 0 892  14
# 1   3  91

# On unseen data (test data set), for a threshold of 0.544, 
# the model has performed very well; an Accuracy of 0.983 was obtained,
# although this is a slight reduction when compared to Model #5

#------------------------------------------------------------------------------------
#



