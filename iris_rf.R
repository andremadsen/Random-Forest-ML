# Download R: https://cran.r-project.org/bin/windows/base/
# Download RStudio: https://www.rstudio.com/products/rstudio/download/#download

# Copy/Paste the following code into RStudio (new Project)
# Press CTRL+Enter to run each line of code below

install.packages("caret")
library(caret)

install.packages("randomForest")
library(randomForest)

install.packages("pROC")
library(pROC)

install.packages("datasets")
library(datasets)

# Load and inspect the demo dataset 'iris'
data(iris)
str(iris)
head(iris)
tail(iris)
View(iris)





# Randomly partition train/test dataframes: 75% / 25% split for Training / Testing (while keeping same prevalence of 'Species' in both datasets)
indexes <- createDataPartition(iris$Species,
                               times = 1,
                               p = 0.75,
                               list = FALSE)

train.MLdata <- iris[indexes,]
test.MLdata <- iris[-indexes,]


# Create randomForest ML-model (using Training data)
RFmodel <- randomForest(Species ~ ., data=train.MLdata, ntree=1001, mtry=3, proximity=TRUE)
RFmodel

# Use the ML model 'RFmodel' to predict Species (in the unseen Test dataset)
result <- predict(RFmodel, test.MLdata[,1:4], type="response")
result
a <- data.frame(test.MLdata$Species, result)
colnames(a) <- c("Species.testdata", "Prediction")
View(a)

b <- confusionMatrix(result, test.MLdata$Species)
b
b$overall[1]    #Using 'Accuracy' as the Key Performance Index




#################################
# BINARY CLASSIFICATION CASE: 
#################################


# New dataset 'iris.temp' 
iris.temp <- iris

# Combine flower species setosa/versicolor =1 VS virginica = 0
iris.temp$Family <- ifelse(iris.temp$Species == "setosa", 1,
                    ifelse(iris.temp$Species == "versicolor", 1,
                    ifelse(iris.temp$Species == "virginica", 0, NA)))

iris.temp$Family <- as.factor(iris.temp$Family)

NewData <- data.frame(iris.temp$Sepal.Length,iris.temp$Sepal.Width,iris.temp$Petal.Length,iris.temp$Petal.Width,iris.temp$Family)
colnames(NewData) <- c("Sepal.Length","Sepal.Width","Petal.Length","Petal.Width","Family")
head(NewData)



# Partition train/test dataframes from NewData: 75% / 25% split for Training / Testing
indexes <- createDataPartition(NewData$Family,
                               times = 1,
                               p = 0.75,
                               list = FALSE)

train.MLdata <- NewData[indexes,]
test.MLdata <- NewData[-indexes,]


# Train new ML model 'RFmodel2' (training data)
RFmodel2 <- randomForest(Family ~ ., data=train.MLdata, ntree=1001, mtry=3, proximity=TRUE)

# New ML model 'RFmodel2' to classify performance (test data)
result2 <- predict(RFmodel2, test.MLdata[,1:4], type="prob")
result2

result2[,2]


# ROC curve to evaluate ML classification performance
class <- test.MLdata$Family    #must be  a 'factor' variable
score <- result2[,2]           #must be numbers
rf.roc <- roc(class, score)

plot(rf.roc)
auc(rf.roc)
coords(rf.roc, "best", transpose=TRUE, ret=c("threshold", "ppv", "npv", "sens", "spec", "accuracy","tp","fp","tn","fn"))

# threshold = optimal cutoff point (e.g. biomarker mmol/L to distinguish healthy vs disease)
# ppv = positive predictive value: True negatives/(True negatives + False Negatives) | NPV = TN/(TN+FN)
# npv = negative predictive value: True positives/(True positives + False positives) | PPV = TP/(TP+FP)
# accuracy = (TP+TN)/total obs

