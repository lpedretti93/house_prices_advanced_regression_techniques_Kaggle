#             BIG DATA LAB - WORKGROUP
#data: https://www.kaggle.com/c/house-prices-advanced-regression-techniques

#install packages needed
#install.packages(dplyr)
#install.packages(dummies)
#install.packages(caret)
#install.packages(Metrics)
#install.packages("randomForest")
#install.packages("party")
#install.packages("xgboost")
#install.packages("gbm")

#loading libraries needed
library(dplyr)
library(dummies)
library(caret)
library(Metrics)

#setting working directory
setwd("C:\\Users\\Luca\\OneDrive\\Documents\\University\\Master in Data Science\\Big Data Lab\\Workgroup")

#importing files
df_train<-read.csv("train.csv",header = T,stringsAsFactors = F)
df_test<-read.csv("test.csv",header = T,stringsAsFactors = F)
df_train$SalePrice <- log(df_train$SalePrice+1)
df_test$SalePrice <- as.numeric(0)
df<-rbind(df_train,df_test)

# 1) PREPROCESSING DATA

#checking type of variables
sapply(df, typeof)

#checking the percentage of missing value per each variable
missing_values <- df %>% summarise_all(funs(sum(is.na(.)/n())))

#replacing every missing value with a 0
df[is.na(df)] <- 0
missing_values <- df %>% summarise_all(funs(sum(is.na(.)/n())))

# 2) FEATURE ENGENEERING

#reconding oredered categorical variables in continuos variables
df$ExterQual<-recode(df$ExterQual,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)
df$ExterCond<-recode(df$ExterCond,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)
df$BsmtQual<-recode(df$BsmtQual,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)
df$BsmtCond<-recode(df$BsmtCond,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)
df$BsmtExposure<-recode(df$BsmtExposure,"No"=1,"Mn"=2,"Av"=3,"Gd"=4)
df$BsmtFinType1<-recode(df$BsmtFinType1,"Unf"=1,"LwQ"=2,"Rec"=3,"BLQ"=4,"ALQ"=5,"GLQ"=6)
df$BsmtFinType2<-recode(df$BsmtFinType2,"Unf"=1,"LwQ"=2,"Rec"=3,"BLQ"=4,"ALQ"=5,"GLQ"=6)
df$HeatingQC<-recode(df$HeatingQC,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)
df$KitchenQual<-recode(df$KitchenQual,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)
df$Functional<-recode(df$Functional,"Sev"=1,"Maj2"=2,"Maj1"=3,"Mod"=4,"Min2"=5,"Min1"=6,"Typ"=7)
df$FireplaceQu<-recode(df$FireplaceQu,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)
df$GarageFinish<-recode(df$GarageFinish,"Unf"=1,"RFn"=2,"Fin"=3)
df$GarageQual<-recode(df$GarageQual,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)
df$GarageCond<-recode(df$GarageCond,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)
df$PoolQC<-recode(df$PoolQC,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=5)
df$Fence<-recode(df$Fence,"MnWw"=1,"GdWo"=2,"MnPrv"=3,"GdPrv"=4)
df[is.na(df)] <- 0

#binarizing categorical variables
df<-dummy.data.frame(df,dummy.classes = "character")


# 3) MODELLING

#splittig again the dataframe in the labeled(df_train) and unlabeled(df_test)
df_train<-df[1:1460,]
df_test<-df[1461:2919,]
df_train$Id<-NULL
id<-df_test$Id
df_train$Id<-NULL

#fixing the seed
set.seed(22)

#splitting into training(80%), test(20%)
id_train<-createDataPartition(df_train$SalePrice,p=0.8,list=F)
train<-df_train[id_train,]
test<-df_train[-id_train,]

##multilinear regression
#mlr=lm(SalePrice~ .,data=train)
#prediction_mlr<- predict(mlr, test)
#prediction <- cbind(id,prediction_mlr)
#rmse(test$SalePrice,prediction_mlr)

#random forest
rf=train(SalePrice~ .,
         data=train,
         method="rf",
         trControl=trainControl(method="cv",number=10))
prediction_rf<- predict(rf,test)
rmse(test$SalePrice,prediction_rf)

#tuned Gradient Boosting Machine
gbm_grid= expand.grid(n.trees = 700, 
              interaction.depth = 5, shrinkage = 0.05,
              n.minobsinnode = 10)
gbm=train(SalePrice~ .,
         data=train,
         method="gbm",
         verbose = FALSE,
         tuneGrid=gbm_grid,
         trControl=trainControl(method="cv",number=10))
prediction_gbm<- predict(gbm,test)
rmse(test$SalePrice,prediction_gbm)

##tuned XGBoost
#xgb=train(SalePrice~ .,
#         data=train,
#         method="xgbTree",
#         trControl=trainControl(method="cv",number=10))
#prediction_xgb<- predict(xgb,test)
#rmse(test$SalePrice,prediction_xgb)

##linear svm
#svm1=train(SalePrice~ .,
#           data=train,
#           method="svmLinear",
#           preProcess = c("center", "scale"),
#           trControl=trainControl(method = "cv",number=10))
#prediction_svm1<- predict(svm1, test)
#rmse(test$SalePrice,prediction_svm1)

#svm with polinomial transformation
svm2=train(SalePrice~ .,
           data=train,
           method="svmPoly",
           preProcess = c("center", "scale"),
           trControl=trainControl(method = "cv",number=10))
prediction_svm2<-predict(svm2, test)
rmse(test$SalePrice,prediction_svm2)

##svm with polinomial transformation
#grid_poly <- expand.grid(degree = c(2,3,4),
#                         scale = c(1,2,4,6,10),
#                         C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75,
#                                 1, 1.5, 2,5))
#svm3=train(SalePrice~ .,
#           data=train,
#           method="svmPoly",
#           preProcess = c("center", "scale"),
#           tuneGrid = grid_poly,
#           trControl=trainControl(method = "cv",number=10))
#prediction_svm3<- predict(svm3, test)
#rmse(test$SalePrice,prediction_svm3)#

###tuned svm with radial transformation
#grid_radial <- expand.grid(sigma = c(0,0.01, 0.02, 0.025, 0.03, 0.04,
#                                     0.05, 0.06, 0.07,0.08, 0.09, 0.1, 0.25, 0.5, 0.75,0.9),
#                           C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75,
#                                 1, 1.5, 2,5))
#svm4=train(SalePrice~ .,
#          data=train,
#          method="svmRadial",
#          tuneGrid = grid_radial,
#          preProcess = c("center", "scale"),
#          tuneLength = 10,
#          trControl=trainControl(method = "cv",number=10))
#prediction_svm4<- predict(svm4, test)
#rmse(test$SalePrice,prediction_svm4)

#combining the best algrithm () with a weighted average of the predictions
prediction_avg<-0.5*prediction_gbm+0.25*prediction_svm2+0.25*prediction_rf
rmse(test$SalePrice,prediction_avg)

# 4) PREDICTING AND SUBMITTING THE TEST SET
#predicting the unlabeled test set
prediction_rf_test<-predict(rf,df_test)
prediction_svm2_test<-predict(svm2,df_test)
prediction_gbm_test<-predict(gbm,df_test)
prediction<-0.5*prediction_gbm_test+0.25*prediction_svm2_test+0.25*prediction_rf_test
prediction<-exp(prediction)-1

#writing the submission file 
submission <- cbind(id,prediction)
write.table(submission,file="submission19.csv",sep=",",
            quote=FALSE,col.names=c("Id","SalePrice"),row.names=FALSE)