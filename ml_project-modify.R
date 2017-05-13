# This R code is for raw dataset summary
# No change about the dataset

# import dataset

summary(train)
dim(train)

# Box plot
boxplot(train$loss, horizontal=TRUE, main="Loss")

# Histogram for continuous attributes
hist(train$loss, breaks=10)
hist(train$cont1, breaks=10)
hist(train$cont2, breaks=10)
hist(train$cont3, breaks=10)
hist(train$cont4, breaks=10)
hist(train$cont5, breaks=10)
hist(train$cont6, breaks=10)
hist(train$cont7, breaks=10)
hist(train$cont8, breaks=10)
hist(train$cont9, breaks=10)
hist(train$cont10, breaks=10)
hist(train$cont11, breaks=10)
hist(train$cont12, breaks=10)
hist(train$cont13, breaks=10)
hist(train$cont14, breaks=10)

# plot for category attributes
plot(train$cat1, main = "cat1")
plot(train$cat2, main = "cat2")
plot(train$cat77, main = "cat77")
plot(train$cat100, main = "cat100")
plot(train$cat110, main = "cat110")
# ....not list all

#res <- cor(train$cont1~train$cont3)
#round(res, 2)


library(corrplot)
trainContData = train[ , c(118,119,120,121,122,123,124,125,126,127,128,129,130,131)]
summary(trainContData)
corMatrix = cor(trainContData)
corMatrix
corrplot(corMatrix, method="circle")

#cor(cont11 with cont12) are high
#cor(cont1 with cont9) are high
plot(train$cont11, train$cont12)
plot(train$cont1, train$cont9)


# try Stepwise regression to find important attributes
library(MASS)

#Delete "id" useless attribute
train$id = NULL
#Delete correclated attribute
train$cont12 = NULL
train$cont9 = NULL
summary(train)

#split data into newTrain and newTest
# set the train set a bit smaller, to run model fast
sample_size = floor(0.6 * nrow(train))
set.seed(123)
newTrain_id = sample(seq_len(nrow(train)), size = sample_size)
newTrain = train[newTrain_id, ]
newTest = train[-newTrain_id, ]

# Zero model, used in scope
Y_0_lm <- lm(loss~1, data = newTrain)

# Model 1: full model
Y_all_lm <- lm(loss~., data = newTrain)

# Model 2 : backward AIC
Y_bw_AIC_lm = step(Y_all_lm, direction = "backward", scope = list(lower = Y_0_lm, upper = Y_all_lm))
# Not success. Due to big dataset, takes moe than 24 hours to run. 

# Model 3 : forward AIC
Y_fw_AIC_lm = step(Y_all_lm, direction = "forward", scope = list(lower = Y_0_lm, upper = Y_all_lm))

# Model4 : backward BIC
Y_bw_BIC_lm = step(Y_all_lm, direction = "backward", scope = list(lower = Y_0_lm, upper = Y_all_lm), k = log(112990))

# Model5 : forward BIC
Y_fw_BIC_lm = step(Y_all_lm, direction = "forward", scope = list(lower = Y_0_lm, upper = Y_all_lm), k = log(112990))

#Compare the 5 models
anova(Y_all_lm, Y_bw_AIC_lm, Y_fw_AIC_lm,Y_bw_BIC_lm, Y_fw_BIC_lm )

