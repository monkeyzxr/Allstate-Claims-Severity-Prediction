# Allstate-Claims-Severity-Prediction
Machine learning project in UTD

Created by Xiangru Zhou and Xunde Wang
*************************************
Introduction and problem description

It is a project from kaggle competitions. https://www.kaggle.com/c/allstate-claims-severity

Insurance companies are always interested in finding better ways to predict claims severity. The dataset we are going to use is from All State insurance company, which contains 116 categorical variables and 14 continuous variables. We need to predict the severity, which is the loss of a claim, from those 130 independent variables. 

Kaggle use MAE (Mean Absolute Error) as evaluation metrics.

Learning: We need to predict the 'loss' based on the other attributes. Hence, this is a regression problem.

*****************************************
Related work

Insurance companies usually build a parametric probability distribution model from previous claims then predict future claim severity by fitting data into that model.

**************************************
Dataset description

The object is to predict the insurance loss from a dataset from an insurance company.

In the train.csv:

Number of instance: 188319 

Number of attributes: 131

Number of category attributes: 116 

Number of continuous attributes: 14 

Target: loss (continuous variable)

In the test.csv:

Number of instance: 12546

Number of attributes: 130

Number of category attributes: 116 

Number of continuous attributes: 14
