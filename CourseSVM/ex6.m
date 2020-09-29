%% Using SVM build during Ng course for Titanic Kaggle competition

%% Initialization
clear ; close all; clc

% Load data
input_data=csvread("train.csv");
y = input_data(2:end,1);
X = input_data(2:end,2:end);

% Normalise input
X = featureNormalize(X);

% Split into training and cross validation sets
y_train = y(1:712,:);
X_train = X(1:712,:);
y_cv = y(713:end,:);
X_cv = X(713:end,:);

% Find optimal C and sigma
[C, sigma] = dataset3Params(X_train, y_train, X_cv, y_cv);

% Get model with optimal C and sigma
model= svmTrain(X_train, y_train, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 

%% ================= Predict for Titanic test set ================

% Load test data
input_test_data=csvread("test.csv");
X_predict = input_test_data(2:end,2:end);

% Normalise input
X_predict = featureNormalize(X_predict);

% Make prediction on test data set
prediction = svmPredict(model, X_predict);

% Add passenger number from test data set to upload to Kaggle
prediction_out = [input_test_data(2:end, 1)  prediction(:,:)];

% Output data - REMEMBER, NO HEADER IN FILE
csvwrite("submission.csv", prediction_out);









