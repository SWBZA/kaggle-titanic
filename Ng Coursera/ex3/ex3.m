% Andrew Ng Coursera course multi-class neural nets
% Applying to Kaggle Titanic competition

%% Initialization
clear ; close all; clc

%% Setup the parameters 
input_layer_size  = 7;  % 7 features chosen out of Titanic data set
num_labels = 2;          % Trying to use multi-class with 1 and 0

% Load Training Data
fprintf('Loading Data ...\n')

input_data=csvread("train.csv");
y = input_data(2:end,1);
X = input_data(2:end,2:end);

m = size(X, 1);

%fprintf('Program paused. Press enter to continue.\n');
%pause;

%% ============ One-vs-All Training ============
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

#fprintf('Program paused. Press enter to continue.\n');
#pause;


%% ================ Predict for One-Vs-All ================
%for i = 0.4:0.001:0.42;
%     threshhold = i;
%     pred = predictOneVsAll(all_theta, X, threshhold);
%     fprintf('\nTraining Set Accuracy for threshhold i=%d : %f\n', i, mean(double(pred == y)) * 100);
%endfor

threshhold=0.408;
pred = predictOneVsAll(all_theta, X, threshhold);

%% ================= Predict for Titanic test set ================

% Load test data
input_test_data=csvread("test.csv");
X_predict = input_test_data(2:end,2:end);


% Make prediction on test data set
threshhold=0.408;
pred = predictOneVsAll(all_theta, X_predict, threshhold);

% Add passenger number from test data set to upload to Kaggle
prediction_out = [input_test_data(2:end, 1)  pred(:,:)];

% Output data - REMEMBER, NO HEADER IN FILE
csvwrite("LogisticRegression.csv", prediction_out);