%% Kaggle Titanic challenge with Andrew Ng course Neural Net
     
% Initialization
clear ; close all; clc

% Setup the parameters
input_layer_size  = 7;   % 7 input variables
hidden_layer_size = 70;  % 3 hidden units per input node
num_labels = 1;          % survive / not-survive 

% Load data
input_data=csvread("train.csv");
y = input_data(2:end,1); % y variable in the first column, labels in first row
X = input_data(2:end,2:end); % exclude first row - contains labels

% Normalise input
X = featureNormalize(X);

% Split into training and cross validation sets
y_train = y(1:712,:);
X_train = X(1:712,:);
y_cv = y(713:end,:);
X_cv = X(713:end,:);
m = size(X, 1);

%% ================ Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('\nParameters unrolled ...\n');

%% ===================  Training NN ===================
%  To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')


%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);  %  Initial suggested = 50

%  You should also try different values of lambda
lambda = 1;
%return;

[J grad] = nnCostFunction(initial_nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

%return;
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
%return;

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


%% ================= Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the survival. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
%return;


%  When predicting test set, remember to normalise

%% ================= Predict for Titanic test set ================

% Load test data
input_test_data=csvread("test.csv");
X_predict = input_test_data(2:end,2:end);

% Normalise input
X_predict = featureNormalize(X_predict);

% Make prediction on test data set
prediction = predict(Theta1, Theta2, X_predict);

% Add passenger number from test data set to upload to Kaggle
prediction_out = [input_test_data(2:end, 1)  prediction(:,:)];

% Output data - REMEMBER, NO HEADER IN FILE
csvwrite("submission.csv", prediction_out);
