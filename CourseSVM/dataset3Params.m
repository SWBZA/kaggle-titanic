function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Original loop range: [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

loop_number = 1;
error = 10000;
for C_loop = [2.6, 3.0, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8]
     for sigma_loop = [2.6, 3.0, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8]
          fprintf("Loop number %d\n out of 121", loop_number)
          loop_number = loop_number + 1;
          % Train with the current C and sigma
          model= svmTrain(X, y, C_loop, @(x1, x2) gaussianKernel(x1, x2, sigma_loop));
          
          % Evaluate error
          predictions = svmPredict(model, Xval); % Predict the error
          error_loop = mean(double(predictions ~= yval));  % Calculate error
          %fprintf("For C = %d and sigma = %d the error is %d \n\n", C_loop, ...
          %sigma_loop, error_loop)
          % Update C and sigma if better fit reached
          if error_loop < error
               error = error_loop; % Update error to smallest reached
               C = C_loop;
               sigma = sigma_loop;
               % fprintf("#####  New best parameter set found. C = %d, sigma = %d, error is %d\n \n", C, sigma, error)
          endif

     endfor
endfor

fprintf("Best parameter set found. C = %d, sigma = %d, error is %d\n \n", C, sigma, error)


% =========================================================================

end
