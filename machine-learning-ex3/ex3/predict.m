function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% For forward propagation, using the vectorized method, here's an outline. 
% This is an implementation of the formula in Figure 2 on Page 11 of ex3.pdf.

% DIMENSION ANALYSIS
% Theta 1 = [25 x 401]
% Theta2 = [10 x 26]
% X = [5000 x 400] (5000 20x20 pixel images)
% p should be [5000 x 1]

% 1. Add a column of 1's to X (the first column), and it becomes a1
% a1 is the input layer
a1 = [ones(m, 1), X];   % [5000 x 401]

% 2. Multiply a1 by Theta1 and it becomes z2
z2 = Theta1 * a1';      % [25 x 5000]

% 3. Add a column of 1's to the sigmoid of z2 for a2
% a2 is the hidden layer
a2 = [ones(1, size(z2,2) ); sigmoid(z2)];   % [26 x 5000]

% 4. Multiply by Theta2, take the sigmoid() and it becomes 'a3'.
% a3 is the output layer
z3 = Theta2 * a2;        % [10x 26] * [26 x 5000] = [10 x 5000]
a3 = sigmoid(z3);       % [10 x 5000]

% 6. Now use the max(a3, [], 2) function to return a vector of the outputs with the highest 'a3' value for each training example. Be sure you account for both return values.
% p is the number corresponding to the highest number in each column of a3
% where each column in a3 corresponds to one input image
% temp is the actual value of the highest value (not used)
[temp, p] = max(a3', [], 2);    % p = [5000 x 1]

% =========================================================================


end
