function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% DIMENSION ANALYSIS
% Theta1 = [25 x 401]
% Theta2 = [10 x 26]
% X = [5000 x 400] (5000 20x20 pixel images)
% p should be [5000 x 1]

% 1.1 Add a column of 1's to X (the first column), and it becomes a1
% a1 is the input layer
a1 = [ones(m, 1) X];   % [5000 x 401]

% 1.2 Multiply a1 by Theta1 and it becomes z2
z2 = a1 * Theta1';      % [5000 x 25]

% 1.3 Add a column of 1's to the sigmoid of z2 for a2
% a2 is the hidden layer
a2 = [ones(m, 1) sigmoid(z2)];   % [5000 x 26]

% 1.4 Multiply by Theta2, take the sigmoid() and it becomes 'a3'.
% a3 is the output layer
z3 = a2 * Theta2';      % [5000 * 26] * [26 * 10] = [5000 x 10]
a3 = sigmoid(z3);       % [5000 x 10] aka h_\theta(x)
 
% 1.5 Generate y_matrix from y    
y_eye = eye(num_labels);
y_matrix = y_eye(y,:);  % [5000 x 10]

% 1.7 Finally calculate the cost function
J = (1/m) * sum( sum( -y_matrix .* log(a3) - (1-y_matrix) .* log(1-a3)));

% 1.8 Adding regularization terms without modifying bias term
J = J + ( (lambda / (2*m) ) * (sum( sum( Theta2(:,2:size(Theta2,2)).^2 ))...
    + sum( sum( Theta1(:,2:size(Theta1,2)).^2 ))));


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% Vectorized Implementation:
d3 = a3 - y_matrix;     % [5000 x 10]
d2 = d3* Theta2(:,2:end) .* sigmoidGradient(z2);  % [5000 x 10] x [10 x 25] = [5000 x 25]
Delta1 = d2' * a1;      % [25 x 5000] x [5000 x 26] = [25 x 401]
Delta2 = d3' * a2;      % [10 x 26]

Theta1_grad = Delta1 / m;
Theta2_grad = Delta2 / m;
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

% Set first column of Thetas to 0 so they aren't included in regularization
Theta1(:,1) = 0;
Theta2(:,1) = 0;
% Add regularization terms
Theta1_grad = Theta1_grad + (lambda/m) * Theta1;
Theta2_grad = Theta2_grad + (lambda/m) * Theta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
