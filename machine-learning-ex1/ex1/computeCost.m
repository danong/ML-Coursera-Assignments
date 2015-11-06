function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% for i=1:m
%       tempA = theta(1) + theta(2) * X(i, 2) - y(i);
%       tempA = tempA ^ 2;
%       J = J + tempA;
% 
% end
% J = J / (2*m);
h = X*theta;
errors = h - y;
errors = errors.^2;
J = sum(errors) / (2*m);

% =========================================================================

end
