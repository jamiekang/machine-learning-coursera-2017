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
h_theta = X * theta;	% X[m,2], theta[2,1], h_theta[m,1,], y[m,1]
J = 1/(2*m) * (h_theta - y)' * (h_theta - y);
% =========================================================================

end

