function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples; m = 12

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% Note: no sigmoid() since it's linear regression
h_theta = X * theta;   % X: 12x2, theta: 2x1, h_theta: 12x1 

J = 1/(2*m) * (h_theta - y)' * (h_theta - y);
theta2 = theta; theta2(1) = 0;  % not to regularize theta(1)
                                % I set theta(1) to zero instead of lambda(1)
termr = lambda /2/m * theta2' * theta2;
J = J + termr;
termg = lambda/m * theta2;
grad = 1/m .* ((h_theta - y)' * X)' + termg;
% =========================================================================

grad = grad(:);

end
