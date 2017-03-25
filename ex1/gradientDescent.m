function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    h_theta = X * theta;    % X[m,2], theta[2,1], h_theta[m,1], y[m,1]
                            % X_j[m,2]
    theta_new = theta - (alpha/m) .* ((h_theta - y)' * X)';

    % print theta to screen
    %fprintf('gradientDescent: ');
    %fprintf('(%f %f) <- (%f %f)\n', theta_new(1), theta_new(2), theta(1), theta(2));

    theta = theta_new;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

    % print theta to screen
    fprintf('gradientDescent: iter=%d, J=%f\n', iter, J_history(iter));
end %end-for

end
