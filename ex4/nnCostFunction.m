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
m = size(X, 1);        % X:5000x400, m: 5000
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));    % Theta1: 25x401
Theta2_grad = zeros(size(Theta2));    % Theta2: 10x26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Feedforward
% Add ones to the X data matrix
X = [ones(m, 1) X];            % X: 5000x401
% Theta1: 25 x 401
a2 = sigmoid(X * Theta1');    % X: 5000x401, Theta1: 25x401, a2 = 5000x25

% Add ones to the X data matrix
a2 = [ones(m, 1) a2];            % a2: 5000x26
% Theta2: 10 x 26
a3 = sigmoid(a2 * Theta2'); % a2: 5000x26, Theta2: 10x26, a3 = 5000x10

% Compute cost
for k = 1:size(a3,2)
    yk(:,k) = (y == k);        % yk: 5000x10, to recode y into vector of (0 or 1), y: 5000x1
endfor

for k = 1:size(a3,2)
    term1 = -yk(:,k) .* log(a3(:,k));              % term1: 5000x1    
    term2 = -(1 - yk(:,k)) .* log(1 - a3(:,k));    % term2: 5000x1
    J = J + 1/m * sum(term1 + term2);              % J: 1x1
endfor

% with regularization
sq_Theta1 = Theta1 .* Theta1;   % sq_Theta1: 25x401
sq_Theta1(:,1) = 0;             % exclude bias inputs
sq_Theta2 = Theta2 .* Theta2;   % sq_Theta2: 10x26
sq_Theta2(:,1) = 0;
term3 = lambda /2/m * (sum(sq_Theta1(:)) + sum(sq_Theta2(:)));
J = J + term3;

%
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

for t=1:m
    a1 = X(t,:);          % X: 5000x401, a1: 1x401
    z2 = a1 * Theta1';    % a1: 1x401, Theta1: 25x401, z2: 1x25
    a2 = sigmoid(z2);     % a2: 1x25

    % Add ones to the X data matrix
    a2 = [1 a2];          % a2: 1x26
    z3 = a2 * Theta2';    % a2: 1x26, Theta2: 10x26, z3: 1x10
    a3 = sigmoid(z3);     % a3: 1x10
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    for k = 1:size(a3,2)
%        delta3 = a3 - yk(t,k);     % a3: 1x10, yk: 5000x10, delta3: 1x10
%    endfor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    delta3 = a3 - yk(t,:);     % a3: 1x10, yk: 5000x10, delta3: 1x10

    delta2 = delta3 * Theta2 .* [1 sigmoidGradient(z2)];    % Theta2: 10x26, delta3: 1x10, z2: 1x25, delta2: 1x26
    delta2 = delta2(2:end);     % delta2: 1x25

    Theta2_grad = Theta2_grad + delta3' * a2;         % Theta2_grad: 10x26, delta3: 1x10, a2: 1x26
    Theta1_grad = Theta1_grad + delta2' * a1;         % Theta1_grad: 25x401, delta2: 1x25, a1: 1x401
endfor	

Theta2_grad = 1/m * Theta2_grad;    % Theta2_grad: 10x26
Theta1_grad = 1/m * Theta1_grad;    % Theta1_grad: 25x401

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

temp2 = lambda /m * Theta2;             % temp2: 10x26
temp2(:,1) = zeros(size(Theta2,1),1);
Theta2_grad = Theta2_grad + temp2;    % Theta2_grad: 10x26

temp1 = lambda /m * Theta1;             % temp1: 25x401
temp1(:,1) = zeros(size(Theta1,1),1);
Theta1_grad = Theta1_grad + temp1;    % Theta1_grad: 25x401
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
