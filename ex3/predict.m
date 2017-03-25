function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);					% X: 5000x400
num_labels = size(Theta2, 1);	% Theta2: 10x26

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);		% p: 5000

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

% Add ones to the X data matrix
X = [ones(m, 1) X];			% X: 5000x401
% Theta1: 25 x 401
a2 = sigmoid(X * Theta1');	% X: 5000x401, Theta1: 25x401, a2 = 5000x25

% Add ones to the X data matrix
a2 = [ones(m, 1) a2];			% a2: 5000x26
% Theta2: 10 x 26
a3 = sigmoid(a2 * Theta2'); % a2: 5000x26, Theta2: 10x26, a3 = 5000x10

[val, p] = max(a3, [], 2);	% Note: return index (not value)
% =========================================================================

end
