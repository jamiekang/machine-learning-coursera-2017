function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);
M = size(X,1);
% You need to return the following variables correctly.
idx = zeros(M, 1);  % idx: m x 1, X: 300x2

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

min_cost = 100000000000000000 * ones(M, 1);
min_cost_index = zeros(M, 1);

% j:1~m
%   i:1~K
for j=1:M
    for i=1:K
        temp_cost(j) = (X(j,:) - centroids(i,:))*(X(j,:) - centroids(i,:))';
        % centrolids: 3x2, centroids(i,:): 1x2 
        if (temp_cost(j) <= min_cost(j))
            min_cost(j) = temp_cost(j);
            min_cost_index(j) = i;
        endif
    endfor
    idx(j) = min_cost_index(j);
endfor
% =============================================================

end

