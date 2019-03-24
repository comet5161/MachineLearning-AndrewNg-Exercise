function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

[m, n] = size(centroids);
centroids = reshape(centroids, [m, n, 1]);
centroids = permute(centroids, [3, 2, 1]); % 第一维与第三维转置
Y = bsxfun(@minus, X, centroids);
Z = sum(Y .^2, 2); % 第3维为样本与不同中心点的距离
[~, idx] = min(Z, [], 3);






% =============================================================

end

