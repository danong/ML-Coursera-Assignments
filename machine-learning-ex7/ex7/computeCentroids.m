function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% m = number of examples: 300
% n = number of dimensions: 2
% K = number of centroids: 3
% idx = index of closest centroid for each point: [300 x 1]

count = zeros(K, 1);
% iterate through training examples
for i = 1:m
    % k gets index of closest centroid
    k = idx(i);
    % add ith example to kth centroid
    centroids(k,:) = centroids(k,:) + X(i,:);
    count(k) = count(k) +1;
end;

for j = 1:K
    if (count(j) ~= 0)
        centroids(j,:) = centroids(j, :) ./ count(j);
    end;
end;




% =============================================================


end

