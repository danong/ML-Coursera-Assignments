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

% Variables Analysis:
% X = 300 examples where each dimension is a 2D vector [300 x 2]
% centroids = 3 centroids where each centroid is a 2D vector[3 x 2]
% K = # of centroids: 3 [1 x 1]
% idx = index of closest centroid for each point[300 x 1]

% iterate through examples
for i = 1:size(X,1)
    % initialize min variable which holds distance and index of closest
    % centroid
    min = [inf, 0];
    % iterate through centroids
    for j = 1:K
        % find euclidean distance between ith training example and jth
        % centroid
        dist = pdist([X(i,:);centroids(j,:)], 'euclidean');
        % record distance and index of lowest training centroid
        if (dist < min(1,1))
            min = [dist; j];
        end;
    end;
    % save index of closest centroid to idx
    idx(i) = min(2, 1);
end;
        




% =============================================================

end

