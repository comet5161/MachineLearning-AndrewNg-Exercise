function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%





y1 = sigmoid(X * theta);
J = sum( -y .* log(y1) - (1 - y) .* log(1 - y1) )/m ;
grad = X' * ( y1 - y ) /m;

theta1 = theta;
theta1(1) = 0;
J = J + lambda * sum(theta1.^2) /(2* m);
grad = grad + theta1 .* lambda ./ m;

%%
% 注：求向量每个元素的平方的和的两种方法有少量误差，如：
% >> vv = [1:0.0000001:2];
% >> vv*vv'
% ans =
%      2.333333583333336e+07
% >> sum(vv.^2)
% ans =
%      2.333333583333349e+07
% >> 
%%



% =============================================================

grad = grad(:);

end
