
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
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%% 求Cost
A1 = [ones(m,1), X]'; %特征数 x 样本数
Z2= Theta1 * A1; % 第2层单元数 x 样本数
A2 = [ones(1, size(Z2,2)); sigmoid(Z2)];
Z3= Theta2 * A2;  % 第3层单元数 x 样本数
A3 = sigmoid(Z3);
H = A3; % 分类数 x 样本

Y = zeros(m, num_labels); % 样本数 x 分类数
% index=sub2ind(size(A),B(:,1),B(:,2));
index = sub2ind(size(Y), [1:m]', y); 
Y( index ) = 1;
J = sum(-Y' .* log(H) - (1-Y)' .* log(1 - H) )   ;
J = sum(J) / m;
 J = J + ( sum( sum(Theta1(:, 2:end) .^2 ) ) + sum( sum(Theta2(:,2:end)  .^2) ) ) * lambda / (2 * m);

%% 实现反向传播

%[第3单元数 x 样本数]
Delta3 = H - Y'; %（大写Δ，小写δ)
% A2 [第3单元数 x 样本数]

Delta3_3D = reshape(Delta3, [ size(Delta3, 1), 1, size(Delta3, 2) ] );

% 将 A2转为3维矩阵
A2_3D = reshape(A2, [size(A2, 1), 1, size(A2, 2)]); % 把样本数量放到第3维
A2_3D = permute(A2_3D, [2, 1, 3]); % 第一维与第二维转置
% Theta2_grad = （每个样本 Delta3 * a2'的和）/ 3。
%[第3单元数 x 1 x 样本数] x [1 x 第2单元数 x 样本数]
Theta2_grad = sum( bsxfun(@times, Delta3_3D , A2_3D), 3) / m; 

%  [第2单元数 x 第3层单元数 ] x [第3单元数 x 样本数] dot [ 第二单元数 x 样本数] 
Delta2 = Theta2(:, 2:end)' * Delta3 .* sigmoidGradient(Z2); % Theta2去掉偏置项。

Delta2_3D = reshape(Delta2, [size(Delta2, 1), 1, size(Delta2, 2)] );

A1_3D = reshape(A1, [size(A1, 1), 1, size(A1, 2)] );
A1_3D = permute(A1_3D, [2, 1, 3]);

Theta1_grad = sum( bsxfun(@times, Delta2_3D,  A1_3D), 3) / m;

Theta1_temp = Theta1;
Theta2_temp = Theta2;
Theta1_temp(:, 1) = 0;
Theta2_temp(:, 1) = 0;

Theta1_grad = Theta1_grad + Theta1_temp * lambda / m;
Theta2_grad = Theta2_grad + Theta2_temp * lambda /m;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

