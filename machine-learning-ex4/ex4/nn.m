function [J grad] = nn()
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
% Setup some useful variables
lambda = 1;
d = load("H:/machine-learning-ex4/ex4/ex4data1.mat");
X = d.X; y = d.y;
d = load("H:/machine-learning-ex4/ex4/ex4weights.mat");
Theta1 = d.Theta1; Theta2 = d.Theta2;
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

%I = eye(num_labels);
%Y = zeros(m, num_labels);
%for i=1:m
%  Y(i, :)= I(y(i), :);
%end



A1 = X;
A1 = [ones(size(A1,1),1) A1];
Z2 = Theta1*A1';
A2 = sigmoid(Z2);
A2 = A2';
A2 = [ones(size(A2,1),1) A2];
Z3 = Theta2*A2';
X = sigmoid(Z3);
A3 = X';
X = X';
J2 = 0;
for i=1:m
  for j=1:size(X,2)
    J = J + (y(i)*log(X(i,j))+(1-y(i))*log(1-X(i,j)));
    end
end
J = (-1/m)*J
su = 0;
for i=1:size(Theta1,2)
  for j=1:size(Theta1,1)
    su = su + Theta1(j,i)^2;
  end
end
for i=1:size(Theta2,2)
  for j=1:size(Theta2,1)
    su = su +  Theta2(j,i)^2;
  end
end
J = J + (lambda/(2*m))*su
%A1 = [ones(m, 1) X];
%Z2 = A1 * Theta1';
%A2 = [ones(size(Z2, 1), 1) sigmoid(Z2)];
%Z3 = A2*Theta2';
%H = A3 = sigmoid(Z3);

%penalty = (lambda/(2*m))*(sum(sum(Theta1(:, 2:end).^2, 2)) + sum(sum(Theta2(:,2:end).^2, 2)));

%J = (1/m)*sum(sum((-y).*log(A3) - (1-y).*log(1-A3), 2))
%J = J + penalty

%Sigma3 = A3 - Y;
%Sigma2 = (Sigma3*Theta2 .* sigmoidGradient([ones(size(Z2, 1), 1) Z2]))(:, 2:end);


%Delta_1 = Sigma2'*A1;
%Delta_2 = Sigma3'*A2;


%Theta2_grad = Delta_2./m + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];



end