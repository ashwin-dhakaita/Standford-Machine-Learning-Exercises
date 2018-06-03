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
htx = X*theta;
htx = sigmoid(htx);
for i=1:m
  if(y(i)==1)
    htx(i) = -log(htx(i));
  elseif(y(i)==0)
    htx(i) = -log(1-htx(i));
  end
end 
J = sum(htx)/m;
htx = X*theta;
htx = sigmoid(htx);
htx = htx-y;
grad = X'*htx;
grad = grad/m;
th = theta.^2;
J = J + ((lambda/(2*m))*(sum(th)-th(1)));
theta = ((lambda/m)*theta);
theta(1) = 0;%theta(1)*(m/lambda);
grad = grad + theta;
% =============================================================

%grad = grad(:);

end
