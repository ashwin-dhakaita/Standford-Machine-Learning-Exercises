function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
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
% =============================================================

end
