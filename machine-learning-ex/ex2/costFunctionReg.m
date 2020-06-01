function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

J = 0;
for i = 1:m
    h = sigmoid(theta' * X(i, :)');
    J = J + ((-y(i) * log(h)) - ((1 - y(i)) * log(1 - h)));
end

J = J / m;

J = J + (lambda / (2 * m)) * (sum(theta .^2) - theta(1)^2);




for j = 1:size(theta)
  for i = 1:m
        h = sigmoid(theta' * X(i, :)');
        grad(j) = grad(j)+ ((1/m) * (h - y(i)) * X(i, j));
        
  end
  if j > 1
    grad(j) = grad(j) + (lambda * theta(j) * (1/m));
  end
end


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
