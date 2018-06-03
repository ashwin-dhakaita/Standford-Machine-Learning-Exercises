function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;
stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
 
    % =============================================================
    pval2 = pval<epsilon;fp=0;fn=0;tp=0;
    for j=1:size(pval2)
      if(pval2(j)==1 && yval(j)==1)
        tp++;
      elseif(pval2(j)==1 && yval(j)==0)
        fp++;
      elseif(pval2(j)==0 && yval(j)==1)
        fn++;
      end
    end
    R = (tp)/(tp+fn);
    P = (tp)/(tp+fp);
    F1 = 2*((P*R)/(P+R));
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
