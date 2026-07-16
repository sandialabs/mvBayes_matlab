function R2 = compute_r2(Ytrue, Ypred)
% COMPUTE_R2  Coefficient of determination R^2 between true and
% predicted values.
%
%   R2 = compute_r2(Ytrue, Ypred)
%
%   Ytrue, Ypred : arrays of identical size (vectors, matrices, ...).
%                  Flattened internally, so this works equally well on:
%                    - a single scalar quantity of interest evaluated
%                      over a validation set (vectors of length Nval)
%                    - a full stacked field (n x Nval matrices), giving
%                      one single "global" R^2 over all spatial points
%                      and all validation snapshots at once.
%   R2           : scalar coefficient of determination
%
%       R2 = 1 - SS_res / SS_tot
%       SS_res = sum( (Ytrue - Ypred).^2 )
%       SS_tot = sum( (Ytrue - mean(Ytrue)).^2 )
%
% R2 = 1 is a perfect surrogate; R2 = 0 means the surrogate is no better
% than predicting the mean of Ytrue; R2 < 0 means it is worse than that.
%
% Note: this is the classical regression R^2. It differs slightly from
% the Pearson-correlation-based validation statistic r used in the paper
% (Eqs. 5.1-5.2) -- see compute_pearson_r.m for that alternative, which
% is scale/bias-insensitive (a perfectly correlated but biased or
% rescaled prediction still gets r = 1, whereas R^2 would be penalized).

yt = Ytrue(:);
yp = Ypred(:);

SSres = sum((yt - yp).^2);
SStot = sum((yt - mean(yt)).^2);

if SStot < eps(max(abs(yt)))*numel(yt)
    % Degenerate case: (near-)constant true signal
    R2 = NaN;
else
    R2 = 1 - SSres/SStot;
end
end
