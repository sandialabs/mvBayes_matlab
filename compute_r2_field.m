function R2_field = compute_r2_field(Ytrue, Ypred)
% COMPUTE_R2_FIELD  Per-spatial-DOF coefficient of determination.
%
%   R2_field = compute_r2_field(Ytrue, Ypred)
%
%   Ytrue, Ypred : n x Nval matrices (n spatial DOFs, Nval validation
%                  snapshots), as produced by predict_mf_nipod /
%                  predict_mono_nipod versus the true field Yval.
%   R2_field     : n x 1 vector, R^2 computed independently at each
%                  spatial location j, across the Nval validation points:
%                     R2_field(j) = 1 - SSres_j / SStot_j
%                  This shows *where* in the field (e.g. which x, or
%                  which region of a mesh) the surrogate reproduces the
%                  design-space variability well or poorly.

SSres = sum((Ytrue - Ypred).^2, 2);
SStot = sum((Ytrue - mean(Ytrue,2)).^2, 2);

R2_field = nan(size(SSres));
% valid = SStot > eps(max(abs(Ytrue(:))))*size(Ytrue,2);
R2_field = 1 - SSres./SStot;
end
