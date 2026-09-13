function basis = basisBspline(fDomain, nBasis, degree)
%BASISBSPLINE B-spline basis matrix, one basis function per row.
%
%   basis = basisBspline(fDomain, nBasis)
%   basis = basisBspline(fDomain, nBasis, degree)
%
%   Equivalent to patsy.bs(fDomain, df=nBasis).' in the Python implementation:
%   a cubic (by default) B-spline basis with nBasis - degree + 1 interior knots
%   placed at quantiles of fDomain, clamped boundary knots, and the first
%   (intercept-like) basis function dropped.
%
%   fDomain : vector of points at which to evaluate the basis.
%   nBasis  : number of basis functions to return. Must be >= degree.
%   degree  : spline degree (default 3, i.e. cubic).
%
%   basis   : nBasis x numel(fDomain) matrix, one basis function per row.
%
%   The collocation matrix comes from SPCOL (Curve Fitting Toolbox). Without
%   that toolbox, an equivalent Cox-de Boor recursion is used instead, so
%   basisType="bspline" does not require the toolbox to be installed.

arguments
    fDomain {mustBeNumeric, mustBeVector}
    nBasis (1,1) {mustBeInteger, mustBePositive}
    degree (1,1) {mustBeInteger, mustBeNonnegative} = 3
end

fDomain = double(fDomain(:)).';
order = degree + 1;

% patsy: n_inner_knots = df - order, then +1 because the intercept is excluded.
nInner = nBasis - order + 1;
if nInner < 0
    error('basisBspline:nBasisTooSmall', ...
        'Must have nBasis >= %d for degree=%d.', degree, degree);
end

knots = mvbInternal.bsplineKnots(fDomain, nBasis, degree);

if mvbInternal.curveFittingAvailable()
    % spcol(knots, k, tau) is the B-spline collocation matrix: entry (i,j) is
    % the j-th B-spline of order k = degree+1 evaluated at tau(i). spcol needs
    % tau nondecreasing, so sort going in and scatter back on the way out.
    [fDomainSorted, sortIdx] = sort(fDomain);
    Bsorted = spcol(knots, order, fDomainSorted);
    if size(Bsorted,2) ~= numel(knots) - order
        error('basisBspline:unexpectedSpcolSize', ...
            'spcol returned %d basis functions, expected %d.', ...
            size(Bsorted,2), numel(knots) - order);
    end
    B = zeros(size(Bsorted));
    B(sortIdx, :) = Bsorted;
else
    B = mvbInternal.bsplineDesign(fDomain, knots, degree);
end

% Drop the first basis function (patsy's include_intercept=False) and return
% one basis function per row.
basis = B(:, 2:end).';

end
