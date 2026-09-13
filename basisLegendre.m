function basis = basisLegendre(fDomain, nLegendre, pFourier)
%BASISLEGENDRE Legendre polynomial basis, one basis function per row.
%
%   basis = basisLegendre(fDomain, nLegendre)
%   basis = basisLegendre(fDomain, nLegendre, pFourier)
%
%   Row n is the ordinary Legendre polynomial P_n evaluated at
%   2*(fDomain/pFourier) - 1, for n = 1 ... 2*nLegendre. The degree-0
%   (constant) polynomial is excluded, matching the Python implementation.
%
%   fDomain   : vector of points at which to evaluate the basis.
%   nLegendre : half the number of basis functions (2*nLegendre rows).
%   pFourier  : period used to rescale fDomain onto [-1, 1] (default 1).
%
%   basis     : 2*nLegendre x numel(fDomain) matrix, one polynomial per row.
%
%   Note: MATLAB's built-in legendre(n, x) returns the associated Legendre
%   functions of degree n for orders m = 0 ... n; its first row (m = 0) is the
%   ordinary Legendre polynomial, which is the only one used here.

arguments
    fDomain {mustBeNumeric, mustBeVector}
    nLegendre (1,1) {mustBeInteger, mustBePositive}
    pFourier (1,1) {mustBeNumeric, mustBePositive} = 1
end

fDomain = double(fDomain(:)).';
fDomainScaled = 2 * (fDomain / pFourier) - 1;

% legendre() requires |x| <= 1; guard against round-off at the endpoints.
fDomainScaled = min(max(fDomainScaled, -1), 1);

nRow = 2 * nLegendre;
basis = zeros(nRow, numel(fDomain));
for i = 1:nRow
    tmp = legendre(i, fDomainScaled);
    basis(i, :) = tmp(1, :);     % order m = 0
end

end
