function tf = isOrthogonal(basis, tol)
%ISORTHOGONAL Check whether the rows of a basis are orthonormal.
%
%   tf = isOrthogonal(basis)
%   tf = isOrthogonal(basis, tol)
%
%   basis : k x q matrix whose rows are basis functions.
%   tol   : absolute tolerance (default 1e-10).
%
%   Replicates numpy.allclose(basis @ basis.T, eye(k), atol=tol), including
%   numpy's default relative tolerance of 1e-5, which loosens the test on the
%   diagonal only.

arguments
    basis {mustBeNumeric}
    tol (1,1) {mustBeNumeric, mustBeNonnegative} = 1e-10
end

G = basis * basis.';
I = eye(size(basis,1));

tf = all(abs(G - I) <= tol + 1e-5 * abs(I), 'all');

end
