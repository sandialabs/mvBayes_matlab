function newBasis = orthogonalize(basis)
%ORTHOGONALIZE Orthonormalize the rows of a basis via QR decomposition.
%
%   newBasis = orthogonalize(basis)
%
%   basis    : k x q matrix whose rows are (not necessarily orthogonal) basis
%              functions, with k <= q.
%   newBasis : k x q matrix with orthonormal rows spanning the same subspace.
%
%   Column signs follow LAPACK's Householder QR, as numpy.linalg.qr does; the
%   sign of a basis function is a convention and does not affect the subspace
%   or any reconstruction built from it.

arguments
    basis {mustBeNumeric}
end

[Q, ~] = qr(basis.', 0);
newBasis = Q.';

end
