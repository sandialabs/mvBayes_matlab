function [Psi, dopt, Xi, energy] = enrich_basis_multifidelity(Phi, y_bar, Z, mL, energy_tol)
% ENRICH_BASIS_MULTIFIDELITY  Low-fidelity enhancement of a
% high-fidelity POD basis, i.e. the "Constrained-POD" (C-POD) step of
% Benamara et al. 2017, Section 2.1.2.
%
%   [Psi, dopt, Xi, energy] = enrich_pod_basis_multifidelity(Phi, y_bar, Z, mL, energy_tol)
%
% Inputs
%   Phi        : n x MH   orthonormal HF basis from build_hf_pod_basis
%   y_bar      : n x 1    HF centering snapshot
%   Z          : n x ML   abundant low-fidelity snapshots (columns)
%   mL         : max number of extra ("low-fidelity") modes to keep
%   energy_tol : optional relative-energy truncation criterion in (0,1]
%                (if given, the number of retained modes is
%                 min(mL, first mode index reaching this cumulated energy))
%
% Outputs
%   Psi    : n x (MH+m) hierarchised multi-fidelity POD basis, Psi = [Phi | Xi]
%   dopt   : n x 1   optimal shift solving Eq. (2.15)
%   Xi     : n x m   extra orthonormal modes completing Phi (Eq. 2.19)
%   energy : relative cumulated eigenvalue energy of the retained Xi modes
%
% This routine solves
%   min_{d,Xi}  J(d,Xi)   s.t.  Xi'*Xi = I,  Phi'*Xi = 0
% (Eq. 2.4-2.14), whose closed-form solution is:
%   dopt = ML/(MH+ML) * mean( Z_perp )                      (Eq. 2.15)
%   Xi   = top mL left singular vectors of the "constrained" POD
%          problem posed on {z_i_perp - dopt} augmented with the
%          extra centering point z0_perp = sqrt(MH+1)*dopt   (Eq. 2.16-2.19)
%
% Because every z_i_perp = (I - Phi*Phi')*z_i already lies in the
% orthogonal complement of span(Phi), and dopt is constructed to also lie
% in that same orthogonal complement, the constrained POD problem posed
% on Q2 coordinates (Appendix B) is *equivalent* to an ordinary SVD
% performed directly on the (already Phi-orthogonal) snapshot matrix in
% the ambient R^n space -- this avoids ever having to form the large
% (n x (n-MH)) explicit null-space basis Q2, which is the practical trick
% used in this implementation.

MH = size(Phi,2);
ML = size(Z,2);

% Center low-fidelity snapshots with the SAME centering snapshot y_bar
% used for the high-fidelity data, then project onto the orthogonal
% complement of Phi:   z_i_perp = (I - Phi*Phi')*(z_i - y_bar)   (Eq. 2.12)
Zc    = Z - y_bar;
Zperp = Zc - Phi*(Phi'*Zc);

% Optimal shift dopt (Eq. 2.15), solution of the sub-problem min_d J(d,Xi)
dopt = (ML/(MH+ML)) * mean(Zperp, 2);

% Extra centering snapshot introduced to make the sum symmetric
% over i = 0..ML  (Eq. 2.16)
z0_perp = sqrt(MH+1) * dopt;

% Snapshot set for the completion POD problem (Eq. 2.17-2.19):
%   v_0 = z0_perp - dopt ,  v_i = z_i_perp - dopt , i = 1..ML
W = [ (z0_perp - dopt) , (Zperp - dopt) ];   % n x (ML+1)

% Classical POD (SVD) on W -> left singular vectors already lie in
% (Im Phi)^perp, so they can be used directly as the extra modes.
[U, S, ~] = svd(W, 'econ');
sv = diag(S);
cum_energy = cumsum(sv.^2) / sum(sv.^2);

if nargin >= 5 && ~isempty(energy_tol)
    m_energy = find(cum_energy >= energy_tol, 1, 'first');
    if isempty(m_energy), m_energy = numel(sv); end
    m = min(mL, m_energy);
else
    m = min(mL, numel(sv));
end

Xi = U(:, 1:m);
energy = cum_energy(1:m);

% Re-orthonormalize against Phi defensively (should already hold to
% machine precision) then assemble the hierarchised basis.
Xi = Xi - Phi*(Phi'*Xi);
for k = 1:size(Xi,2)
    v = Xi(:,k);
    for j = 1:k-1
        v = v - (Xi(:,j)'*v)*Xi(:,j);
    end
    nv = norm(v);
    if nv > 1e-12
        Xi(:,k) = v/nv;
    end
end

Psi = [Phi, Xi];
end
