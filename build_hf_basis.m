function [Phi, y_bar, explained] = build_hf_basis(Y)
% BUILD_HF_POD_BASIS  High-fidelity decomposition and centering
% snapshot selection (Benamara et al. 2017, Section 2.1.1).
%
%   [Phi, y_bar] = build_hf_basis(Y)
%
%   Y     : n x MH matrix, MH sparse high-fidelity snapshots (columns)
%   Phi   : n x MH orthonormal basis spanning span(Y - y_bar)   (Eq. 2.3, "Q1")
%   y_bar : n x 1 centering snapshot (here the usual mean mu(y), Eq. 2.1)
%
% Since no truncation is performed on Phi, the projection error on the
% known high-fidelity snapshots is exactly zero (Eq. 2.2), which is the
% key property exploited later to build a *hierarchised* multi-fidelity
% basis: truncating the completed basis Psi never re-introduces error on
% the interpolation of the high-fidelity data.

y_bar = mean(Y,2);
Ybar  = Y - y_bar;

% Economy-size QR: Ybar = [Phi] * [R],  Phi'*Phi = I_MH
% (equivalent to Eq. 2.3, [Y] = [Q1|Q2][R], keeping only Q1 = Phi)
[Phi, R] = qr(Ybar, 0);
[n, ~] = size(Ybar);

[~, S, ~] = svd(R);
latent = diag(S).^2 / (n - 1); 
explained = cumsum(latent)/sum(latent);
no = find(explained <= 0.999, 1, 'last');

MH = size(Y,2);
Phi = Phi(:, 1:no);
end
