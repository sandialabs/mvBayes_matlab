function crps = crps(obs, ensemble)
% CRPS  Vectorized O(N*M*log M) CRPS using the sort trick
%
%   crps = crps(obs, ensemble)
%
%   obs      : [N,1] vector of observations
%   ensemble : [N,M] matrix of ensemble members
%   crps     : [N,1] vector of CRPS values
%
% Uses the identity:
%   E|X - X'| = (2/M^2) * sum_{k=1}^{M} (2k - M - 1) * x_(k)
% where x_(k) are the sorted ensemble values (ascending), per row.
% This avoids the O(M^2) pairwise matrix entirely.

obs = obs(:);
[N, M] = size(ensemble);

if numel(obs) ~= N
    error('Number of observations must match number of ensemble rows.');
end

% Term 1: E|X - y| per row
term1 = mean(abs(ensemble - obs), 2);            % [N,1]

% Term 2: E|X - X'| per row via sorted-sum trick
xsorted = sort(ensemble, 2);                      % [N,M] ascending
k = 1:M;                                          % 1xM
weights = (2*k - M - 1);                          % 1xM, ranges -(M-1) to (M-1)
term2 = (2/M^2) * sum(weights .* xsorted, 2);     % [N,1]

crps = term1 - 0.5*term2;
end