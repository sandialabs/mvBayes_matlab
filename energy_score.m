function es = energy_score(obs, ensemble)
% ENERGY_SCORE  Multivariate generalization of CRPS (captures dependence)
%   obs      : [N, T]     observed curves
%   ensemble : [M, N, T]  ensemble of curves
%   es       : [N,1]      energy score per case

[M, N, T] = size(ensemble);
es = zeros(N,1);

for i = 1:N
    X = squeeze(ensemble(:,i,:));      % M x T
    y = obs(i,:);                       % 1 x T

    term1 = mean(sqrt(sum((X - y).^2, 2)));   % E||X - y||

    % E||X - X'|| via pairwise distances
    D = pdist2(X, X);                          % M x M
    term2 = sum(D(:)) / (M^2);

    es(i) = term1 - 0.5*term2;
end
end