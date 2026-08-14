function out = mvCV(bayesModel, X, Y, varargin)
%MVCV Cross-Validation (CV) of a Multivariate Bayesian Regression Model
%
%   out = mvCV(bayesModel, X, Y, 'Name', Value, ...)
%
%   Inputs:
%     bayesModel     Handle to a Bayesian regression model-fitting function whose
%                    first argument is an nxp input matrix and whose second
%                    argument is an n-vector of numeric responses.
%     X              nxp matrix of predictors, where n is the total number of
%                    examples (training + test) and p is the number of inputs.
%     Y              nxq response matrix, where q is the number of
%                    multivariate/functional responses.
%
%   Name-value options:
%     'nTrain'          Number of examples in the training set. If empty,
%                       nTrain = n - nTest; unless nTest is also empty, in which
%                       case nTrain = n - floor(n/2).
%     'nTest'           Number of examples in the test set. If empty,
%                       nTest = n - nTrain.
%     'nRep'            Number of repetitions of the CV process (default 1).
%     'seed'            Randomization seed for the train/test split. If empty,
%                       no seed is set.
%     'coverageTarget'  Level of coverage desired (default 0.95).
%     'idxSamples'      Which posterior samples to use (default "default").
%     'uqTruncMethod'   "gaussian" or "empirical" (default "gaussian").
%     'multifidelity'  true or false to run MF (default false)
%
%   Any additional name-value pairs are forwarded to mvBayes (including
%   arguments to bayesModel).
%
%   Output:
%     out   Struct containing the out-of-sample RMSE for each replication,
%           fitting and prediction times, and other metrics.

% ---------------------------------------------------------------- Parse args
opts = struct( ...
    'nTrain',         [], ...
    'nTest',          [], ...
    'nRep',           1, ...
    'seed',           [], ...
    'coverageTarget', 0.95, ...
    'idxSamples',     "default", ...
    'uqTruncMethod',  "gaussian", ...
    'multifidelity', false);

knownNames = fieldnames(opts);
extraArgs  = {};
iArg = 1;
while iArg <= numel(varargin)
    name = varargin{iArg};
    if ~(ischar(name) || isstring(name))
        error('mvCV:badOption', 'Optional arguments must be name-value pairs.');
    end
    if iArg == numel(varargin)
        error('mvCV:badOption', 'Option "%s" is missing a value.', char(name));
    end
    idxMatch = find(strcmpi(char(name), knownNames), 1);
    if isempty(idxMatch)
        extraArgs(end+1:end+2) = varargin(iArg:iArg+1); %#ok<AGROW>
    else
        opts.(knownNames{idxMatch}) = varargin{iArg+1};
    end
    iArg = iArg + 2;
end

nTrain         = opts.nTrain;
nTest          = opts.nTest;
nRep           = opts.nRep;
seed           = opts.seed;
coverageTarget = opts.coverageTarget;
idxSamples     = opts.idxSamples;
uqTruncMethod  = lower(string(opts.uqTruncMethod));
MF = opts.multifidelity;

% -------------------------------------------------------------------- Setup
[n, ~] = size(X);
q      = size(Y, 2);
alpha  = 1 - coverageTarget;

if isempty(nTest)
    if isempty(nTrain)
        nTest  = floor(n / 2);   % half in test set
        nTrain = n - nTest;
    elseif nTrain >= n
        error('mvCV:badSplit', 'Must have nTrain < size(X, 1)');
    else
        nTest = n - nTrain;
    end
else
    if nTest >= n
        error('mvCV:badSplit', 'Must have nTest < size(X, 1)');
    elseif isempty(nTrain)
        nTrain = n - nTest;
    elseif nTrain + nTest > n
        error('mvCV:badSplit', 'Must have nTrain + nTest <= n');
    end
end

% ------------------------------------------------------------ Fold indices
if isempty(seed)
    rng('shuffle');
else
    rng(seed);
end

idxTest  = cell(nRep, 1);
idxTrain = cell(nRep, 1);
for r = 1:nRep
    idxTest{r} = randperm(n, nTest);
    remaining  = setdiff(1:n, idxTest{r});
    idxTrain{r} = remaining(randperm(numel(remaining), nTrain));
end

rng('shuffle'); 

% ------------------------------------------------------------------ Run CV
rmse          = zeros(nRep, 1);
rSquared      = zeros(nRep, 1);
coverage      = zeros(nRep, 1);
intervalWidth = zeros(nRep, 1);
intervalScore = zeros(nRep, 1);
fitTime       = zeros(nRep, 1);
predictTime   = zeros(nRep, 1);

for r = 1:nRep
    % Set up train/test split
    Xtrain = X(idxTrain{r}, :);
    Ytrain = Y(idxTrain{r}, :);
    Xtest  = X(idxTest{r},  :);
    Ytest  = Y(idxTest{r},  :);

    % Fit model
    startFit = tic;
    if MF
        fit = mvBayesMF(bayesModel, Xtrain, Ytrain, extraArgs{:});
    else
        fit = mvBayes(bayesModel, Xtrain, Ytrain, extraArgs{:});
    end
    fitTime(r) = toc(startFit);

    % Predict: preds is nSamples x nTest x q
    startPred = tic;
    preds = fit.predict(Xtest, 'idxSamples', idxSamples);
    predictTime(r) = toc(startPred);

    nSamples = size(preds, 1);
    nPred    = nSamples * nTest;

    Yhat = reshape(median(preds, 1), [nTest, q]);

    % Calculate RMSE and R-squared
    sqErr       = (Ytest - Yhat).^2;
    rmse(r)     = sqrt(mean(sqErr(:)));
    baseErr     = (Ytest - mean(Ytrain, 1)).^2;
    rSquared(r) = 1 - mean(sqErr(:)) / mean(baseErr(:));

    % Get truncation error for UQ
    switch uqTruncMethod
        case "gaussian"
            truncErrorVar = cov(fit.basisInfo.truncError);
            truncErrorMat = mvnDraw(truncErrorVar, nPred);
        case "empirical"
            idxResample   = randi(nTrain, [nPred, 1]);
            truncErrorMat = fit.basisInfo.truncError(idxResample, :);
        otherwise
            error('mvCV:badUqTruncMethod', ...
                'uqTruncMethod must be "gaussian" or "empirical".');
    end
    preds = preds + reshape(truncErrorMat, [nSamples, nTest, q]);
    clear truncErrorMat

    % Get regression error for UQ
    nBasis = fit.basisInfo.nBasis;
    coefsResidError = zeros(nSamples, nTest, nBasis);
    for k = 1:nBasis
        residSD = repmat(fit.bmList{k}.samples.residSD(:), 1, nTest);
        coefsResidError(:, :, k) = residSD .* randn(nSamples, nTest);
    end
    % (nSamples*nTest x nBasis) * (nBasis x q) -> nSamples x nTest x q
    residError = reshape(coefsResidError, [nPred, nBasis]) * fit.basisInfo.basis;
    clear coefsResidError
    preds = preds + reshape(residError, [nSamples, nTest, q]);
    clear residError

    % Calculate distance from posterior mean
    distBound = zeros(nTest, 1);
    for idx = 1:nTest
        predsIdx     = reshape(preds(:, idx, :), [nSamples, q]);
        distSamples  = sqrt(mean((predsIdx - Yhat(idx, :)).^2, 2));
        distBound(idx) = quantileLinear(distSamples, coverageTarget);
    end
    distTest = sqrt(mean((Ytest - Yhat).^2, 2));

    % Calculate UQ metrics
    distRatio        = distTest ./ distBound;
    coverage(r)      = mean(distRatio <= 1);
    intervalWidth(r) = exp(mean(log(distBound)));
    intervalScore(r) = intervalWidth(r) * ...
        exp(mean(log(distRatio) .* (distRatio > 1)) / alpha);
end

% ---------------------------------------------------------- Output results
out = struct();
out.rmse           = rmse;
out.rSquared       = rSquared;
out.coverageTarget = coverageTarget;
out.coverage       = coverage;
out.intervalWidth  = intervalWidth;
out.intervalScore  = intervalScore;
out.fitTime        = fitTime;
out.predictTime    = predictTime;
out.effectiveArgs  = struct( ...
    'nTrain',         nTrain, ...
    'nTest',          nTest, ...
    'nRep',           nRep, ...
    'seed',           seed, ...
    'coverageTarget', coverageTarget, ...
    'idxSamples',     idxSamples, ...
    'uqTruncMethod',  uqTruncMethod);

end

% =========================================================================
function Z = mvnDraw(Sigma, N)
%MVNDRAW Draw N rows from a zero-mean multivariate normal with covariance Sigma.
%   Uses an eigendecomposition so that rank-deficient covariance matrices
%   (common for truncation error) are handled without error.
Sigma = (Sigma + Sigma') / 2;
[V, D] = eig(Sigma);
d = max(real(diag(D)), 0);
A = real(V) * diag(sqrt(d));
Z = randn(N, size(Sigma, 1)) * A';
end

% =========================================================================
function qv = quantileLinear(x, p)
%QUANTILELINEAR Linear-interpolation quantile matching numpy.quantile defaults.
%   MATLAB's built-in quantile uses a different plotting position, so this
%   helper is used to keep results consistent with the original Python code.
x = sort(x(:));
n = numel(x);
if n == 1
    qv = x;
    return
end
h  = (n - 1) * p + 1;
lo = floor(h);
hi = ceil(h);
qv = x(lo) + (h - lo) * (x(hi) - x(lo));
end
