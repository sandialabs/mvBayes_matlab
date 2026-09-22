classdef MockBayesModel
    %MOCKBAYESMODEL Minimal deterministic univariate Bayesian regression model.
    %
    %   A lightweight stand-in for a real emulator (e.g. @bass, @bppr) so that
    %   the mvBayes / mvCV pipeline can be exercised end-to-end in tests without
    %   any external dependency. It satisfies exactly the contract mvBayes needs:
    %
    %     * it is constructed as bayesModel(X, y) with an n x p input matrix X
    %       and an n x 1 response y;
    %     * it exposes a settable `samples` property (so mvbInternal.setSamples
    %       can attach a samples container);
    %     * `samples` is a bayesModelSamples holding `residSD` (nSamples x 1) and
    %       the posterior coefficient draws `beta` (nSamples x (p+1));
    %     * predict(Xtest, idxSamples) returns an nSamplesUsed x nTest matrix,
    %       taking idxSamples as an optional positional argument (mirroring how
    %       BASS's predict takes mcmc_use positionally), so the positional branch
    %       of mvbInternal.idxSamplesArgs is exercised.
    %
    %   The "posterior" is a plain least-squares fit of y on [1 X], plus small
    %   deterministic (seeded) Gaussian jitter around it, so predictions are
    %   reproducible and a linear response is recovered almost exactly.

    properties
        beta0       % least-squares coefficients, (p+1) x 1 (intercept first)
        samples     % bayesModelSamples: beta (nSamples x (p+1)), residSD (nSamples x 1)
        nSamples
    end

    methods
        function obj = MockBayesModel(X, y, nSamples)
            arguments
                X {mustBeNumeric}
                y {mustBeNumeric}
                nSamples (1,1) {mustBeNumeric, mustBePositive} = 50
            end

            y = y(:);
            n = size(X, 1);
            Xd = [ones(n,1), X];          % design matrix with intercept
            obj.beta0 = Xd \ y;           % (p+1) x 1 least-squares fit

            resid = y - Xd * obj.beta0;
            sigma = std(resid, 1);        % population SD, matches numpy default
            if sigma == 0
                sigma = 1e-6;             % keep residSD strictly positive
            end

            obj.nSamples = nSamples;

            % Deterministic posterior draws: seed locally so repeated fits of the
            % same component are reproducible, then restore the global RNG.
            rngState = rng;
            cleaner = onCleanup(@() rng(rngState));
            rng(1234, 'twister');

            p1 = numel(obj.beta0);
            betaDraws = obj.beta0.' + 0.01 * sigma * randn(nSamples, p1);
            % residSD draws: positive, centered near the fit residual SD.
            residSDDraws = sigma * (0.9 + 0.2 * rand(nSamples, 1));

            s = bayesModelSamples();
            s.beta = betaDraws;
            s.residSD = residSDDraws;
            obj.samples = s;
        end

        function pred = predict(obj, Xtest, idxSamples, varargin)
            %PREDICT Posterior predictions, nSamplesUsed x nTest.
            %
            %   Accepts idxSamples either positionally --  predict(Xtest, idx)
            %   -- or as a name-value pair -- predict(Xtest, 'idxSamples', idx).
            %   Both conventions are used by mvBayes depending on how it detects
            %   the predict signature, so the mock supports both. Keeping
            %   idxSamples as a named input (rather than folding it into
            %   varargin) means mvbInternal.methodInputNames still reports it,
            %   so mvBayes recognizes and forwards it. idxSamples may be
            %   "default" (all draws), "final" (last draw), or a numeric
            %   index/vector.
            if nargin < 3
                idxSamples = "default";
            end
            % Name-value form: idxSamples arrived as the literal name and the
            % real value is the next argument.
            if (ischar(idxSamples) || isstring(idxSamples)) ...
                    && strcmpi(idxSamples, "idxSamples") && ~isempty(varargin)
                idxSamples = varargin{1};
            end

            if (ischar(idxSamples) || isstring(idxSamples)) && strcmpi(idxSamples, "default")
                idx = 1:obj.nSamples;
            elseif (ischar(idxSamples) || isstring(idxSamples)) && strcmpi(idxSamples, "final")
                idx = obj.nSamples;
            else
                idx = double(idxSamples(:)).';
            end

            beta = obj.samples.beta(idx, :);      % nUse x (p+1)
            Xd = [ones(size(Xtest,1),1), Xtest];  % nTest x (p+1)
            pred = beta * Xd.';                   % nUse x nTest
        end
    end
end
