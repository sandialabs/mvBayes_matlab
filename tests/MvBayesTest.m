classdef MvBayesTest < matlab.unittest.TestCase
    %MVBAYESTEST End-to-end tests for the mvBayes pipeline using a mock model.
    %
    %   MockBayesModel (tests/mocks) stands in for a real emulator so the fit,
    %   predict and mvSobol code paths run without BASS/BPPR. The response is a
    %   noise-free linear map of the inputs, which a linear mock recovers almost
    %   exactly, so accuracy assertions are meaningful.
    %
    %   Run with:  results = runtests('tests')      (or the "test" build task)

    properties
        X
        Y
        nBasis = 3
        nSamplesModel = 40
    end

    methods (TestClassSetup)
        function addPaths(testCase)
            here = fileparts(mfilename('fullpath'));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fileparts(here)));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(here, 'mocks')));
        end

        function buildData(testCase)
            rng(7);
            n = 60; p = 3; q = 12;
            X = rand(n, p);
            % Coefficients are linear in X; map them through a fixed basis so the
            % response is genuinely multivariate but perfectly linear in X.
            W = randn(p, testCase.nBasis);
            basis = orthogonalize(randn(testCase.nBasis, q));
            coefs = X * W;                       % n x nBasis
            testCase.X = X;
            testCase.Y = coefs * basis;          % n x q, noise-free
        end
    end

    methods
        function fit = makeFit(testCase, varargin)
            nS = testCase.nSamplesModel;
            model = @(Xin, yin) MockBayesModel(Xin, yin, nS);
            fit = mvBayes(model, testCase.X, testCase.Y, ...
                'nBasis', testCase.nBasis, ...
                'residSDExtract', @(m) m.samples.residSD, ...
                varargin{:});
        end
    end

    methods (Test)

        function fitPopulatesModelList(testCase)
            fit = testCase.makeFit();
            testCase.verifyEqual(numel(fit.bmList), testCase.nBasis);
            testCase.verifyEqual(fit.nSamples, testCase.nSamplesModel);
            testCase.verifyEqual(fit.basisInfo.nBasis, testCase.nBasis);
        end

        function predictReturnsExpectedShape(testCase)
            fit = testCase.makeFit();
            Xtest = rand(15, size(testCase.X, 2));
            Ypost = fit.predict(Xtest);
            % nSamples x nTest x q
            testCase.verifySize(Ypost, [testCase.nSamplesModel, 15, size(testCase.Y,2)]);
        end

        function predictMeanOnlySqueezes(testCase)
            fit = testCase.makeFit();
            Xtest = rand(15, size(testCase.X, 2));
            Ymean = fit.predict(Xtest, 'returnMeanOnly', true);
            testCase.verifySize(Ymean, [15, size(testCase.Y,2)]);
        end

        function predictReturnPostCoefsStruct(testCase)
            fit = testCase.makeFit();
            Xtest = rand(8, size(testCase.X, 2));
            out = fit.predict(Xtest, 'returnPostCoefs', true);
            testCase.verifyTrue(isstruct(out));
            testCase.verifyTrue(isfield(out, 'Ypost') && isfield(out, 'postCoefs'));
            testCase.verifySize(out.Ypost, [testCase.nSamplesModel, 8, size(testCase.Y,2)]);
            testCase.verifySize(out.postCoefs, [testCase.nSamplesModel, 8, testCase.nBasis]);
        end

        function predictRecoversLinearResponse(testCase)
            % Noise-free linear problem: posterior mean should match truth well.
            fit = testCase.makeFit();
            Ymean = fit.predict(testCase.X, 'returnMeanOnly', true);
            r2 = compute_r2(testCase.Y, Ymean);
            testCase.verifyGreaterThan(r2, 0.99);
        end

        function addResidErrorRunsAndInflatesSpread(testCase)
            fit = testCase.makeFit();
            Xtest = rand(10, size(testCase.X, 2));
            rng(0);
            base = fit.predict(Xtest);
            rng(0);
            withErr = fit.predict(Xtest, 'addResidError', true);
            testCase.verifySize(withErr, size(base));
            % Added residual error should, on average, increase the posterior
            % spread across samples.
            testCase.verifyGreaterThan(mean(std(withErr,0,1),'all'), ...
                mean(std(base,0,1),'all'));
        end

        function addTruncErrorRuns(testCase)
            fit = testCase.makeFit();
            Xtest = rand(10, size(testCase.X, 2));
            Ypost = fit.predict(Xtest, 'addTruncError', true);
            testCase.verifySize(Ypost, [testCase.nSamplesModel, 10, size(testCase.Y,2)]);
            testCase.verifyTrue(all(isfinite(Ypost), 'all'));
        end

        function predictBadIdxSamplesErrors(testCase)
            % A non-coercible idxSamples raises mvBayes:badIdxSamples, but only
            % once idxSamplesArg is a recognized predict argument.
            fit = testCase.makeFit('idxSamplesArg', 'idxSamples');
            testCase.verifyError( ...
                @() fit.predict(testCase.X, 'idxSamples', {struct('a',1)}), ...
                ?MException);
        end

        function nCoresAdjustClamps(testCase)
            fit = testCase.makeFit();
            % Never exceeds the number of components.
            testCase.verifyLessThanOrEqual(fit.nCoresAdjust(1000), testCase.nBasis);
            testCase.verifyGreaterThanOrEqual(fit.nCoresAdjust(1000), 1);
        end

        function plotSobolBeforeMvSobolErrors(testCase)
            fit = testCase.makeFit();
            testCase.verifyError(@() fit.plotSobol(), 'mvBayes:noSobol');
        end

        function mvSobolMonteCarloPopulatesIndices(testCase)
            testCase.assumeTrue(exist('sobolset', 'file') > 0, ...
                'Statistics and Machine Learning Toolbox (sobolset) not available.');
            fit = testCase.makeFit();
            fit = fit.mvSobol(true, 2^9, "final");   % nMC set -> Monte Carlo path
            p = size(testCase.X, 2);
            testCase.verifySize(fit.firstOrderSobol, [1, p, size(testCase.Y,2)]);
            testCase.verifySize(fit.totalOrderSobol, [1, p, size(testCase.Y,2)]);
            testCase.verifySize(fit.varTotal, [1, size(testCase.Y,2)]);
            testCase.verifyTrue(all(isfinite(fit.firstOrderSobol), 'all'));
        end

        function mvSobolFirstOrderOnly(testCase)
            testCase.assumeTrue(exist('sobolset', 'file') > 0, ...
                'Statistics and Machine Learning Toolbox (sobolset) not available.');
            fit = testCase.makeFit();
            fit = fit.mvSobol(false, 2^9, "final");
            testCase.verifyNotEmpty(fit.firstOrderSobol);
            testCase.verifyEmpty(fit.totalOrderSobol);
        end

    end
end
