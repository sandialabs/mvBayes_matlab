classdef MvCVTest < matlab.unittest.TestCase
    %MVCVTEST Tests for the mvCV cross-validation driver using a mock model.
    %
    %   Run with:  results = runtests('tests')      (or the "test" build task)

    properties
        X
        Y
        model
        nBasis = 3
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
            rng(11);
            n = 50; p = 3; q = 10;
            X = rand(n, p);
            W = randn(p, testCase.nBasis);
            basis = orthogonalize(randn(testCase.nBasis, q));
            testCase.X = X;
            testCase.Y = (X * W) * basis;
            testCase.model = @(Xin, yin) MockBayesModel(Xin, yin, 30);
        end
    end

    methods
        function out = runCV(testCase, varargin)
            % mvCV computes an energy score, which uses pdist2 from the
            % Statistics and Machine Learning Toolbox.
            testCase.assumeTrue(exist('pdist2', 'file') > 0, ...
                'Statistics and Machine Learning Toolbox (pdist2) not available.');
            out = mvCV(testCase.model, testCase.X, testCase.Y, ...
                'nBasis', testCase.nBasis, ...
                'residSDExtract', @(m) m.samples.residSD, ...
                varargin{:});
        end
    end

    methods (Test)

        function returnsAllDocumentedFields(testCase)
            nRep = 3;
            out = testCase.runCV('nRep', nRep, 'nTrain', 30, 'nTest', 15, 'seed', 1);
            fields = {'rmse','rSquared','coverageTarget','coverage', ...
                'energyScore','crps','intervalWidth','intervalScore', ...
                'fitTime','predictTime','effectiveArgs'};
            for i = 1:numel(fields)
                testCase.verifyTrue(isfield(out, fields{i}), ...
                    sprintf('missing field %s', fields{i}));
            end
            % Per-rep vectors are nRep x 1 and finite.
            for f = {'rmse','rSquared','coverage','energyScore','intervalWidth', ...
                    'intervalScore','fitTime','predictTime'}
                testCase.verifySize(out.(f{1}), [nRep 1], ...
                    sprintf('size of %s', f{1}));
            end
            testCase.verifyTrue(all(isfinite(out.rmse)));
            % crps is an alias of energyScore.
            testCase.verifyEqual(out.crps, out.energyScore);
            % effectiveArgs echoes the resolved split.
            testCase.verifyEqual(out.effectiveArgs.nTrain, 30);
            testCase.verifyEqual(out.effectiveArgs.nTest, 15);
            testCase.verifyEqual(out.effectiveArgs.nRep, nRep);
        end

        function seedMakesRunReproducible(testCase)
            out1 = testCase.runCV('nRep', 2, 'nTrain', 30, 'nTest', 15, 'seed', 42);
            out2 = testCase.runCV('nRep', 2, 'nTrain', 30, 'nTest', 15, 'seed', 42);
            testCase.verifyEqual(out1.rmse, out2.rmse, 'AbsTol', 1e-10);
            testCase.verifyEqual(out1.rSquared, out2.rSquared, 'AbsTol', 1e-10);
        end

        function goodPredictionGivesHighRSquared(testCase)
            % Noise-free linear response -> CV R^2 should be high.
            out = testCase.runCV('nRep', 1, 'nTrain', 35, 'nTest', 15, 'seed', 3);
            testCase.verifyGreaterThan(out.rSquared, 0.9);
        end

        function empiricalUqTruncMethodRuns(testCase)
            out = testCase.runCV('nRep', 1, 'nTrain', 30, 'nTest', 15, ...
                'seed', 5, 'uqTruncMethod', "empirical");
            testCase.verifyEqual(out.effectiveArgs.uqTruncMethod, "empirical");
            testCase.verifyTrue(isfinite(out.coverage));
        end

        function gaussianUqTruncMethodRuns(testCase)
            out = testCase.runCV('nRep', 1, 'nTrain', 30, 'nTest', 15, ...
                'seed', 5, 'uqTruncMethod', "gaussian");
            testCase.verifyEqual(out.effectiveArgs.uqTruncMethod, "gaussian");
        end

        % ------------------------------------------------------- error paths ----
        function badOptionNonStringName(testCase)
            testCase.verifyError( ...
                @() mvCV(testCase.model, testCase.X, testCase.Y, 5, 10), ...
                'mvCV:badOption');
        end

        function badOptionMissingValue(testCase)
            testCase.verifyError( ...
                @() mvCV(testCase.model, testCase.X, testCase.Y, 'nRep'), ...
                'mvCV:badOption');
        end

        function badSplitNTrainTooLarge(testCase)
            n = size(testCase.X, 1);
            testCase.verifyError( ...
                @() mvCV(testCase.model, testCase.X, testCase.Y, 'nTrain', n), ...
                'mvCV:badSplit');
        end

        function badSplitNTestTooLarge(testCase)
            n = size(testCase.X, 1);
            testCase.verifyError( ...
                @() mvCV(testCase.model, testCase.X, testCase.Y, 'nTest', n), ...
                'mvCV:badSplit');
        end

        function badSplitSumTooLarge(testCase)
            n = size(testCase.X, 1);
            testCase.verifyError( ...
                @() mvCV(testCase.model, testCase.X, testCase.Y, ...
                    'nTrain', n-2, 'nTest', 5), ...
                'mvCV:badSplit');
        end

        function badUqTruncMethod(testCase)
            % Fires inside the per-rep loop, so a valid split is needed.
            testCase.verifyError( ...
                @() testCase.runCV('nRep', 1, 'nTrain', 30, 'nTest', 15, ...
                    'seed', 1, 'uqTruncMethod', "notamethod"), ...
                'mvCV:badUqTruncMethod');
        end

    end
end
