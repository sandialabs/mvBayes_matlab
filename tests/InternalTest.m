classdef InternalTest < matlab.unittest.TestCase
    %INTERNALTEST Unit tests for the mvbInternal static helpers.
    %
    %   These helpers underpin mvBayes/mvBayesMF and have no external
    %   dependencies of their own, so they are tested directly.
    %
    %   Run with:  results = runtests('tests')      (or the "test" build task)

    properties (Constant)
        Tol = 1e-12
    end

    methods (TestClassSetup)
        function addPaths(testCase)
            here = fileparts(mfilename('fullpath'));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fileparts(here)));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(here, 'mocks')));
        end
    end

    methods (Test)

        % --------------------------------------------------- quantileLinear ----
        function quantileLinearKnownValues(testCase)
            % Linear-interpolation quantiles, matching numpy.percentile default.
            x = [1 2 3 4];
            testCase.verifyEqual(mvbInternal.quantileLinear(x, 0),   1, 'AbsTol', testCase.Tol);
            testCase.verifyEqual(mvbInternal.quantileLinear(x, 1),   4, 'AbsTol', testCase.Tol);
            testCase.verifyEqual(mvbInternal.quantileLinear(x, 0.5), 2.5, 'AbsTol', testCase.Tol);
            % h = (n-1)*p + 1 = 3*0.25 + 1 = 1.75 -> x(1) + .75*(x(2)-x(1)) = 1.75
            testCase.verifyEqual(mvbInternal.quantileLinear(x, 0.25), 1.75, 'AbsTol', testCase.Tol);
        end

        function quantileLinearVectorProbs(testCase)
            x = 0:10;
            q = mvbInternal.quantileLinear(x, [0 0.5 1]);
            testCase.verifyEqual(q, [0 5 10], 'AbsTol', testCase.Tol);
        end

        function quantileLinearSingleElement(testCase)
            % n == 1 branch returns the sole value for any probability.
            testCase.verifyEqual(mvbInternal.quantileLinear(7, 0.3), 7, 'AbsTol', testCase.Tol);
        end

        % ------------------------------------------------- resolveIdxSamples ----
        function resolveIdxSamplesFinal(testCase)
            testCase.verifyEqual(mvbInternal.resolveIdxSamples("final", 100), 100);
        end

        function resolveIdxSamplesDefault(testCase)
            testCase.verifyEqual(mvbInternal.resolveIdxSamples("default", 5), 1:5);
        end

        function resolveIdxSamplesNumericBecomesRow(testCase)
            out = mvbInternal.resolveIdxSamples([3;7;9], 100);
            testCase.verifyEqual(out, [3 7 9]);
            testCase.verifySize(out, [1 3]);
        end

        % ------------------------------------------------------ isModelParam ----
        function isModelParamTrueForScalarsAndVectors(testCase)
            testCase.verifyTrue(mvbInternal.isModelParam(3.14));
            testCase.verifyTrue(mvbInternal.isModelParam(1:10));
            testCase.verifyTrue(mvbInternal.isModelParam((1:5).'));
            testCase.verifyTrue(mvbInternal.isModelParam(true));
            testCase.verifyTrue(mvbInternal.isModelParam([true false true]));
        end

        function isModelParamFalseForOtherTypes(testCase)
            testCase.verifyFalse(mvbInternal.isModelParam([]));
            testCase.verifyFalse(mvbInternal.isModelParam(magic(3)));   % matrix
            testCase.verifyFalse(mvbInternal.isModelParam('abc'));
            testCase.verifyFalse(mvbInternal.isModelParam("abc"));
            testCase.verifyFalse(mvbInternal.isModelParam({1,2}));
            testCase.verifyFalse(mvbInternal.isModelParam(struct('a',1)));
            testCase.verifyFalse(mvbInternal.isModelParam(@sin));
        end

        % ------------------------------------------------------ bsplineKnots ----
        function bsplineKnotsClampedAndSorted(testCase)
            degree = 3;
            nBasis = 8;
            fDomain = linspace(0, 1, 99);
            knots = mvbInternal.bsplineKnots(fDomain, nBasis, degree);

            % Nondecreasing.
            testCase.verifyGreaterThanOrEqual(diff(knots), 0);
            % Length: nInner + 2*(degree+1) interior + clamped endpoints, where
            % nInner = nBasis - (degree+1) + 1.
            order = degree + 1;
            nInner = nBasis - order + 1;
            testCase.verifyEqual(numel(knots), nInner + 2*order);
            % Endpoints clamped: repeated order (= degree+1) times.
            testCase.verifyEqual(sum(knots == min(fDomain)), order);
            testCase.verifyEqual(sum(knots == max(fDomain)), order);
        end

        % ------------------------------------- hasSamples / getSamples / set ----
        function samplesHelpersOnStruct(testCase)
            bm = struct('samples', struct('residSD', (1:4).'));
            testCase.verifyTrue(mvbInternal.hasSamples(bm));
            s = mvbInternal.getSamples(bm);
            testCase.verifyEqual(s.residSD, (1:4).');
            testCase.verifyTrue(mvbInternal.hasSamplesField(s, 'residSD'));
            testCase.verifyFalse(mvbInternal.hasSamplesField(s, 'missing'));
        end

        function hasSamplesFalseWhenEmpty(testCase)
            bm = struct('samples', []);
            testCase.verifyFalse(mvbInternal.hasSamples(bm));
            testCase.verifyEqual(mvbInternal.getSamples(struct('other', 1)), []);
        end

        function samplesHelpersOnObject(testCase)
            bm = MockBayesModel(rand(10,2), rand(10,1), 12);
            testCase.verifyTrue(mvbInternal.hasSamples(bm));
            s = mvbInternal.getSamples(bm);
            testCase.verifyTrue(mvbInternal.hasSamplesField(s, 'residSD'));
        end

        function setSamplesRoundTrip(testCase)
            bm = MockBayesModel(rand(10,2), rand(10,1), 6);
            newS = bayesModelSamples(struct('residSD', ones(6,1)));
            bm = mvbInternal.setSamples(bm, newS);
            testCase.verifyEqual(mvbInternal.getSamples(bm).residSD, ones(6,1));
        end

        function setSamplesErrorsWhenNoProperty(testCase)
            bm = NoSamplesModel();
            testCase.verifyError(@() mvbInternal.setSamples(bm, struct('a',1)), ...
                'mvBayes:noSamplesProperty');
        end

        % --------------------------------------------------- methodInputNames ----
        function methodInputNamesKnownMethod(testCase)
            bm = MockBayesModel(rand(10,2), rand(10,1), 5);
            names = mvbInternal.methodInputNames(bm, 'predict');
            % Signature is predict(obj, Xtest, idxSamples); the declared inputs
            % must include the test input and the idxSamples argument that
            % mvBayes.predict looks for by name.
            testCase.verifyTrue(ismember('Xtest', names));
            testCase.verifyTrue(ismember('idxSamples', names));
        end

        function methodInputNamesUnknownMethod(testCase)
            bm = MockBayesModel(rand(10,2), rand(10,1), 5);
            testCase.verifyEqual(mvbInternal.methodInputNames(bm, 'noSuchMethod'), {});
        end

    end
end
