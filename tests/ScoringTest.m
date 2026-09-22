classdef ScoringTest < matlab.unittest.TestCase
    %SCORINGTEST Unit tests for the scoring and linear-algebra utilities.
    %
    %   Covers crps, energy_score, compute_r2, compute_r2_field, orthogonalize
    %   and isOrthogonal. These functions have no external dependencies, so they
    %   are checked against directly-computed reference values and mathematical
    %   identities.
    %
    %   Run with:  results = runtests('tests')      (or the "test" build task)

    properties (Constant)
        Tol = 1e-12
    end

    methods (TestClassSetup)
        function addPackageToPath(testCase)
            here = fileparts(mfilename('fullpath'));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fileparts(here)));
        end
    end

    methods (Test)

        % ---------------------------------------------------------- crps ----
        function crpsMatchesPairwiseDefinition(testCase)
            % crps must equal  E|X-y| - 0.5 E|X-X'|  computed directly.
            rng(0);
            N = 4; M = 25;
            ensemble = randn(N, M);
            obs = randn(N, 1);

            expected = zeros(N,1);
            for i = 1:N
                x = ensemble(i,:);
                term1 = mean(abs(x - obs(i)));
                D = abs(x(:) - x(:).');   % M x M pairwise
                term2 = mean(D(:));
                expected(i) = term1 - 0.5*term2;
            end

            actual = crps(obs, ensemble);
            testCase.verifySize(actual, [N 1]);
            testCase.verifyEqual(actual, expected, 'AbsTol', 1e-10);
        end

        function crpsAcceptsRowObs(testCase)
            % obs is flattened with (:), so a row vector is accepted.
            ensemble = [1 2 3; 4 5 6];
            testCase.verifyEqual(crps([1 4], ensemble), crps([1;4], ensemble), ...
                'AbsTol', testCase.Tol);
        end

        function crpsRejectsMismatchedObs(testCase)
            % Wrong number of observations is an error (untagged).
            testCase.verifyError(@() crps([1;2;3], randn(2, 10)), ?MException);
        end

        % --------------------------------------------------- energy_score ----
        function energyScoreMatchesPairwiseDefinition(testCase)
            testCase.assumeTrue(exist('pdist2', 'file') > 0, ...
                'Statistics and Machine Learning Toolbox (pdist2) not available.');
            rng(1);
            M = 20; N = 3; T = 4;
            ensemble = randn(M, N, T);
            obs = randn(N, T);

            expected = zeros(N,1);
            for i = 1:N
                X = reshape(ensemble(:,i,:), [M, T]);
                y = obs(i,:);
                term1 = mean(sqrt(sum((X - y).^2, 2)));
                D = sqrt(sum((reshape(X,[M,1,T]) - reshape(X,[1,M,T])).^2, 3));
                term2 = sum(D(:)) / (M^2);
                expected(i) = term1 - 0.5*term2;
            end

            actual = energy_score(obs, ensemble);
            testCase.verifySize(actual, [N 1]);
            testCase.verifyEqual(actual, expected, 'AbsTol', 1e-9);
        end

        function energyScoreZeroWhenEnsembleEqualsObs(testCase)
            testCase.assumeTrue(exist('pdist2', 'file') > 0, ...
                'Statistics and Machine Learning Toolbox (pdist2) not available.');
            % An ensemble whose members all equal the observation scores 0.
            M = 8; N = 2; T = 3;
            obs = [1 2 3; 4 5 6];
            ensemble = zeros(M, N, T);
            for m = 1:M
                ensemble(m,:,:) = reshape(obs, [1 N T]);
            end
            testCase.verifyEqual(energy_score(obs, ensemble), zeros(N,1), ...
                'AbsTol', testCase.Tol);
        end

        % --------------------------------------------------- compute_r2 ----
        function computeR2PerfectPrediction(testCase)
            y = randn(20, 3);
            testCase.verifyEqual(compute_r2(y, y), 1, 'AbsTol', testCase.Tol);
        end

        function computeR2MeanOnlyPredictionIsZero(testCase)
            rng(2);
            y = randn(50, 1);
            yhat = mean(y) * ones(size(y));
            testCase.verifyEqual(compute_r2(y, yhat), 0, 'AbsTol', 1e-12);
        end

        function computeR2ConstantTruthIsNaN(testCase)
            % Degenerate SStot == 0 branch returns NaN.
            y = 5 * ones(10, 1);
            testCase.verifyTrue(isnan(compute_r2(y, y + 0.1)));
        end

        function computeR2FlattensInput(testCase)
            % A matrix and its vectorized form give the same global R^2.
            rng(3);
            Yt = randn(6, 5);
            Yp = Yt + 0.1*randn(6, 5);
            testCase.verifyEqual(compute_r2(Yt, Yp), compute_r2(Yt(:), Yp(:)), ...
                'AbsTol', testCase.Tol);
        end

        % ----------------------------------------------- compute_r2_field ----
        function computeR2FieldMatchesRowwise(testCase)
            rng(4);
            Yt = randn(5, 30);
            Yp = Yt + 0.2*randn(5, 30);
            field = compute_r2_field(Yt, Yp);
            testCase.verifySize(field, [5 1]);
            for j = 1:5
                testCase.verifyEqual(field(j), compute_r2(Yt(j,:), Yp(j,:)), ...
                    'AbsTol', 1e-10);
            end
        end

        % ------------------------------------- orthogonalize / isOrthogonal ----
        function orthogonalizeProducesOrthonormalRows(testCase)
            rng(5);
            basis = randn(4, 30);
            newBasis = orthogonalize(basis);
            testCase.verifyTrue(isOrthogonal(newBasis));
            testCase.verifyEqual(newBasis * newBasis.', eye(4), 'AbsTol', 1e-12);
        end

        function orthogonalizePreservesSubspace(testCase)
            % Reconstruction of a signal lying in the row space is unchanged by
            % re-expressing it in the orthonormalized basis.
            rng(6);
            basis = randn(3, 20);
            Q = orthogonalize(basis);
            coefs = randn(1, 3);
            signal = coefs * basis;                 % in the row space
            reproj = (signal * Q.') * Q;            % project onto and back
            testCase.verifyEqual(reproj, signal, 'AbsTol', 1e-10);
        end

        function isOrthogonalTrueForIdentityRows(testCase)
            [Q, ~] = qr(randn(25, 5), 0);
            testCase.verifyTrue(isOrthogonal(Q.'));
        end

        function isOrthogonalFalseForNonOrthonormal(testCase)
            testCase.verifyFalse(isOrthogonal([1 0 0; 1 1 0]));
        end

    end
end
