classdef SamplesTest < matlab.unittest.TestCase
    %SAMPLESTEST Unit tests for the bayesModelSamples dynamic container.
    %
    %   Run with:  results = runtests('tests')      (or the "test" build task)

    methods (TestClassSetup)
        function addPackageToPath(testCase)
            here = fileparts(mfilename('fullpath'));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fileparts(here)));
        end
    end

    methods (Test)

        function emptyConstruction(testCase)
            s = bayesModelSamples();
            testCase.verifyEmpty(properties(s));
        end

        function constructFromStructCopiesFields(testCase)
            src = struct('residSD', (1:5).', 'beta', magic(3));
            s = bayesModelSamples(src);
            testCase.verifyEqual(s.residSD, (1:5).');
            testCase.verifyEqual(s.beta, magic(3));
            testCase.verifyTrue(isprop(s, 'residSD'));
            testCase.verifyTrue(isprop(s, 'beta'));
        end

        function dynamicPropertyOnAssignment(testCase)
            % Assigning an undeclared field creates the property (subsasgn).
            s = bayesModelSamples();
            s.residSD = rand(20, 1);
            testCase.verifyTrue(isprop(s, 'residSD'));
            testCase.verifySize(s.residSD, [20 1]);
        end

        function toStructRoundTrips(testCase)
            s = bayesModelSamples();
            s.a = 1:3;
            s.b = "hello";
            out = s.toStruct();
            testCase.verifyTrue(isstruct(out));
            testCase.verifyEqual(out.a, 1:3);
            testCase.verifyEqual(out.b, "hello");
        end

        function handleSemanticsShareState(testCase)
            % bayesModelSamples is a handle class: a "copy" is a reference.
            s = bayesModelSamples();
            s.residSD = 1;
            ref = s;
            ref.residSD = 99;
            testCase.verifyEqual(s.residSD, 99);
        end

        function badInputErrors(testCase)
            testCase.verifyError(@() bayesModelSamples(42), ...
                'bayesModelSamples:badInput');
        end

    end
end
