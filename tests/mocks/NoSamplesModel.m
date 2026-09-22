classdef NoSamplesModel
    %NOSAMPLESMODEL Object with no settable `samples`, for error-path tests.
    %
    %   mvbInternal.setSamples(bm, ...) tries `bm.samples = ...` and, if that
    %   throws, wraps it as mvBayes:noSamplesProperty. This class has no
    %   `samples` property, so assigning one raises, exercising that branch.

    properties
        value = 0
    end
end
