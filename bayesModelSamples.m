classdef bayesModelSamples < dynamicprops
    %BAYESMODELSAMPLES Container for posterior samples of a Bayesian model.
    %
    %   Analogue of the empty `bayesModelSamples` class in the Python
    %   implementation: a bag of named posterior sample arrays to which new
    %   fields can be added at any time.
    %
    %   s = bayesModelSamples()        creates an empty container.
    %   s = bayesModelSamples(struct)  copies each field of a struct into it.
    %
    %   Properties are created on first assignment, so
    %
    %       s = bayesModelSamples();
    %       s.residSD = rand(1000,1);
    %
    %   works without declaring `residSD` up front. This is a handle class, so
    %   samples attached to a model in `mvBayes.fit` stay attached even when the
    %   model object itself is copied.

    methods
        function obj = bayesModelSamples(s)
            arguments
                s = []
            end

            if isempty(s)
                return
            end

            if isstruct(s)
                names = fieldnames(s);
                for i = 1:numel(names)
                    obj.(names{i}) = s.(names{i});
                end
            elseif isobject(s)
                names = properties(s);
                for i = 1:numel(names)
                    obj.(names{i}) = s.(names{i});
                end
            else
                error('bayesModelSamples:badInput', ...
                    'Input must be a struct, an object, or empty.');
            end
        end

        function obj = subsasgn(obj, S, val)
            % Create the property on first assignment, mirroring Python's
            % ability to set arbitrary attributes on an instance.
            if ~isempty(S) && strcmp(S(1).type, '.') ...
                    && (ischar(S(1).subs) || isstring(S(1).subs)) ...
                    && ~isprop(obj, S(1).subs)
                obj.addprop(char(S(1).subs));
            end
            obj = builtin('subsasgn', obj, S, val);
        end

        function s = toStruct(obj)
            %TOSTRUCT Return the samples as a plain struct.
            names = properties(obj);
            s = struct();
            for i = 1:numel(names)
                s.(names{i}) = obj.(names{i});
            end
        end
    end
end
