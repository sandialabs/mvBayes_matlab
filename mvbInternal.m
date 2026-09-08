classdef mvbInternal
    %MVBINTERNAL Shared helpers for mvBayes and mvBayesMF.
    %
    %   Static utilities only; this class is not meant to be instantiated.

    methods (Static)

        function names = methodInputNames(objIn, methodName)
            %METHODINPUTNAMES Names of a method's declared inputs.
            mc = metaclass(objIn);
            m  = mc.MethodList(strcmp({mc.MethodList.Name}, methodName));
            if isempty(m)
                names = {};
            else
                inputs = m.Signature.Inputs;
                names = cell(1,length(inputs));
                for i = 1:length(inputs)
                    names{i} = inputs(i).Identifier.Name;
                end
            end
        end

        function kinds = methodInputKinds(objIn, methodName)
            %METHODINPUTKINDS Argument kinds ("required"/"optional"/"namevalue")
            %   when the MATLAB release exposes them; {} otherwise.
            kinds = {};
            try
                mc = metaclass(objIn);
                m  = mc.MethodList(strcmp({mc.MethodList.Name}, methodName));
                if isempty(m)
                    return
                end
                inputs = m.Signature.Inputs;
                kinds = cell(1, numel(inputs));
                for i = 1:numel(inputs)
                    kinds{i} = char(string(inputs(i).Kind));
                end
            catch
                kinds = {};
            end
        end

        function args = idxSamplesArgs(bm, argName, idxSamples)
            %IDXSAMPLESARGS Trailing arguments of bm.predict(Xtest, ...) that
            %   pass idxSamples, honoring whether the target argument is
            %   positional or name-value. (BASS's predict, for instance, takes
            %   mcmc_use positionally.)
            argName = char(argName);
            names = mvbInternal.methodInputNames(bm, 'predict');
            kinds = mvbInternal.methodInputKinds(bm, 'predict');

            pos = find(strcmp(names, argName), 1);
            if isempty(pos)
                args = {};
                return
            end

            isNameValue = true;   % fall back to name-value if the kind is unknown
            if ~isempty(kinds) && pos <= numel(kinds) && ~isempty(kinds{pos})
                isNameValue = strcmpi(kinds{pos}, 'namevalue');
            end

            if isNameValue
                args = {argName, idxSamples};
            else
                % Positional: names{1} is the object and names{2} is the test
                % input, which the caller has already supplied.
                if pos ~= 3
                    error('mvBayes:idxSamplesPosition', ...
                        ['''%s'' is a positional argument of the bayesModel predict ' ...
                         'method, but not the first one after the test inputs, so it ' ...
                         'cannot be passed automatically. Wrap the model''s predict ' ...
                         'method, or expose the argument as a name-value pair.'], argName);
                end
                args = {idxSamples};
            end
        end

        function tf = hasSamples(bm)
            %HASSAMPLES True if bm carries a non-empty 'samples' field/property.
            if isstruct(bm)
                tf = isfield(bm, 'samples');
            else
                tf = isprop(bm, 'samples');
            end
            tf = tf && ~isempty(bm.samples);
        end

        function s = getSamples(bm)
            %GETSAMPLES The samples container of bm, or [] if it has none.
            if isstruct(bm) && isfield(bm, 'samples')
                s = bm.samples;
            elseif ~isstruct(bm) && isprop(bm, 'samples')
                s = bm.samples;
            else
                s = [];
            end
        end

        function bm = setSamples(bm, samples)
            %SETSAMPLES Attach samples to bm, struct or object.
            try
                bm.samples = samples;
            catch ME
                error('mvBayes:noSamplesProperty', ...
                    ['Could not attach posterior samples to the object returned by ' ...
                     'bayesModel (%s). Return a struct, or a class with a ''samples'' ' ...
                     'property. Original error: %s'], class(bm), ME.message);
            end
        end

        function tf = hasSamplesField(s, name)
            %HASSAMPLESFIELD True if the samples container s has the named field.
            if isempty(s)
                tf = false;
            elseif isstruct(s)
                tf = isfield(s, name);
            else
                tf = isprop(s, name);
            end
        end

        function tf = isModelParam(val)
            %ISMODELPARAM True for scalar and vector samples (the "plottable"
            %   ones), matching the Python helper of the same name.
            if isempty(val) || isa(val, 'function_handle') || ischar(val) ...
                    || isstring(val) || iscell(val) || isstruct(val)
                tf = false;
                return
            end
            tf = (isnumeric(val) || islogical(val)) && (isscalar(val) || isvector(val));
        end

        function idxUse = resolveIdxSamples(idxSamples, nSamples)
            %RESOLVEIDXSAMPLES Turn "final"/"default"/numeric into MCMC indices.
            if (ischar(idxSamples) || isstring(idxSamples)) && strcmpi(idxSamples, "final")
                idxUse = nSamples;
            elseif (ischar(idxSamples) || isstring(idxSamples)) && strcmpi(idxSamples, "default")
                idxUse = 1:nSamples;
            else
                idxUse = double(idxSamples(:)).';
            end
        end

    end
end
