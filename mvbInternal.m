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

        function B = bsplineDesign(x, knots, degree)
            %BSPLINEDESIGN Cox-de Boor recursion for a B-spline design matrix.
            %
            %   Equivalent to spcol(knots, degree+1, x) from the Curve Fitting
            %   Toolbox, used by basisBspline when that toolbox is unavailable
            %   and kept alongside it so the two can be compared in tests.
            %
            %   Returns numel(x) x (numel(knots)-degree-1), one basis function
            %   per column.
            x = double(x(:));
            knots = sort(double(knots(:))).';
            nKnot = numel(knots);
            nBases = nKnot - degree - 1;

            % Degree 0: indicator of each half-open knot span.
            N = zeros(numel(x), nKnot - 1);
            for i = 1:(nKnot - 1)
                N(:, i) = (knots(i) <= x) & (x < knots(i+1));
            end

            % The half-open convention leaves the right endpoint with no
            % support, so assign it to the last non-degenerate span (as spcol
            % and scipy's splev do).
            atMax = (x == knots(end));
            if any(atMax)
                last = find(knots(1:end-1) < knots(2:end), 1, 'last');
                N(atMax, :) = 0;
                N(atMax, last) = 1;
            end

            % Raise the degree one level at a time. Terms with a zero
            % denominator come from repeated knots and contribute nothing.
            for d = 1:degree
                Nnew = zeros(numel(x), nKnot - 1 - d);
                for i = 1:(nKnot - 1 - d)
                    den1 = knots(i+d) - knots(i);
                    den2 = knots(i+d+1) - knots(i+1);
                    term = zeros(numel(x), 1);
                    if den1 > 0
                        term = term + ((x - knots(i)) / den1) .* N(:, i);
                    end
                    if den2 > 0
                        term = term + ((knots(i+d+1) - x) / den2) .* N(:, i+1);
                    end
                    Nnew(:, i) = term;
                end
                N = Nnew;
            end

            B = N(:, 1:nBases);
        end

        function knots = bsplineKnots(fDomain, nBasis, degree)
            %BSPLINEKNOTS Clamped knot vector used by basisBspline, exposed so
            %   that tests can build a collocation matrix independently.
            order = degree + 1;
            nInner = nBasis - order + 1;
            if nInner > 0
                q = linspace(0, 1, nInner + 2);
                innerKnots = mvbInternal.quantileLinear(fDomain, q(2:end-1));
            else
                innerKnots = [];
            end
            knots = sort([innerKnots(:).', ...
                repmat(min(fDomain), 1, order), repmat(max(fDomain), 1, order)]);
        end

        function tf = curveFittingAvailable()
            %CURVEFITTINGAVAILABLE True if the Curve Fitting Toolbox's B-spline
            %   collocation routine SPCOL is on the path.
            tf = exist('spcol', 'file') > 0;
        end

        function tf = parallelAvailable()
            %PARALLELAVAILABLE True if the Parallel Computing Toolbox is usable.
            %   (Without it MATLAB still runs parfor serially, so this only
            %   controls the reported nCores.)
            try
                tf = ~isempty(ver('parallel')) ...
                    && license('test', 'Distrib_Computing_Toolbox');
            catch
                tf = false;
            end
        end

        function n = numCoresAvailable()
            %NUMCORESAVAILABLE Physical cores on this machine.
            try
                n = feature('numcores');          % undocumented but exact
            catch
                n = maxNumCompThreads;            % documented fallback
            end
        end

        function ensurePool(nCores)
            %ENSUREPOOL Start a parallel pool with nCores workers if none is
            %   open. An existing pool is reused as-is rather than restarted,
            %   so a user's own pool configuration is never torn down.
            try
                pool = gcp('nocreate');
                if isempty(pool)
                    parpool(nCores);
                elseif pool.NumWorkers < nCores
                    fprintf(['Using the existing parallel pool with %d workers ' ...
                        '(%d requested).\n'], pool.NumWorkers, nCores);
                end
            catch ME
                fprintf(['Could not start a parallel pool (%s). Continuing ' ...
                    'without one.\n'], ME.message);
            end
        end

        function q = quantileLinear(x, probs)
            %QUANTILELINEAR Quantiles by linear interpolation, matching the
            %   default of numpy.percentile (MATLAB's built-in quantile uses a
            %   different plotting position).
            x = sort(double(x(:)));
            n = numel(x);
            probs = double(probs);
            q = zeros(size(probs));
            for i = 1:numel(probs)
                if n == 1
                    q(i) = x;
                    continue
                end
                h  = (n - 1) * probs(i) + 1;
                lo = floor(h);
                hi = ceil(h);
                q(i) = x(lo) + (h - lo) * (x(hi) - x(lo));
            end
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
