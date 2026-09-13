classdef mvBayes

    properties
        X
        Y
        nMV
        bayesModel
        basisInfo
        bmList
        firstOrderSobol
        totalOrderSobol
        varTotal
        residSDExtract
        idxSamplesArg
        samplesExtract
        nSamples
    end

    methods
        function obj = mvBayes(bayesModel, X, Y, options)
            arguments
                bayesModel
                X
                Y
                options.basisType = "pca"
                options.customBasis = []
                options.nBasis = nan
                options.propVarExplained = 0.99
                options.nCores = 1
                options.center = true
                options.scale = false
                options.residSDExtract = []
                options.samplesExtract = []
                options.idxSamplesArg = "idxSamples"
                options.thresh = 1e-15
            end

            obj.X = X;
            obj.Y = Y;
            obj.nMV = size(Y,2);
            obj.bayesModel = bayesModel;
            obj.residSDExtract = options.residSDExtract;
            obj.idxSamplesArg = options.idxSamplesArg;
            obj.samplesExtract = options.samplesExtract;

            obj.basisInfo = basisSetup(Y, options.basisType, options.customBasis, ...
                options.nBasis, options.propVarExplained, options.center, ...
                options.scale, options.thresh);

            obj = obj.fit(options.nCores);

        end

        function nCores = nCoresAdjust(obj, nCores)
            %NCORESADJUST Clamp nCores to the number of components, the
            %   availability of the Parallel Computing Toolbox, and the number of
            %   cores on this machine.
            nCores = min(nCores, obj.basisInfo.nBasis);
            if nCores > 1 && ~mvbInternal.parallelAvailable()
                fprintf(['Parallel Computing Toolbox not available. ' ...
                    'Setting nCores=1.\n']);
                nCores = 1;
            else
                nCoresAvailable = mvbInternal.numCoresAvailable();
                if nCores > nCoresAvailable
                    fprintf(['Only %d cores are available. Using all available ' ...
                        'cores.\n'], nCoresAvailable);
                    nCores = nCoresAvailable;
                end
            end
        end

        function obj = fit(obj, nCores)
            %FIT Fit bayesModel for each basis component.
            %
            %   nCores : number of workers to use when fitting the independent
            %            models (default 1). Requires the Parallel Computing
            %            Toolbox; without it, fitting falls back to serial.
            arguments
                obj
                nCores (1,1) {mustBeNumeric, mustBePositive} = 1
            end

            nCores = obj.nCoresAdjust(nCores);

            fprintf('Starting mvBayes with %d components, using %d cores.\n', ...
                obj.basisInfo.nBasis, nCores)

            nBasis = obj.basisInfo.nBasis;
            bayesModel = obj.bayesModel;
            Xfit = obj.X;
            coefs = obj.basisInfo.coefs;

            bmList = cell(nBasis,1);
            if nCores == 1
                for k = 1:nBasis
                    bmList{k} = fitBayesModel(bayesModel, Xfit, coefs(:,k), k);
                end
            else
                mvbInternal.ensurePool(nCores);
                parfor k = 1:nBasis
                    bmList{k} = fitBayesModel(bayesModel, Xfit, coefs(:,k), k);
                end
            end

            % Get Samples. This must happen before bmList is stored on obj:
            % bmList holds value objects in the general case, so writing to the
            % local copy after assigning obj.bmList would discard the samples.
            for k = 1:obj.basisInfo.nBasis
                if isempty(obj.samplesExtract)
                    if ~mvbInternal.hasSamples(bmList{k})
                        if k == 1
                            fprintf("Generating 'samples' attribute, since it was absent in 'bmList{1}'\n")
                        end
                        bmList{k} = mvbInternal.setSamples(bmList{k}, bayesModelSamples());
                    elseif isstruct(mvbInternal.getSamples(bmList{k}))
                        % Normalize struct samples into the samples container so
                        % that new fields can always be added.
                        bmList{k} = mvbInternal.setSamples(bmList{k}, ...
                            bayesModelSamples(mvbInternal.getSamples(bmList{k})));
                    end
                else
                    bmList{k} = mvbInternal.setSamples(bmList{k}, obj.samplesExtract(bmList{k}));
                end
            end

            obj.bmList = bmList;

            % Get Residual SD
            if isempty(obj.residSDExtract)
                if ~mvbInternal.hasSamplesField(mvbInternal.getSamples(obj.bmList{1}), 'residSD')
                    fprintf("Approximating 'residSD', since 'residSDExtract' is empty\n")
                    out = obj.predict(obj.X, 'returnPostCoefs', true, 'nCores', nCores);
                    for k = 1:obj.basisInfo.nBasis
                        resid = obj.basisInfo.coefs(:,k)' - out.postCoefs(:, :, k);
                        % Normalize by N (not N-1) to match numpy's np.std default.
                        obj.bmList{k}.samples.residSD = std(resid, 1, 2);
                    end
                end
            else
                for k = 1:obj.basisInfo.nBasis
                    obj.bmList{k}.samples.residSD = obj.residSDExtract(obj.bmList{k});
                end
            end

            obj.nSamples = length(obj.bmList{1}.samples.residSD);

        end

        function out = predict(obj, Xtest, options)
            arguments
                obj
                Xtest
                options.idxSamples = "default"
                options.returnPostCoefs = false
                options.returnMeanOnly = false
                options.addResidError = false
                options.addTruncError = false
                options.nCores = 1
                options.idxSamplesArg = []
            end
            idxSamples = options.idxSamples;
            returnPostCoefs = options.returnPostCoefs;
            returnMeanOnly = options.returnMeanOnly;
            idxSamplesArg = options.idxSamplesArg;
            addResidError = options.addResidError;
            addTruncError = options.addTruncError;

            if isempty(idxSamplesArg)
                idxSamplesArg = obj.idxSamplesArg;
            end

            if (ischar(idxSamples) || isstring(idxSamples)) && strcmp(idxSamples, "default")
                % nothing to do

            elseif ~ismember(idxSamplesArg, mvbInternal.methodInputNames(obj.bmList{1}, 'predict'))
                fprintf(['''%s'' is not an argument of the bayesModel predict ' ...
                    'function...setting idxSamples=''default''\n'], idxSamplesArg);
                idxSamples = "default";

            else
                if (ischar(idxSamples) || isstring(idxSamples)) && strcmp(idxSamples, "final")
                    idxSamples = obj.nSamples;
                elseif isnumeric(idxSamples) || islogical(idxSamples)
                    idxSamples = double(idxSamples(:)).';   % scalar or vector, both fine
                elseif iscell(idxSamples)
                    idxSamples = cell2mat(cellfun(@double, idxSamples(:).', 'UniformOutput', false));
                else
                    try
                        idxSamples = double(idxSamples);
                    catch
                        error('mvBayes:badIdxSamples', ...
                            ['''idxSamples'' must be ''default'', ''final'', ' ...
                            'numeric, or coercible to numeric.']);
                    end
                end
            end

            if (ischar(idxSamples) || isstring(idxSamples)) && strcmpi(idxSamples, 'default')
                args = {};
            else
                args = mvbInternal.idxSamplesArgs(obj.bmList{1}, idxSamplesArg, idxSamples);
            end
            % In almost all cases use nCores=1 here, to avoid competing with
            % parallelism inside bayesModel's own predict method.
            nCores = obj.nCoresAdjust(options.nCores);

            nBasis = obj.basisInfo.nBasis;
            bmList = obj.bmList;
            coefsCell = cell(nBasis,1);
            if nCores == 1
                for k = 1:nBasis
                    coefsCell{k} = bmList{k}.predict(Xtest, args{:});
                end
            else
                mvbInternal.ensurePool(nCores);
                parfor k = 1:nBasis
                    coefsCell{k} = bmList{k}.predict(Xtest, args{:});
                end
            end
            postCoefs = cat(3, coefsCell{:});
            clear coefsCell

            % Residual error is added to the coefficients before the basis
            % expansion, so that it propagates into the returned response.
            if addResidError
                for k = 1:obj.basisInfo.nBasis
                    residSD = obj.bmList{k}.samples.residSD(:);
                    residError = randn(size(postCoefs,1), size(postCoefs,2)) .* residSD;
                    postCoefs(:, :, k) = postCoefs(:, :, k) + residError;
                end
            end

            if strcmpi(obj.basisInfo.basisType, "pns")
                PNS = obj.basisInfo.basisConstruct;
                N = size(postCoefs,1) * size(postCoefs,2);
                nBasis = obj.basisInfo.nBasis;
                inmat = zeros(size(PNS.radii,1), N);
                inmat(1:nBasis, :) = reshape(postCoefs, N, nBasis)';
                tmp = fastPNSe2s(inmat, PNS) * PNS.radius;
                YstandardPost = reshape(tmp, size(postCoefs,1), size(postCoefs,2), size(tmp,2));
                clear tmp
            else
                YstandardPost = pagemtimes(permute(postCoefs, [2 3 1]), obj.basisInfo.basis);
                YstandardPost = permute(YstandardPost, [3 1 2]);
            end
            center = repmat(obj.basisInfo.Ycenter', 1, size(YstandardPost,2), size(YstandardPost,1));
            center = permute(center, [3 2 1]);
            Ypost = YstandardPost .* obj.basisInfo.Yscale + center;
            clear YstandardPost

            if addTruncError
                nDraw = size(Ypost,1) * size(Ypost,2);
                idxResample = randi(size(obj.Y,1), nDraw, 1);
                truncError = obj.basisInfo.truncError(idxResample, :);
                Ypost = Ypost + reshape(truncError, size(Ypost));
                clear truncError
            end

            if returnMeanOnly
                Ypost = squeeze(mean(Ypost, 1));
                postCoefs = squeeze(mean(postCoefs, 1));
            end

            if returnPostCoefs
                out.Ypost = Ypost;
                out.postCoefs = postCoefs;
            else
                out = Ypost;
            end
        end

        function fig=plot(obj)

            idxMV = 1:obj.basisInfo.nMV;
            Xtest = obj.X;
            Ytest = obj.basisInfo.Y;
            coefs = obj.basisInfo.coefs;

            if strcmpi(obj.basisInfo.basisType,"pns")
                Ycentered = Ytest - mean(Ytest,1);
            else
                Ycentered = Ytest - obj.basisInfo.Ycenter;
            end

            args = {'idxSamples', 'final', 'returnPostCoefs', true, 'idxSamplesArg', obj.idxSamplesArg};

            out_pred = obj.predict(Xtest, args{:});

            R = Ytest - squeeze(out_pred.Ypost);
            if size(coefs,2) == 1
                RbasisCoefs = coefs(:) - out_pred.postCoefs(:);
            else
                RbasisCoefs = coefs - squeeze(out_pred.postCoefs);
            end

            fig=figure();
            subplot(1,2,1)
            hold on
            map = tab20;

            mseOverall = mean(R(:).^2) * size(Ytest,2);
            plot(idxMV, Ycentered(1,:), color=[0.7, 0.7, 1.0, 0.5])
            plot(idxMV, R(1,:), color=[0,0,0,.5])
            plot(idxMV, Ycentered', color=[0.7, 0.7, 1.0, 0.5])
            plot(idxMV, R', color=[0,0,0,.5])
            legend('Original', 'Residual')
            xlabel('Multivariate Index')
            ylabel('Residuals')
            title(sprintf('Overal MSE = %0.4g', mseOverall/size(Ytest,2)))

            mseBasis = zeros(obj.basisInfo.nBasis,1);
            varBasis = zeros(obj.basisInfo.nBasis,1);
            if strcmpi(obj.basisInfo.basisType,"pns")
                for k = 1:obj.basisInfo.nBasis
                    mseBasis(k) = mean(RbasisCoefs(:,k).^2);
                    varBasis(k) = mean(coefs(:,k).^2);
                end
            else
                for k = 1:obj.basisInfo.nBasis
                    mseBasis(k) = mean(RbasisCoefs(:,k).^2);
                    varBasis(k) = obj.basisInfo.varExplained(k)*(size(Ytest,1)-1)/(size(Ytest,1));
                end
            end

            subplot(1,2,2)
            r2Basis = 1 - mseBasis ./ varBasis;
            varOverall = sum(obj.basisInfo.varExplained)*(size(Ytest,1)-1)/(size(Ytest,1));
            if strcmpi(obj.basisInfo.basisType,"pns")
                r2Overall = 1 - (mseOverall / size(Ytest,2)) / varOverall;
            else
                r2Overall = 1 - mseOverall / varOverall;
            end

            scatter(1:obj.basisInfo.nBasis, r2Basis, 50, map(1:obj.basisInfo.nBasis,:), 'filled')
            xlabel("Component")
            ylabel("R^2")
            title(sprintf('Overall R^2 = %0.3g', r2Overall))
            yline(r2Overall, '--', 'Color',[0.5, 0.5, 0.5])

        end

        function traceplot(obj, modelParams, labels, plotTitle, file)
            %TRACEPLOT Trace plots of model parameters
            %
            %   obj.traceplot(modelParams, labels, plotTitle, file, ...)
            %
            %   modelParams : char/string or cell array of strings specifying names of
            %                 model parameters to plot. These should be fields
            %                 (struct) or properties (object) of `samples` on each
            %                 element of obj.bmList. If [] or omitted, selects
            %                 "plottable" fields/properties of samples (scalars and
            %                 vectors), including residSD.
            %   labels      : char/string or cell array of strings labeling each
            %                 model parameter. Default is to use modelParams.
            %   plotTitle   : title for the whole figure. Default is no title.
            %   file        : file path to save the plot. Default ([]) is not to
            %                 save, but to just leave the figure open (in place of
            %                 plt.show()).
            %
            %   Returns nothing.

            arguments
                obj
                modelParams = []
                labels = []
                plotTitle = []
                file = []
            end

            bmList = obj.bmList;      % cell array of bayesModel-like objects
            nBasis = obj.basisInfo.nBasis;

            % ---- default modelParams: auto-detect "plottable" attributes ----
            if isempty(modelParams)
                samp1 = bmList{1}.samples;

                if isstruct(samp1)
                    allAttrs = fieldnames(samp1);
                else
                    allAttrs = properties(samp1);
                end

                modelParams = {};
                for i = 1:numel(allAttrs)
                    attr = allAttrs{i};
                    val = samp1.(attr);
                    if mvbInternal.isModelParam(val)
                        modelParams{end+1} = attr; %#ok<AGROW>
                    end
                end
            elseif ischar(modelParams) || isstring(modelParams)
                modelParams = {char(modelParams)};
            end

            % ---- default labels ----
            if ischar(labels) || isstring(labels)
                labels = {char(labels)};
            elseif isempty(labels)
                labels = modelParams;
            end

            nParams = numel(modelParams);
            if nParams > 8
                warning('Currently, must have length(modelParams) <= 8. Plotting the first 8.');
                modelParams = modelParams(1:8);
                labels = labels(1:8);
                nParams = 8;
            end

            nrow = ceil(nParams / 2);
            if nParams == 1
                ncol = 1;
            else
                ncol = 2;
            end

            fig = figure('Position', [100 100 800 600]);
            cmap = tab20;   % qualitative 20-color palette (analog of "tab20")

            for j = 1:nParams
                subplot(nrow, ncol, j);
                hold on
                for k = 1:nBasis
                    s = bmList{k}.samples;

                    if isstruct(s) && isfield(s, modelParams{j})
                        param = s.(modelParams{j});
                    elseif isobject(s) && isprop(s, modelParams{j})
                        param = s.(modelParams{j});
                    else
                        error('No attribute named %s', modelParams{j});
                    end

                    colorIdx = mod(k - 1, 20) + 1;
                    plot(param, 'Color', cmap(colorIdx, :));
                end
                hold off
                ylabel(labels{j});
                xlabel('MCMC iteration');
            end

            if ~isempty(plotTitle)
                sgtitle(plotTitle);
            end

            if isempty(file)
                % leave figure visible (analog of plt.show())
            else
                exportgraphics(fig, file);
            end
        end

        function obj = mvSobol(obj, totalSobol, nMC, idxSamples)
            %MVSOBOL Sobol' indices, by Monte Carlo or (for BASS) closed form.
            %
            %   The indices keep a leading posterior-sample dimension:
            %   firstOrderSobol and totalOrderSobol are nSamplesUsed x p x nMV.
            arguments
                obj
                totalSobol = true
                nMC = nan
                idxSamples = "final"
            end

            p = size(obj.X,2);

            if strcmpi(obj.basisInfo.basisType, "pns") && isnan(nMC)
                nMC = 2^12;
            end

            useBASS = strcmpi(class(obj.bmList{1}), "BassModel") && isnan(nMC);

            if useBASS
                mod = BassBasis(obj.X, obj.Y, obj.basisInfo.basis',nan,nan,nan,nan,nan,nan,false);
                mod.bm_list = obj.bmList;

                if totalSobol
                    maxOrder = min(p, obj.bmList{1}.prior.maxInt);
                else
                    maxOrder = 1;
                end

                idxUse = mvbInternal.resolveIdxSamples(idxSamples, obj.nSamples);
                nUse = numel(idxUse);

                firstOrder = zeros(nUse, p, obj.basisInfo.nMV);
                if totalSobol
                    totalOrder = zeros(nUse, p, obj.basisInfo.nMV);
                else
                    totalOrder = [];
                end
                varTot = zeros(nUse, obj.basisInfo.nMV);

                for i = 1:nUse
                    obj_sob = sobolBasis(mod);
                    obj_sob = obj_sob.decomp(maxOrder, nan, idxUse(i));

                    firstOrder(i, :, :) = obj_sob.S_var(1:p, :);
                    if totalSobol
                        totalOrder(i, :, :) = obj_sob.T_var;
                    end
                    varTot(i, :) = obj_sob.S_var(1,:) ./ obj_sob.S(1,:);
                end
            else
                if isnan(nMC)
                    nMC = 2^12;
                end

                % Generate random samples of parameters according to Saltelli
                % (2010) method.
                qrng = sobolset(2*p);
                qrng = scramble(qrng,'MatousekAffineOwen');
                baseSequence = net(qrng,nMC);
                A = baseSequence(:, 1:p);
                B = baseSequence(:, (p+1):(2*p));
                clear baseSequence
                AB = zeros(p*nMC,p);
                for j = 1:p
                    idx = 1:p;
                    idx(j) = [];
                    AB(((j-1)*nMC+1):(j*nMC), idx) = A(:,idx);
                    AB(((j-1)*nMC+1):(j*nMC), j) = B(:,j);
                end
                saltelliSequence = [A; B; AB];
                clear A B AB

                xmin = min(obj.X);
                xrange = max(obj.X) - xmin;
                saltelliSequence = saltelliSequence .* xrange;
                saltelliSequence = saltelliSequence + xmin;

                % evaluate model at those param values
                saltelliMC = obj.predict(saltelliSequence, 'idxSamples', idxSamples);
                nUse = size(saltelliMC, 1);

                % transform the samples (center each posterior sample over the
                % Monte Carlo dimension)
                saltelliMC = saltelliMC - mean(saltelliMC, 2);

                % Estimate Sobol' Indices
                modA = saltelliMC(:, 1:nMC, :);
                modB = saltelliMC(:, (nMC+1):(2*nMC), :);

                % Normalize by N (not N-1) to match numpy's np.var default.
                varTot = reshape(var(saltelliMC, 1, 2), nUse, obj.basisInfo.nMV);

                firstOrder = zeros(nUse, p, obj.basisInfo.nMV);
                if totalSobol
                    totalOrder = zeros(nUse, p, obj.basisInfo.nMV);
                else
                    totalOrder = [];
                end
                for j = 1:p
                    modAB = saltelliMC(:, ((2+(j-1))*nMC+1):((2+j)*nMC), :);

                    firstOrder(:, j, :) = mean(modB .* (modAB - modA), 2);

                    if totalSobol
                        totalOrder(:, j, :) = 0.5 * mean((modA - modAB).^2, 2);
                    end
                end
                clear saltelliMC modA modB modAB
            end

            obj.firstOrderSobol = firstOrder;
            obj.totalOrderSobol = totalOrder;

            sumFirst = sum(reshape(mean(firstOrder, 1), p, obj.basisInfo.nMV), 1);
            obj.varTotal = max([varTot; sumFirst], [], 1);

        end

        function plotSobol(obj, options)
            %PLOTSOBOL Plot the Sobol' indices computed by mvSobol.
            %
            %   Left   - first-order indices normalized to sum to one at each
            %            multivariate index
            %   Center - first-order indices on the original variance scale
            %   Right  - total-order indices (only if they were computed)
            arguments
                obj
                options.totalSobol = true
                options.labels = []
                options.idxMV = []
                options.waterfall = false
                options.xlabel = "Multivariate Index"
                options.plotTitle = []
                options.file = []
            end

            if isempty(obj.firstOrderSobol)
                error('mvBayes:noSobol', ...
                    "Sobol' indices have not been computed. Need to run mvSobol before plotSobol.");
            end

            p = size(obj.X,2);

            idxMV = options.idxMV;
            if isempty(idxMV)
                idxMV = 1:obj.nMV;
            end
            idxMV = idxMV(:).';

            labels = options.labels;
            if isempty(labels)
                labels = cell(1,p);
                for i = 1:p
                    labels{i} = sprintf('X%d', i);
                end
            elseif ~iscell(labels)
                labels = cellstr(labels);
            end
            labels = [labels(:).', {'Higher-Order'}];

            % Line styles cycle through four options, one per predictor.
            lty = repmat(["-", "--", ":", "-."], 1, ceil(p/4));
            lty = [lty(1:p), "-"];

            rgb = zeros(p+1,3);
            rgb(1:p, :) = brewermap(p, 'Paired');
            rgb(p+1,:) = [153, 153, 153] / 255;

            % Posterior means of the indices
            firstOrder = reshape(mean(obj.firstOrderSobol, 1), p, obj.nMV);
            firstOrderRel = firstOrder ./ obj.varTotal;

            hasTotal = options.totalSobol && ~isempty(obj.totalOrderSobol);
            if options.totalSobol && isempty(obj.totalOrderSobol)
                fprintf("Total-order Sobol' indices have not been computed and will not be plotted.\n");
            end
            nPanel = 2 + hasTotal;

            figure()
            [~, ord] = sort(idxMV);

            % ---- Panel 1: relative first-order indices ----
            subplot(1,nPanel,1)
            hold on
            if options.waterfall
                meanX = [firstOrderRel; 1.0 - sum(firstOrderRel,1)];
                sens = cumsum(meanX, 1);
                for j = 1:(p+1)
                    x2 = [idxMV(ord), flip(idxMV(ord))];
                    if j == 1
                        inBetween = [zeros(1,numel(idxMV)), flip(sens(j, ord))];
                    else
                        inBetween = [sens(j-1, ord), flip(sens(j, ord))];
                    end
                    fill(x2, inBetween, rgb(j,:), 'DisplayName', labels{j})
                end
            else
                for j = 1:p
                    plot(idxMV, firstOrderRel(j,:), 'LineStyle', lty(j), ...
                        'Color', rgb(j,:), 'LineWidth', 3, 'DisplayName', labels{j});
                end
                plot(idxMV, 1.0 - sum(firstOrderRel,1), 'LineStyle', lty(p+1), ...
                    'Color', rgb(p+1,:), 'LineWidth', 3, 'DisplayName', labels{p+1});
            end
            xlabel(options.xlabel)
            ylabel("Relative First-Order Sobol' Index")
            title("First-Order Relative Sensitivity")
            ylim([0,1])
            xlim([min(idxMV), max(idxMV)])

            % ---- Panel 2: first-order indices on the variance scale ----
            subplot(1,nPanel,2)
            hold on
            if options.waterfall
                sens_var = [cumsum(firstOrder,1); obj.varTotal];
                for j = 1:(p+1)
                    x2 = [idxMV(ord), flip(idxMV(ord))];
                    if j == 1
                        inBetween = [zeros(1, numel(idxMV)), flip(sens_var(j,ord))];
                    else
                        inBetween = [sens_var(j-1, ord), flip(sens_var(j,ord))];
                    end
                    fill(x2, inBetween, rgb(j,:), 'DisplayName', labels{j})
                end
                ylim([0, max(sens_var(:))*1.05])
            else
                for j = 1:p
                    plot(idxMV, firstOrder(j,:), 'LineStyle', lty(j), ...
                        'Color', rgb(j,:), 'LineWidth', 3, 'DisplayName', labels{j});
                end
                plot(idxMV, obj.varTotal - sum(firstOrder,1), 'LineStyle', lty(p+1), ...
                    'Color', rgb(p+1,:), 'LineWidth', 3, 'DisplayName', labels{p+1});
                ylim([0, max(firstOrder(:))*1.05])
            end
            xlabel(options.xlabel)
            ylabel("First-Order Sobol' Index")
            title("First-Order Sensitivity")
            xlim([min(idxMV), max(idxMV)])
            legend('Location', 'northwest');

            % ---- Panel 3: total-order indices ----
            if hasTotal
                totalOrder = reshape(mean(obj.totalOrderSobol, 1), p, obj.nMV);
                subplot(1,nPanel,3)
                hold on
                for j = 1:p
                    plot(idxMV, totalOrder(j,:), 'LineStyle', lty(j), ...
                        'Color', rgb(j,:), 'LineWidth', 3, 'DisplayName', labels{j});
                end
                xlabel(options.xlabel)
                ylabel("Total-Order Sobol' Index")
                title("Total Sensitivity")
                ylim([0, max(totalOrder(:))*1.05])
                xlim([min(idxMV), max(idxMV)])
            end

            if ~isempty(options.plotTitle)
                sgtitle(options.plotTitle);
            end

            if ~isempty(options.file)
                exportgraphics(gcf, options.file);
            end

        end
    end
end

% =========================================================================
function bm = fitBayesModel(bayesModel, X, y, k)
%FITBAYESMODEL Fit one component, naming the component if it fails.
try
    bm = bayesModel(X, y);
catch ME
    error('mvBayes:bayesModelFailed', ...
        'Error fitting model %d: %s', k, ME.message);
end
end
