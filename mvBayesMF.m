classdef mvBayesMF

    properties
        XH
        XL
        Y
        Z
        nMV
        bayesModel
        basisInfo
        bmList
        bmList_br
        firstOrderSobol
        totalOrderSobol
        varTotal
        residSDExtract
        idxSamplesArg
        samplesExtract
        nSamples
        mL
    end

    methods
        function obj = mvBayesMF(bayesModel, XH, XL, Y, Z, options)
            arguments
                bayesModel
                XH
                XL
                Y
                Z
                options.basisType = "pca"
                options.mL = 20
                options.nCores = 1
                options.residSDExtract = []
                options.samplesExtract = []
                options.idxSamplesArg = "idxSamples"
            end

            obj.XH = XH;
            obj.XL = XL;
            obj.Y = Y;
            obj.Z = Z;
            obj.nMV = size(Y,2);
            obj.bayesModel = bayesModel;
            obj.residSDExtract = options.residSDExtract;
            obj.idxSamplesArg = options.idxSamplesArg;
            obj.samplesExtract = options.samplesExtract;
            obj.mL = options.mL;

            obj.basisInfo = basisSetupMF(Y, Z, options.basisType, options.mL);

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
            %FIT Fit the low-fidelity and bias-correction models for each
            %   basis component.
            %
            %   nCores : number of workers to use (default 1). Requires the
            %            Parallel Computing Toolbox; without it, fitting falls
            %            back to serial.
            arguments
                obj
                nCores (1,1) {mustBeNumeric, mustBePositive} = 1
            end

            nCores = obj.nCoresAdjust(nCores);

            fprintf('Starting mvBayes with %d components, using %d cores.\n', ...
                obj.basisInfo.nBasis, nCores)

            nBasis = obj.basisInfo.nBasis;
            bayesModel = obj.bayesModel;
            XHfit = obj.XH;
            XLfit = obj.XL;
            coefs = obj.basisInfo.coefs;
            coefs_br = obj.basisInfo.coefs_br;

            bmList = cell(nBasis,1);
            bmList_br = cell(nBasis,1);
            if nCores == 1
                for k = 1:nBasis
                    [bmList{k}, bmList_br{k}] = fitComponentMF( ...
                        bayesModel, XHfit, XLfit, coefs(:,k), coefs_br(:,k), k);
                end
            else
                mvbInternal.ensurePool(nCores);
                parfor k = 1:nBasis
                    [bmList{k}, bmList_br{k}] = fitComponentMF( ...
                        bayesModel, XHfit, XLfit, coefs(:,k), coefs_br(:,k), k);
                end
            end
            % Get Samples. This must happen before the lists are stored on obj:
            % they hold value objects in the general case, so writing to the
            % local copies afterwards would discard the samples. Each of the two
            % model lists is handled independently.
            for k = 1:obj.basisInfo.nBasis
                if isempty(obj.samplesExtract)
                    if ~mvbInternal.hasSamples(bmList{k})
                        if k == 1
                            fprintf("Generating 'samples' attribute, since it was absent in 'bmList{1}'\n")
                        end
                        bmList{k} = mvbInternal.setSamples(bmList{k}, bayesModelSamples());
                    elseif isstruct(mvbInternal.getSamples(bmList{k}))
                        bmList{k} = mvbInternal.setSamples(bmList{k}, bayesModelSamples(mvbInternal.getSamples(bmList{k})));
                    end

                    if ~mvbInternal.hasSamples(bmList_br{k})
                        if k == 1
                            fprintf("Generating 'samples' attribute, since it was absent in 'bmList_br{1}'\n")
                        end
                        bmList_br{k} = mvbInternal.setSamples(bmList_br{k}, bayesModelSamples());
                    elseif isstruct(mvbInternal.getSamples(bmList_br{k}))
                        bmList_br{k} = mvbInternal.setSamples(bmList_br{k}, bayesModelSamples(mvbInternal.getSamples(bmList_br{k})));
                    end
                else
                    bmList{k} = mvbInternal.setSamples(bmList{k}, obj.samplesExtract(bmList{k}));
                    bmList_br{k} = mvbInternal.setSamples(bmList_br{k}, obj.samplesExtract(bmList_br{k}));
                end
            end

            obj.bmList = bmList;
            obj.bmList_br = bmList_br;

            % Get Residual SD
            if isempty(obj.residSDExtract)
                if ~mvbInternal.hasSamplesField(mvbInternal.getSamples(obj.bmList{1}), 'residSD')
                    fprintf("Approximating 'residSD', since 'residSDExtract' is empty\n")
                    out = obj.predict(obj.XH, 'returnPostCoefs', true, 'nCores', nCores);
                    for k = 1:obj.basisInfo.nBasis
                        resid = obj.basisInfo.coefs(:,k)' - out.postCoefs(:, :, k);
                        % Normalize by N (not N-1) to match numpy's np.std default.
                        obj.bmList{k}.samples.residSD = std(resid, 1, 2);
                        obj.bmList_br{k}.samples.residSD = std(resid, 1, 2);
                    end
                end
            else
                for k = 1:obj.basisInfo.nBasis
                    obj.bmList{k}.samples.residSD = obj.residSDExtract(obj.bmList{k});
                    obj.bmList_br{k}.samples.residSD = obj.residSDExtract(obj.bmList_br{k});
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
                    idxSamples = obj.nSamples;          % see note 3
                elseif isnumeric(idxSamples) || islogical(idxSamples)
                    idxSamples = double(idxSamples(:)).';   % scalar or vector, both fine
                elseif iscell(idxSamples)
                    idxSamples = cell2mat(cellfun(@double, idxSamples(:).', 'UniformOutput', false));
                else
                    try
                        idxSamples = double(idxSamples);
                    catch
                        error('MyClass:badIdxSamples', ...
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
            bmList_br = obj.bmList_br;
            coefsCell = cell(nBasis,1);
            if nCores == 1
                for k = 1:nBasis
                    coefsCell{k} = bmList{k}.predict(Xtest, args{:}) ...
                        + bmList_br{k}.predict(Xtest, args{:});
                end
            else
                mvbInternal.ensurePool(nCores);
                parfor k = 1:nBasis
                    coefsCell{k} = bmList{k}.predict(Xtest, args{:}) ...
                        + bmList_br{k}.predict(Xtest, args{:});
                end
            end
            postCoefs = cat(3, coefsCell{:});
            clear coefsCell

            % Residual error is added to the coefficients before the basis
            % expansion, so that it propagates into the returned response.
            if addResidError
                for k = 1:obj.basisInfo.nBasis
                    residSD = obj.bmList{k}.samples.residSD(:);
                    residSD_br = obj.bmList_br{k}.samples.residSD(:);
                    residError = randn(size(postCoefs,1), size(postCoefs,2)) .* residSD;
                    residError_br = randn(size(postCoefs,1), size(postCoefs,2)) .* residSD_br;
                    postCoefs(:, :, k) = postCoefs(:, :, k) + residError + residError_br;
                end
            end

            YstandardPost = pagemtimes(permute(postCoefs, [2 3 1]), obj.basisInfo.basis);
            YstandardPost = permute(YstandardPost, [3 1 2]);

            center = repmat(obj.basisInfo.Zcenter', 1, size(YstandardPost,2), size(YstandardPost,1));
            center = permute(center, [3 2 1]);
            Ypost = YstandardPost + center;
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

        function fig = plot(obj)

            idxMV = 1:obj.basisInfo.nMV;
            XtestH = obj.XH;
            XtestL = obj.XL;
            Ytest = obj.basisInfo.Y;
            Ztest = obj.basisInfo.Y;
            coefs = obj.basisInfo.coefs;
            coefs_br = obj.basisInfo.coefs_br;
            truncError = obj.basisInfo.truncError;

            Ycentered = Ytest - obj.basisInfo.Ycenter;

            args = {'idxSamples', 'final', 'returnPostCoefs', true, 'idxSamplesArg', obj.idxSamplesArg};

            out_pred = obj.predict(XtestH, args{:});

            R = Ytest - squeeze(out_pred.Ypost);
            if size(coefs,2) == 1
                RbasisCoefs = coefs(:) - out_pred.postCoefs(:);
            else
                RbasisCoefs = coefs - squeeze(out_pred.postCoefs);
            end

            fig = figure();
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
            for k = 1:obj.basisInfo.nBasis
                mseBasis(k) = mean(RbasisCoefs(:,k).^2);
            end

            subplot(1,2,2)
            r2Basis = compute_r2_field(coefs', squeeze(out_pred.postCoefs)');
            varOverall = sum(obj.basisInfo.varExplained)*(size(Ytest,1)-1)/(size(Ytest,1));
            r2Overall = compute_r2(Ytest, squeeze(out_pred.Ypost));

            scatter(1:obj.basisInfo.nBasis, [obj.basisInfo.propVarExplained; obj.basisInfo.propVarExplained_enhanced], 50, map(1:obj.basisInfo.nBasis,:), 'filled')
            xlabel("Component")
            ylabel("Var Explained")
            xline(floor((length(obj.basisInfo.propVarExplained) + length(obj.basisInfo.propVarExplained_enhanced))/2)-1, '--','Color',[0.5, 0.5, 0.5])
            title(sprintf('Overall R^2 = %0.3g', r2Overall))

        end

        function obj = mvSobol(obj, totalSobol, nMC, idxSamples)
            %MVSOBOL Sobol' indices by Monte Carlo.
            %
            %   The indices keep a leading posterior-sample dimension:
            %   firstOrderSobol and totalOrderSobol are nSamplesUsed x p x nMV.
            arguments
                obj
                totalSobol = true
                nMC = nan
                idxSamples = "final"
            end

            p = size(obj.XH,2);

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

            xmin = min(obj.XH);
            xrange = max(obj.XH) - xmin;
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
                error('mvBayesMF:noSobol', ...
                    "Sobol' indices have not been computed. Need to run mvSobol before plotSobol.");
            end

            p = size(obj.XH,2);

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
function [bm, bm_br] = fitComponentMF(bayesModel, XH, XL, coefsK, coefsBrK, k)
%FITCOMPONENTMF Fit the low-fidelity model and its bias correction for one
%   basis component, naming the component if either fit fails.
try
    bm = bayesModel(XL, coefsBrK);
    lf_at_H = mean(squeeze(predict(bm, XH)), 1);
    residual = coefsK - lf_at_H.';
    bm_br = bayesModel(XH, residual);
catch ME
    error('mvBayesMF:bayesModelFailed', ...
        'Error fitting model %d: %s', k, ME.message);
end
end
