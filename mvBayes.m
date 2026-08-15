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
                options.nBasis = nan
                options.propVarExplained = 0.99
                options.center = true
                options.scale = false
                options.residSDExtract = []
                options.samplesExtract = []
                options.idxSamplesArg = "idxSamples"
            end

            obj.X = X;
            obj.Y = Y;
            obj.nMV = size(Y,2);
            obj.bayesModel = bayesModel;
            obj.residSDExtract = options.residSDExtract;
            obj.idxSamplesArg = options.idxSamplesArg;
            obj.samplesExtract = options.samplesExtract;

            obj.basisInfo = basisSetup(Y, options.basisType, options.nBasis, options.propVarExplained, options.center, options.scale);

            obj = obj.fit();

        end

        function obj = fit(obj)
            fprintf('Starting mvBayes with %d components\n', obj.basisInfo.nBasis)

            bmList = cell(obj.basisInfo.nBasis,1);
            for k = 1:obj.basisInfo.nBasis
                bmList{k} = obj.bayesModel(obj.X, obj.basisInfo.coefs(:,k));
            end
            obj.bmList = bmList;

            % Get Samples
            for k = 1:obj.basisInfo.nBasis
                if isempty(obj.samplesExtract)
                    if isobject(bmList{k}) && ~isprop(bmList{k}, 'samples')
                        if k == 1
                            fprintf("Generating 'samples' attribute, since it was absent in 'bmList{1}'")
                        end
                        bmList{k}.samples = bayesModelSamples();
                        continue
                    end

                    if isstruct(bmList{k}) && ~isfield(bmList{k}, 'samples')
                        if k == 1
                            fprintf("Generating 'samples' attribute, since it was absent in 'bmList{1}'")
                        end
                        bmList{k}.samples = bayesModelSamples();
                        continue
                    end
                else
                    bmList{k}.samples = obj.samplesExtract(bmList{k});
                end

            end

            % Get Residual SD
            if isempty(obj.residSDExtract)
                fprintf("Approximating 'residSD', since 'residSDExtract' is NaN\n")
                out = obj.predict(obj.X, 'returnPostCoefs', true);
                for k = 1:obj.basisInfo.nBasis
                    resid = obj.basisInfo.coefs(:,k)' - out.postCoefs(:, :, k);
                    obj.bmList{k}.samples.residSD = std(resid,0, 2);
                end
            else
                for k = 1:obj.basisInfo.nBasis
                    obj.bmList{k}.samples.residSD = obj.residSDExtract(bmList{k});
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
                options.idxSamplesArg = []
            end
            idxSamples = options.idxSamples;
            returnPostCoefs = options.returnPostCoefs;
            returnMeanOnly = options.returnMeanOnly;
            idxSamplesArg = options.idxSamplesArg;
            addResidError = options.addResidError;

            if isempty(idxSamplesArg)
                idxSamplesArg = obj.idxSamplesArg;
            end

            if (ischar(idxSamples) || isstring(idxSamples)) && strcmp(idxSamples, "default")
                % nothing to do

            elseif ~ismember(idxSamplesArg, methodInputNames(obj.bmList{1}, 'predict'))
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

            if strcmpi(idxSamples, 'default')
                args = {};
            else
                args = {idxSamplesArg, idxSamples};
            end
            postCoefs1 = obj.bmList{1}.predict(Xtest, args{:});
            postCoefs = zeros(size(postCoefs1,1), size(postCoefs1,2), obj.basisInfo.nBasis);
            postCoefs(:, :, 1) = postCoefs1;
            clear postCoefs1
            for k = 2:obj.basisInfo.nBasis
                postCoefs(:, :, k) = obj.bmList{k}.predict(Xtest, args{:});
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

            if addResidError
                for k = 1:obj.basisInfo.nBasis
                    mu = zeros(length(obj.bmList{k}.samples.residSD), 1);
                    n = size(postCoefs,2);
                    residError = mvnrnd(mu, obj.bmList{k}.samples.residSD, n);
                    postCoefs(:, :, k) = postCoefs(:, :, k) + residError;
                end
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
                    if isvector(val)
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

        % ------------------------------------------------------------------
        function tf = isModelParam(val)
            % Mirrors the Python isModelParam helper: returns true for scalars and
            % vectors (excluding strings/chars, empties, and function handles).
            if isempty(val) || isa(val, 'function_handle') || ischar(val) || isstring(val)
                tf = false;
                return
            end
            try
                tf = isnumeric(val) && (isscalar(val) || isvector(val));
            catch
                tf = false;
            end
        end

        function obj = mvSobol(obj, totalSobol, nMC)
            arguments
                obj
                totalSobol = true
                nMC = nan
            end

            p = size(obj.X,2);

            if strcmpi(obj.basisInfo.basisType, "pns") && isnan(nMC)
                nMC = 2^12;
            end

            if strcmpi(class(obj.bmList{1}), "BassModel") && isnan(nMC)
                mod = BassBasis(obj.X, obj.Y, obj.basisInfo.basis',nan,nan,nan,nan,nan,nan,false);
                mod.bm_list = obj.bmList;

                obj_sob = sobolBasis(mod);
                obj_sob = obj_sob.decomp(1);

                obj.firstOrderSobol = zeros(p, obj.basisInfo.nMV);
                if totalSobol
                    obj.totalOrderSobol = zeros(p, obj.basisInfo.nMV);
                else
                    obj.totalOrderSobol = nan;
                end
                obj.varTotal = zeros(p, obj.basisInfo.nMV);
                obj.firstOrderSobol = obj_sob.S_var(1:p,:);
                if totalSobol
                    obj.totalOrderSobol = obj_sob.T_var;
                end
                obj.varTotal = obj_sob.S_var(1,:) ./ obj_sob.S(1,:);

                obj.varTotal = max([obj.varTotal; sum(obj.firstOrderSobol,1)]);
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
                saltelliMC = obj.predict(saltelliSequence, length(obj.bmList{1}.samples.s2));
                saltelliMC = squeeze(saltelliMC);

                % transform the samples
                meanS = mean(saltelliMC);
                saltelliMC = saltelliMC - meanS;

                % Estimate Sobol' Indices
                modA = saltelliMC(1:nMC, :);
                modB = saltelliMC((nMC+1):(2*nMC), :);
                modAB = zeros(p, size(modA,1), size(modA,2));
                for j = 1:p
                    modAB(j,:,:) = saltelliMC(((2+(j-1))*nMC+1):((2+j)*nMC), :);
                end

                obj.varTotal = var(saltelliMC, 0, 1);
                clear saltelliMC

                obj.firstOrderSobol = zeros(p, obj.basisInfo.nMV);
                if totalSobol
                    obj.totalOrderSobol = zeros(p, obj.basisInfo.nMV);
                else
                    obj.totalOrderSobol = nan;
                end
                for j = 1:p
                    obj.firstOrderSobol(j, :) = mean(modB .* (squeeze(modAB(j,:,:))-modA));

                    if totalSobol
                        obj.totalOrderSobol(j, :) = 0.5 * mean((modA-squeeze(modAB(j,:,:))).^2);
                    end
                end

                obj.varTotal = max([obj.varTotal; sum(obj.firstOrderSobol,1)]);

            end

        end

        function plotSobol(obj, labels)

            arguments
                obj
                labels = nan
            end

            if ~isnan(obj.totalOrderSobol)
                totalSobol = true;
            end

            p = size(obj.X,2);
            idxMV = linspace(0, 1, obj.nMV);

            if isscalar(labels) && isnan(labels)
                labels = cell(1,p+1);
                for i=1:p
                    labels{i} = sprintf('X%d', i);
                end
            end
            labels{p+1} = "Higher Order";

            lty = repmat(["-", "--", ":", "-."], 1, mod(4, p));
            lty = lty(1:p);
            lty = [lty, "-"];

            rgb = zeros(p+1,3);
            rgb(1:p, :) = brewermap(p, 'Paired');
            rgb(p+1,:) = [153, 153, 153] / 255;

            firstOrderRel = obj.firstOrderSobol ./ obj.varTotal;

            figure()
            if totalSobol
                subplot(1,3,1)
                hold on
                [~, ord] = sort(idxMV);
                meanX = [firstOrderRel; 1.0-sum(firstOrderRel)];

                sens = cumsum(meanX);

                for j=1:(p+1)
                    x2 = [idxMV(ord) flip(idxMV(ord))];
                    if j==1
                        inBetween = [zeros(1,length(idxMV(ord))), flip(sens(j, ord))];
                    else
                        inBetween = [sens(j-1, ord), flip(sens(j,ord))];
                    end
                    fill(x2, inBetween, rgb(j,:), 'DisplayName', labels{j})
                end
                xlabel("Time")
                ylabel("Relative First-Order Sobol' Index")
                title("First-Order Relative Sensitivity")
                ylim([0,1])
                xlim([min(idxMV), max(idxMV)])

                subplot(1,3,2)
                hold on
                sens_var = [cumsum(obj.firstOrderSobol); obj.varTotal];

                for j = 1:(p+1)
                    x2 = [idxMV(ord), flip(idxMV(ord))];
                    if j == 1
                        inBetween = [zeros(1, length(idxMV(ord))), flip(sens_var(j,ord))];
                    else
                        inBetween = [sens_var(j-1, ord), flip(sens_var(j,ord))];
                    end

                    fill(x2, inBetween, rgb(j,:), 'DisplayName', labels{j})
                end
                ylim([0, max(inBetween)+3])
                xlabel("Time")
                ylabel("First-Order Sobol' Index")
                title("First-Order Sensitivity")
                xlim([min(idxMV), max(idxMV)])
                legend;

                subplot(1,3,3)
                hold on
                for j=1:p
                    plot(idxMV, obj.totalOrderSobol(j,:), 'LineStyle', lty(j), 'Color', rgb(j,:), 'LineWidth', 3, 'DisplayName', labels{j});
                end
                xlabel("Time")
                ylabel("Total-Order Sobol' Index")
                title("Total Sensitivity")
                ylim([0, max(obj.totalOrderSobol(:))*1.05])
                xlim([min(idxMV), max(idxMV)])

            else
                subplot(1,2,1)
                hold on
                [~, ord] = sort(idxMV);
                meanX = [firstOrderRel; 1.0-sum(firstOrderRel)];

                sens = cumsum(meanX);

                for j=1:(p+1)
                    x2 = [idxMV(ord) flip(idxMV(ord))];
                    if j==1
                        inBetween = [zeros(1,length(idxMV(ord))), flip(sens(j, ord))];
                    else
                        inBetween = [sens(j-1, ord), flip(sens(j,ord))];
                    end
                    fill(x2, inBetween, rgb(j,:), 'DisplayName', labels{j})
                end
                xlabel("Time")
                ylabel("Relative First-Order Sobol' Index")
                title("First-Order Relative Sensitivity")
                ylim([0,1])
                xlim([min(idxMV), max(idxMV)])

                subplot(1,2,2)
                hold on
                sens_var = [cumsum(obj.firstOrderSobol); obj.varTotal];

                for j = 1:(p+1)
                    x2 = [idxMV(ord), flip(idxMV(ord))];
                    if j == 1
                        inBetween = [zeros(1, length(idxMV(ord))), flip(sens_var(j,ord))];
                    else
                        inBetween = [sens_var(j-1, ord), flip(sens_var(j,ord))];
                    end

                    fill(x2, inBetween, rgb(j,:), 'DisplayName', labels{j})
                end
                ylim([0, max(inBetween)+3])
                xlabel("Time")
                ylabel("First-Order Sobol' Index")
                title("First-Order Sensitivity")
                xlim([min(idxMV), max(idxMV)])
                legend;
            end

        end
    end
end

function names = methodInputNames(objIn, methodName)
mc = metaclass(objIn);
m  = mc.MethodList(strcmp({mc.MethodList.Name}, methodName));
if isempty(m)
    names = {};
else
    inputs = m.Signature.Inputs;
    names = cell(1,length(inputs));
    for i = 1:length(inputs)
        tmp = inputs(i).Identifier.Name;
        names{i} = tmp;
    end
end
end
