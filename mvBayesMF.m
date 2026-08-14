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

            obj = obj.fit();

        end

        function obj = fit(obj)
            fprintf('Starting mvBayes with %d components\n', obj.basisInfo.nBasis)

            bmList = cell(obj.basisInfo.nBasis,1);
            bmList_br = cell(obj.basisInfo.nBasis,1);
            for k = 1:obj.basisInfo.nBasis
                bmList{k}  = obj.bayesModel(obj.XL, obj.basisInfo.coefs_br(:,k));
                lf_at_H   = mean(squeeze(predict(bmList{k}, obj.XH)),1);
                residual  = obj.basisInfo.coefs(:,k) - lf_at_H';
                bmList_br{k}  = obj.bayesModel(obj.XH, residual);
            end
            obj.bmList = bmList;
            obj.bmList_br = bmList_br;

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

                    if isobject(bmList_br{k}) && ~isprop(bmList_br{k}, 'samples')
                        if k == 1
                            fprintf("Generating 'samples' attribute, since it was absent in 'bmList_br{1}'")
                        end
                        bmList_br{k}.samples = bayesModelSamples();
                        continue
                    end

                    if isstruct(bmList_br{k}) && ~isfield(bmList_br{k}, 'samples')
                        if k == 1
                            fprintf("Generating 'samples' attribute, since it was absent in 'bmList_br{1}'")
                        end
                        bmList_br{k}.samples = bayesModelSamples();
                        continue
                    end
                else
                    bmList{k}.samples = obj.samplesExtract(bmList{k});
                    bmList_br{k}.samples = obj.samplesExtract(bmList_br{k});
                end

            end

            % Get Residual SD
            if isempty(obj.residSDExtract)
                fprintf("Approximating 'residSD', since 'residSDExtract' is NaN\n")
                out = obj.predict(obj.X, 'returnPostCoefs', true);
                for k = 1:obj.basisInfo.nBasis
                    resid = obj.basisInfo.coefs(:,k)' - out.postCoefs(:, :, k);
                    bmList{k}.samples.residSD = std(resid,0, 2);
                end
            else
                for k = 1:obj.basisInfo.nBasis
                    bmList{k}.samples.residSD = obj.residSDExtract(bmList{k});
                end
            end

            obj.nSamples = length(bmList{1}.samples.residSD);
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
            postCoefs2 = obj.bmList_br{1}.predict(Xtest, args{:});
            postCoefs = zeros(size(postCoefs1,1), size(postCoefs1,2), obj.basisInfo.nBasis);
            postCoefs(:, :, 1) = postCoefs1 + postCoefs2;
            clear postCoefs1 postCoefs2
            for k = 2:obj.basisInfo.nBasis
                lf = obj.bmList{k}.predict(Xtest, args{:});
                br = obj.bmList_br{k}.predict(Xtest, args{:});
                postCoefs(:, :, k) = lf + br;
            end

            YstandardPost = pagemtimes(permute(postCoefs, [2 3 1]), obj.basisInfo.basis);
            YstandardPost = permute(YstandardPost, [3 1 2]);

            center = repmat(obj.basisInfo.Zcenter', 1, size(YstandardPost,2), size(YstandardPost,1));
            center = permute(center, [3 2 1]);
            Ypost = YstandardPost + center;
            clear YstandardPost

            if addResidError
                for k = 1:obj.basisInfo.nBasis
                    mu = zeros(length(obj.bmList{k}.samples.residSD), 1);
                    n = size(postCoefs,2);
                    residError = mvnrnd(mu, obj.bmList{k}.samples.residSD, n);
                    residError_br = mvnrnd(mu, obj.bmList_br{k}.samples.residSD, n);
                    postCoefs(:, :, k) = postCoefs(:, :, k) + residError + residError_br;
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
            xline((length(obj.basisInfo.propVarExplained) + length(obj.basisInfo.propVarExplained_enhanced))/2, '--','Color',[0.5, 0.5, 0.5])
            title(sprintf('Overall R^2 = %0.3g', r2Overall))

        end

        function obj = mvSobol(obj, totalSobol, nMC)
            arguments
                obj
                totalSobol = true
                nMC = nan
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

        function plotSobol(obj, labels)

            arguments
                obj
                labels = nan
            end

            if ~isnan(obj.totalOrderSobol)
                totalSobol = true;
            end

            p = size(obj.XH,2);
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
