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
    end

    methods
        function obj = mvBayes(bayesModel, X, Y, basisType, nBasis, propVarExplained, center, scale)
            arguments
                bayesModel
                X
                Y
                basisType = "pca"
                nBasis = nan
                propVarExplained = 0.99
                center = true
                scale = false
            end

            obj.X = X;
            obj.Y = Y;
            obj.nMV = size(Y,2);
            obj.bayesModel = bayesModel;

            obj.basisInfo = basisSetup(Y, basisType, nBasis, propVarExplained, center, scale);

            obj = obj.fit();

        end

        function obj = fit(obj)
            fprintf('Starting mvBayes with %d components\n', obj.basisInfo.nBasis)

            bmList = cell(obj.basisInfo.nBasis,1);
            for k = 1:obj.basisInfo.nBasis
                bmList{k} = obj.bayesModel(obj.X, obj.basisInfo.coefs(:,k));
            end
            obj.bmList = bmList;
        end

        function out = predict(obj, Xtest, mcmc_use, returnPostCoefs, returnMeanOnly)
            arguments
                obj
                Xtest
                mcmc_use = nan
                returnPostCoefs = false
                returnMeanOnly = false
            end

            postCoefs1 = obj.bmList{1}.predict(Xtest, mcmc_use);
            postCoefs = zeros(size(postCoefs1,1), size(postCoefs1,2), obj.basisInfo.nBasis);
            postCoefs(:, :, 1) = postCoefs1;
            clear postCoefs1
            for k = 2:obj.basisInfo.nBasis
                postCoefs(:, :, k) = obj.bmList{k}.predict(Xtest, mcmc_use);
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

        function plot(obj)

            idxMV = 1:obj.basisInfo.nMV;
            Xtest = obj.X;
            Ytest = obj.basisInfo.Y;
            coefs = obj.basisInfo.coefs;
            truncError = obj.basisInfo.truncError;

            if strcmpi(obj.basisInfo.basisType,"pns")
                Ycentered = Ytest - mean(Ytest,1);
            else
                Ycentered = Ytest - obj.basisInfo.Ycenter;
            end

            out_pred = obj.predict(Xtest, length(obj.bmList{1}.samples.s2), true);

            R = Ytest - squeeze(out_pred.Ypost);
            if size(coefs,2) == 1
                RbasisCoefs = coefs(:) - out_pred.postCoefs(:);
            else
                RbasisCoefs = coefs - squeeze(out_pred.postCoefs);
            end

            figure()
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
                saltelliMC = obj.predict(saltelliSequence, obj.bmList{1}.nstore);
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

            if isnan(labels)
                labels = strings(1,p+1);
                for i=1:p
                    labels(i) = sprintf('X%d', i);
                end
                labels(p+1) = "Higher Order";
            end

            lty = resize(["-", "--", ":", "-."], p);
            lty = [lty, "-"];

            rgb = zeros(p+1,3);
            rgb(1:p, :) = brewermap(p, 'Paired');
            rgb(p+1,:) = [153, 153, 153] / 255;

            firstOrderRel = obj.firstOrderSobol ./ obj.varTotal;

            figure(105)
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
                    fill(x2, inBetween, rgb(j,:), 'DisplayName', labels(j))
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

                    fill(x2, inBetween, rgb(j,:), 'DisplayName', labels(j))
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
                    plot(idxMV, obj.totalOrderSobol(j,:), 'LineStyle', lty(j), 'Color', rgb(j,:), 'LineWidth', 3, 'DisplayName', labels(j));
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
                    fill(x2, inBetween, rgb(j,:), 'DisplayName', labels(j))
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

                    fill(x2, inBetween, rgb(j,:), 'DisplayName', labels(j))
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
