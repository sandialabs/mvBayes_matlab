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
    end

    methods
        function obj = mvBayesMF(bayesModel, XH, XL, Y, Z, basisType, mL)
            arguments
                bayesModel
                XH
                XL
                Y
                Z
                basisType = "pca"
                mL = 20
            end

            obj.XH = XH;
            obj.XL = XL;
            obj.Y = Y;
            obj.Z = Z;
            obj.nMV = size(Y,2);
            obj.bayesModel = bayesModel;

            obj.basisInfo = basisSetupMF(Y, Z, basisType, mL);

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
            postCoefs2 = obj.bmList_br{1}.predict(Xtest, mcmc_use);
            postCoefs = zeros(size(postCoefs1,1), size(postCoefs1,2), obj.basisInfo.nBasis);
            postCoefs(:, :, 1) = postCoefs1 + postCoefs2;
            clear postCoefs1 postCoefs2
            for k = 2:obj.basisInfo.nBasis
                lf = obj.bmList{k}.predict(Xtest, mcmc_use);
                br = obj.bmList_br{k}.predict(Xtest, mcmc_use);
                postCoefs(:, :, k) = lf + br;
            end

            YstandardPost = pagemtimes(permute(postCoefs, [2 3 1]), obj.basisInfo.basis);
            YstandardPost = permute(YstandardPost, [3 1 2]);

            center = repmat(obj.basisInfo.Zcenter', 1, size(YstandardPost,2), size(YstandardPost,1));
            center = permute(center, [3 2 1]);
            Ypost = YstandardPost + center;
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
            XtestH = obj.XH;
            XtestL = obj.XL;
            Ytest = obj.basisInfo.Y;
            Ztest = obj.basisInfo.Y;
            coefs = obj.basisInfo.coefs;
            coefs_br = obj.basisInfo.coefs_br;
            truncError = obj.basisInfo.truncError;

            Ycentered = Ytest - obj.basisInfo.Ycenter;

            out_pred = obj.predict(XtestH, length(obj.bmList{1}.samples.s2), true);

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
            for k = 1:obj.basisInfo.nBasis
                mseBasis(k) = mean(RbasisCoefs(:,k).^2);
                varBasis(k) = obj.basisInfo.varExplained(k)*(size(Ytest,1)-1)/(size(Ytest,1));
            end

            subplot(1,2,2)
            r2Basis = compute_r2_field(coefs', squeeze(out_pred.postCoefs)');
            varOverall = sum(obj.basisInfo.varExplained)*(size(Ytest,1)-1)/(size(Ytest,1));
            r2Overall = compute_r2(Ytest, squeeze(out_pred.Ypost));

            scatter(1:obj.basisInfo.nBasis, r2Basis, 50, map(1:obj.basisInfo.nBasis,:), 'filled')
            xlabel("Component")
            ylabel("MSE")
            title(sprintf('Overall R^2 = %0.3g', r2Overall))
            yline(r2Overall, '--', 'Color',[0.5, 0.5, 0.5])

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
