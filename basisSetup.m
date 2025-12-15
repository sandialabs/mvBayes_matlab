classdef basisSetup
    % Compute basis components for a matrix Y. Used in mvBayes.

    properties
        Y
        nMV
        basisType
        varExplained
        center
        scale
        Ycenter
        Yscale
        propVarExplained
        propVarCumSum
        truncError
        nBasis
        basis
        basisConstruct
        coefs
        tt
    end

    methods
        function obj = basisSetup(Y, basisType, nBasis, propVarExplained, center, scale)
            arguments
                Y {mustBeNumeric}
                basisType = "pca"
                nBasis = nan
                propVarExplained = 0.99
                center = true
                scale = false
            end

            obj.Y = Y;
            obj.nMV = size(Y,2);
            obj.basisType = basisType;
            obj.Ycenter = 0;
            obj.Yscale = 1;
            if strcmpi(basisType, "pns")
                center = false;
                scale = false;
            end
            if center
                obj.Ycenter = mean(Y);
            end
            if scale
                obj.Yscale = std(Y);
                obj.Yscale(obj.Yscale==0) = 1;
            end
            Ystandard = (Y-obj.Ycenter)./obj.Yscale;

            if strcmpi(basisType, "pca")
                [V, d] = eig(cov(Ystandard), 'vector');
                [obj.varExplained, ind] = sort(d, 'descend');
                basis = V(:, ind)';
                coefs = Ystandard * basis';
            elseif strcmpi(basisType, "pns")
                [n, d] = size(Y);
                obj.tt = linspace(0, 1, d);

                Y = Y';
                radius = mean(sqrt(sum(Y.^2)));
                pnsdat = Y./repmat(sqrt(sum(Y.^2)),d,1);

                [resmat, PNS] = fastpns(pnsdat, 1);
                coefs = resmat';
                basis = zeros(size(resmat,1), size(Y,1));
                PNS.radius = radius;
                obj.basisConstruct = PNS;

                obj.varExplained = sum(abs(resmat.^2), 2) / n;
            else
                error('Un-supported basisType')
            end

            obj.propVarCumSum = cumsum(obj.varExplained) / sum(obj.varExplained);
            if isnan(nBasis)
                obj.nBasis = find(obj.propVarCumSum <= propVarExplained, 1, 'last');
                if isempty(obj.nBasis)
                    obj.nBasis = 1;
                end
            else
                obj.nBasis = nBasis;
            end

            obj.propVarExplained = obj.propVarCumSum(1:obj.nBasis);
            obj.basis = basis(1:obj.nBasis,:);
            obj.coefs = coefs(:, 1:obj.nBasis);
            Ytrunc = obj.getYtruc();
            obj.truncError = obj.Y - Ytrunc;
        end

        function Ytrunc = getYtruc(obj, Ytest, coefs, nBasis)
            arguments
                obj
                Ytest = nan
                coefs = nan
                nBasis = nan
            end

            if isnan(coefs)
                coefs = obj.getCoefs(Ytest);
            end
            if isnan(nBasis) || nBasis > obj.nBasis
                nBasis = obj.nBasis;
            end
            if strcmpi(obj.basisType, "pns")
                PNS = obj.basisConstruct;
                radius = obj.basisConstruct.radius;
                inmat = zeros(size(PNS.radii,1), size(coefs,1));
                inmat(1:nBasis, :) = coefs(:, 1:nBasis)';
                YtruncStandard = fastPNSe2s(inmat, PNS) * radius;
            else
                YtruncStandard = coefs(:, 1:nBasis) * obj.basis(1:nBasis, :);
            end

            Ytrunc = YtruncStandard * obj.Yscale + obj.Ycenter;
        end

        function coefs = getCoefs(obj, Ytest)
            arguments
                obj
                Ytest = nan
            end

            if isnan(Ytest)
                coefs = obj.coefs;
            else
                YtestStandard = (Ytest - obj.Ycenter) / obj.Yscale;
                if strcmpi(obj.basisType,"pns")
                    [n, d] = size(Ytest);
                    tt = linspace(0, 1, d);
                    psi = zeros(d,n);
                    binsize = mean(diff(tt));
                    for k = 1:n
                        psi(:, k) = sqrt(gradient(Ytest(k, :), binsize));
                    end
                    pnsdat = psi./repmat(sqrt(sum(psi.^2)),d,1);
                    coefs = PNSs2e(pnsdat, obj.basisConstruct);
                else
                    coefs = YtestStandard * obj.basis';
                end
            end
        end
    end
end
