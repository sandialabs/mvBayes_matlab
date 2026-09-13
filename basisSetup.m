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
        basisMean       % mean removed by the basis construction itself (PCA)
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
        function obj = basisSetup(Y, basisType, customBasis, nBasis, propVarExplained, center, scale, thresh, basisTransform)
            %BASISSETUP Compute basis components for the response matrix Y.
            %
            %   Y                : n x nMV response matrix.
            %   basisType        : "pca", "pns", "bspline", "legendre" or "custom".
            %   customBasis      : k x nMV basis matrix, required when
            %                      basisType is "custom"; ignored otherwise.
            %   nBasis           : number of components. Required for "bspline"
            %                      and "legendre"; NaN selects it from
            %                      propVarExplained where that is possible.
            %   propVarExplained : proportion of variance to explain when
            %                      choosing nBasis (default 0.99).
            %   center, scale    : whether to center/scale Y beforehand.
            %   thresh           : eigenvalue threshold, PNS only.
            %   basisTransform   : optional metric in which a custom basis is
            %                      orthonormalized (used by supervised basis
            %                      updates).
            arguments
                Y {mustBeNumeric}
                basisType = "pca"
                customBasis = []
                nBasis = nan
                propVarExplained = 0.99
                center = true
                scale = false
                thresh = 1e-15
                basisTransform = []
            end

            obj.Y = Y;
            obj.nMV = size(Y,2);
            obj.basisType = basisType;
            obj.Ycenter = 0;
            obj.Yscale = 1;
            obj.basisMean = 0;
            if strcmpi(basisType, "pns")
                center = false;
                scale = false;
            end
            if center
                obj.Ycenter = mean(Y, 1);
            end
            if scale
                % Normalize by N (not N-1) to match numpy's np.std default.
                obj.Yscale = std(Y, 1, 1);
                obj.Yscale(obj.Yscale==0) = 1;
            end
            obj.center = center;
            obj.scale = scale;
            Ystandard = (Y - obj.Ycenter) ./ obj.Yscale;

            if strcmpi(basisType, "pca")
                % Singular value decomposition, matching sklearn.decomposition.PCA:
                % the data are centered internally, explained variance is the
                % unbiased eigenvalue, and components are sign-flipped by
                % sklearn's svd_flip with u_based_decision = false, i.e. the
                % largest-magnitude loading of each component is made positive.
                % (Signs are a convention only; reconstructions are unaffected.)
                n = size(Ystandard, 1);
                obj.basisMean = mean(Ystandard, 1);
                Ycentered = Ystandard - obj.basisMean;

                [U, S, V] = svd(Ycentered, 'econ');
                sv = diag(S);
                basis = V';                           % components, one per row

                [~, jMax] = max(abs(basis), [], 2);
                signs = sign(basis(sub2ind(size(basis), (1:size(basis,1))', jMax)));
                signs(signs == 0) = 1;
                basis = basis .* signs;
                U = U .* signs.';

                obj.varExplained = (sv.^2) / (n - 1);
                coefs = U .* sv.';                    % == Ycentered * basis'
            elseif any(strcmpi(basisType, ["bspline", "legendre", "custom"]))
                if any(strcmpi(basisType, ["bspline", "legendre"]))
                    if isnan(nBasis)
                        error('basisSetup:nBasisRequired', ...
                            "nBasis must be specified for basisType='%s'", basisType);
                    end
                    if strcmpi(basisType, "bspline")
                        if nBasis < 3
                            error('basisSetup:nBasisTooSmall', ...
                                "Must have nBasis >= 3 for basisType='bspline'");
                        end
                        customBasis = basisBspline(linspace(0,1,obj.nMV), nBasis);
                    else
                        if mod(nBasis, 2) == 1
                            fprintf("nBasis must be even for basisType='legendre'. Setting nBasis+=1\n");
                            nBasis = nBasis + 1;
                        end
                        customBasis = basisLegendre(linspace(0,1,obj.nMV), nBasis/2, 1);
                    end
                else
                    if isempty(customBasis)
                        error('basisSetup:customBasisRequired', ...
                            "Must provide customBasis if basisType=='custom'");
                    end
                    if size(customBasis,2) ~= obj.nMV
                        error('basisSetup:customBasisSize', ...
                            'size(customBasis,2) ~= size(Y,2)');
                    end
                end

                obj.basisConstruct = customBasisConstruct(customBasis, Ystandard, basisTransform);
                basis = obj.basisConstruct.basis;
                coefs = obj.basisConstruct.coefs;
                obj.varExplained = obj.basisConstruct.varExplained;
            elseif strcmpi(basisType, "pns")
                [n, d] = size(Y);
                obj.tt = linspace(0, 1, d);

                Yt = Y';
                radius = mean(sqrt(sum(Yt.^2)));
                pnsdat = Yt ./ repmat(sqrt(sum(Yt.^2)), d, 1);

                % n_pc = 1 selects the "Approx" (99% variance) rule in fastpns.
                [resmat, PNS] = fastpns(pnsdat, 1, 1, 0.05, 100, thresh);
                coefs = resmat';
                basis = zeros(size(resmat,1), size(Yt,1));
                PNS.radius = radius;
                obj.basisConstruct = PNS;

                obj.varExplained = sum(abs(resmat.^2), 2) / n;
            else
                error('basisSetup:badBasisType', 'Un-supported basisType')
            end

            obj.propVarCumSum = cumsum(obj.varExplained) / sum(obj.varExplained);
            if isnan(nBasis)
                % Smallest number of components explaining at least
                % propVarExplained of the variance.
                nBasis = find(obj.propVarCumSum > propVarExplained, 1, 'first');
                if isempty(nBasis)
                    nBasis = numel(obj.propVarCumSum);
                end
            end
            obj.nBasis = min(nBasis, size(basis,1));

            obj.propVarExplained = obj.propVarCumSum(obj.nBasis);
            obj.basis = basis(1:obj.nBasis,:);
            obj.coefs = coefs(:, 1:obj.nBasis);
            Ytrunc = obj.getYtrunc();
            obj.truncError = obj.Y - Ytrunc;
        end

        function Ytrunc = getYtrunc(obj, Ytest, coefs, nBasis)
            %GETYTRUNC Reconstruction of Ytest from the first nBasis components.
            %
            %   If Ytest is supplied, its coefficients are computed; if coefs is
            %   supplied, it is used directly; if neither is supplied, obj.coefs
            %   is used. Pass [] to leave an argument unset.
            arguments
                obj
                Ytest = []
                coefs = []
                nBasis = []
            end

            if isempty(coefs)
                coefs = obj.getCoefs(Ytest);
            end
            if isempty(nBasis) || nBasis > obj.nBasis
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

            Ytrunc = YtruncStandard .* obj.Yscale + obj.Ycenter;
        end

        function coefs = getCoefs(obj, Ytest)
            %GETCOEFS Project Ytest onto the basis. Pass [] to return obj.coefs.
            arguments
                obj
                Ytest = []
            end

            if isempty(Ytest)
                coefs = obj.coefs;
            else
                YtestStandard = (Ytest - obj.Ycenter) ./ obj.Yscale;
                if strcmpi(obj.basisType,"pns")
                    [n, d] = size(Ytest);
                    ttLocal = linspace(0, 1, d);
                    psi = zeros(d,n);
                    binsize = mean(diff(ttLocal));
                    for k = 1:n
                        psi(:, k) = sqrt(gradient(Ytest(k, :), binsize));
                    end
                    pnsdat = psi./repmat(sqrt(sum(psi.^2)),d,1);
                    coefs = PNSs2e(pnsdat, obj.basisConstruct);
                    coefs = coefs(:, 1:obj.nBasis);
                elseif strcmpi(obj.basisType, "pca")
                    coefs = (YtestStandard - obj.basisMean) * obj.basis';
                else
                    % bspline, legendre, custom: project through the basis
                    % construction, which may carry a basisTransform.
                    coefs = obj.basisConstruct.transform(YtestStandard);
                    coefs = coefs(:, 1:obj.nBasis);
                end
            end
        end

        function Ytest = preprocessY(obj, Ytest)
            %PREPROCESSY Hook for subclasses. Returns Ytest, or obj.Y if unset.
            arguments
                obj
                Ytest = []
            end
            if isempty(Ytest)
                Ytest = obj.Y;
            end
        end
    end
end
