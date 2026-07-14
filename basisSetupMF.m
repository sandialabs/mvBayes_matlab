classdef basisSetupMF
    % Compute basis components for a matrix Y. Used in mvBayes.

    properties
        Y
        Z
        nMV
        basisType
        varExplained
        Ycenter
        Yscale
        Zcenter
        propVarExplained
        propVarCumSum
        truncError
        nBasis
        basis
        basisConstruct
        coefs
        coefs_lf
        tt
    end

    methods
        function obj = basisSetupMF(Y, Z, basisType, mL)
            arguments
                Y {mustBeNumeric}
                Z {mustBeNumeric}
                basisType = "pca"
                mL = 20
            end

            obj.Y = Y;
            obj.Z = Z;
            obj.nMV = size(Y,2);
            obj.basisType = basisType;
            obj.Ycenter = 0;
            obj.Yscale = 1;

            if scale
                obj.Yscale = std(Y);
                obj.Yscale(obj.Yscale==0) = 1;
            end

            if strcmpi(basisType, "pca")
                [Phi, obj.Ycenter, explained] = build_hf_basis(obj.Y');                       
                [obj.basis, dopt, ~, ~] = enrich_basis_multifidelity(Phi, obj.Ycenter, obj.Z', mL, .999); 
                obj.Zcenter = obj.Ycenter + dopt;     % Eq. 2.5

                obj.coefs     = obj.basis' * (obj.Y - obj.Zcenter);           % m x MH   HF POD coefficients
                obj.coefs_lf  = obj.basis' * (obj.Z - obj.Zcenter);           % m x ML   LF POD coefficients
            else
                error('Un-supported basisType')
            end

            
            obj.nBasis = size(obj.basis,1);

            obj.propVarExplained = explained;
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
            YtruncStandard = coefs(:, 1:nBasis) * obj.basis(1:nBasis, :);

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
                coefs = YtestStandard * obj.basis';
            end
        end
    end
end
