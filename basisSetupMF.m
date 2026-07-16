classdef basisSetupMF
    % Compute basis components for a matrix Y. Used in mvBayes.

    properties
        Y
        Z
        nMV
        basisType
        varExplained
        Ycenter
        Zcenter
        propVarExplained
        propVarCumSum
        truncError
        nBasis
        basis
        basisConstruct
        coefs
        coefs_br
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

            if strcmpi(basisType, "pca")
                [Phi, obj.Ycenter, latent] = build_hf_basis(obj.Y');                       
                [Psi, dopt, ~, ~] = enrich_basis_multifidelity(Phi, obj.Ycenter, obj.Z', mL, .999);
                obj.basis = Psi';
                obj.Zcenter = (obj.Ycenter + dopt)';     % Eq. 2.5
                obj.Ycenter = obj.Ycenter';

                obj.coefs     = (obj.basis * (obj.Y - obj.Zcenter)')';           % m x MH   HF POD coefficients
                obj.coefs_br  = (obj.basis * (obj.Z - obj.Zcenter)')';           % m x ML   LF POD coefficients

                obj.propVarExplained = cumsum(latent)/sum(latent);
                obj.varExplained = latent;
            else
                error('Un-supported basisType')
            end

            
            obj.nBasis = size(obj.basis,1);

            Ytrunc = obj.getYtruc();
            obj.truncError = obj.Y - Ytrunc;
        end

        function Ytrunc = getYtruc(obj, Ytest, coefs, coefs_br, nBasis)
            arguments
                obj
                Ytest = nan
                coefs = nan
                coefs_br = nan
                nBasis = nan
            end

            if isnan(nBasis) || nBasis > obj.nBasis
                nBasis = obj.nBasis;
            end

            if isnan(coefs)
                [coefs, coefs_br] = obj.getCoefs(Ytest);
                YtruncStandard = (coefs(:, 1:nBasis)) * obj.basis(1:nBasis, :);
            else
                YtruncStandard = (coefs(:, 1:nBasis) + coefs_br(:, 1:nBasis)) * obj.basis(1:nBasis, :);
            end

            Ytrunc = YtruncStandard + obj.Zcenter;
        end

        function [coefs, coefs_br] = getCoefs(obj, Ytest)
            arguments
                obj
                Ytest = nan
            end

            if isnan(Ytest)
                coefs = obj.coefs;
                coefs_br = obj.coefs_br;
            else
                YtestStandard = (Ytest - obj.Ycenter);
                coefs = YtestStandard * obj.basis';
            end
        end
    end
end
