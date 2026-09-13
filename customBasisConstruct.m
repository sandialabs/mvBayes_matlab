classdef customBasisConstruct
    %CUSTOMBASISCONSTRUCT Project Y onto a user-supplied (or generated) basis.
    %
    %   Used by basisSetup for basisType "bspline", "legendre" and "custom".
    %   The supplied basis is orthonormalized if it is not already orthonormal;
    %   with a basisTransform it is orthonormalized in the metric that transform
    %   induces.

    properties
        basis
        coefs
        varExplained
        basisTransform
        basisTransformSqrt
        logDet          % set by mvBayes.updateBasis; unused otherwise
    end

    methods
        function obj = customBasisConstruct(customBasis, Ystandard, basisTransform)
            arguments
                customBasis {mustBeNumeric}
                Ystandard {mustBeNumeric}
                basisTransform = []
            end

            obj.basisTransform = basisTransform;

            if isempty(basisTransform)
                varTotal = sum(var(Ystandard, 1, 1));
                if isOrthogonal(customBasis)
                    obj.basis = customBasis;
                else
                    obj.basis = orthogonalize(customBasis);
                end
            else
                % Symmetrize before the eigendecomposition so that a transform
                % assembled from a covariance estimate stays numerically Hermitian.
                [Q, D] = eig((basisTransform + basisTransform.') / 2, 'vector');
                obj.basisTransformSqrt = Q * diag(sqrt(D)) * Q.';
                varTotal = sum(var(Ystandard * obj.basisTransformSqrt, 1, 1));
                if isOrthogonal(customBasis * obj.basisTransformSqrt)
                    obj.basis = customBasis;
                else
                    covSqrt = Q * diag(sqrt(1 ./ D)) * Q.';
                    obj.basis = orthogonalize(customBasis * obj.basisTransformSqrt) * covSqrt;
                end
            end

            obj.coefs = obj.transform(Ystandard);

            % Normalize by N (not N-1) to match numpy's np.var default.
            obj.varExplained = var(obj.coefs, 1, 1).';

            % When the basis does not span the full multivariate index, the
            % leftover variance becomes a final "truncation" component.
            if size(customBasis,1) < size(customBasis,2)
                obj.varExplained = [obj.varExplained; varTotal - sum(obj.varExplained)];
            end
        end

        function coefs = transform(obj, Ystandard)
            %TRANSFORM Project standardized responses onto the basis.
            if isempty(obj.basisTransform)
                coefs = Ystandard * obj.basis.';
            else
                coefs = Ystandard * obj.basisTransform * obj.basis.';
            end
        end
    end
end
