import numpy as np


class IPCARegressor:
    """
    This class implements the PPR algorithm as detailed in math.pdf.
    Parameters
    ----------
    n_factors : int, default=3:
        The number of latent factors to estimate
    intercept : str, default='restricted'
        Determines whether the model is estimated with a restricted intercept
        or whether instrumental variables are used in intercept expression
    """

    def __init__(self, n_factors=1, intercept='restricted', max_iter=10000,
                 iter_tol=10e-6):

        # paranoid parameter checking to make it easier for users to know when
        # they have gone awry and to make it safe to assume some variables can
        # only have certain settings
        if not isinstance(n_factors, int) or n_factors < 1:
            raise ValueError('n_factors must be an int greater / equal 1.')
        if intercept not in ['unrestricted', 'restricted']:
            raise NotImplementedError('intercept ' + intercept + ' not supported')
        if not isinstance(iter_tol, float) or iter_tol >= 1:
            raise ValueError('Iteration tolerance must be smaller than 1.')

        # Save parameters to the object
        params = locals()
        for k, v in params.items():
            if k != 'self':
                setattr(self, k, v)

    def fit(self, Z=None, Y=None, PSF=None):
        """Fits the regressor to the data
        Parameters
        ----------
        Z: array-like of shape (n_samples,n_characts,n_time), i.e. characteristics
        Y: array_like of shape (n_samples,n_time), i.e dependent variables

        Optional parameters
        -------------------
        PSF: array-like of shape (n_PSF, n_time), i.e. pre-specified factors
        """

        if np.size(Z, axis=0) != np.size(Y, axis=0):
            raise ValueError('Number of entities in Z and Y must be identical.')
        if np.size(Y, axis=1) != np.size(Z, axis=2):
            raise ValueError('Number of samples in Z and Y must be identical.')
        if np.size(Z, axis=1) < self.n_factors:
            raise ValueError('Number of factors exceeds number of characteristics.')
        if PSF is not None:
            if np.size(PSF, 1) != np.size(Y, axis=1):
                raise ValueError('Number of samples in PSF does not match Z or Y')

        # Establish dimensions
        n_time = np.size(Z, axis=2)
        n_characts = np.size(Z, axis=1)
        n_samples = np.size(Z, axis=0)

        # Obtain pre-specified factors if passed
        if PSF is not None:
            UsePreSpecFactors = True
        else:
            UsePreSpecFactors = False

        # Check whether elements are missing
        nan_mask = self._nan_check(Z, Y)

        # Define characteristics weighted matrices
        X = np.full((n_characts, n_time), np.nan)
        for t in range(n_time):
            X[:, t] = np.transpose(Z[np.squeeze(nan_mask[:, t]), :, t]).dot(Y[np.squeeze(nan_mask[:, t]), t])/np.sum(nan_mask[:, t])
        # Define W matrix
        W = np.full((n_characts, n_characts, n_time), np.nan)
        for t in range(n_time):
            W[:, :, t] = np.transpose(Z[np.squeeze(nan_mask[:, t]), :, t]).dot(np.squeeze(Z[nan_mask[:, t], :, t]))/np.sum(nan_mask[:, t])

        # Initialization ALS
        Gamma_Old, s, v = np.linalg.svd(X)
        Gamma_Old = Gamma_Old[:, :self.n_factors]
        s = s[:self.n_factors]
        v = v.T
        v = v[:, :self.n_factors]
        Factor_Old = np.diag(s).dot(v.T)

        # Estimation Step
        tol_current = 1
        iter = 0

        while((iter <= self.max_iter) and (tol_current > self.iter_tol)):
            # Alternating least squares procedure
            if UsePreSpecFactors:
                Gamma_New, Factor_New = self.ALS_fit(Gamma_Old, W, X, nan_mask, PSF=PSF)
                tol_current = np.amax(Gamma_New.reshape((-1, 1))-Gamma_Old.reshape((-1, 1)))
            else:
                Gamma_New, Factor_New = self.ALS_fit(Gamma_Old, W, X, nan_mask)
                tol_current = np.amax(np.vstack((Gamma_New.reshape((-1, 1))-Gamma_Old.reshape((-1, 1)),Factor_New.reshape((-1,1))-Factor_Old.reshape((-1,1)))))

            # Compute update size
            Factor_Old = Factor_New
            Gamma_Old = Gamma_New
            iter += 1
            print(tol_current)

        return Gamma_New, Factor_New

    def ALS_fit(self, Gamma_Old, W, X, nan_mask, **kwargs):

        # Determine whether any per-specified factors were passed
        UsePreSpecFactors = False
        if 'PSF' in kwargs:
            PSF = kwargs.get("PSF")
            K_PSF, T_PSF = np.shape(PSF)
            UsePreSpecFactors = True

        # Determine number of factors to be estimated
        T = np.size(nan_mask, axis=1)
        if UsePreSpecFactors:
            L, Ktilde = np.shape(Gamma_Old)
            K = Ktilde - K_PSF
        else:
            L, K = np.shape(Gamma_Old)
            Ktilde = K

        # ALS Step 1
        F_New = np.nan
        if K > 0:
            if UsePreSpecFactors:
                F_New = np.full((K, T), np.nan)
                for t in range(T):
                    m1 = Gamma_Old[:, :K].T.dot(W[:, :, t]).dot(Gamma_Old[:, :K])
                    m2 = Gamma_Old[:, :K].T.dot(X[:, t])-Gamma_Old[:, :K].T.dot(W[:, :, t]).dot(Gamma_Old[:, K:Ktilde]).dot(PSF[:, t])
                    F_New[:, t] = np.squeeze(np.linalg.solve(m1, m2.reshape((-1, 1))))
            else:
                F_New = np.full((K, T), np.nan)
                for t in range(T):
                    m1 = Gamma_Old.T.dot(W[:, :, t]).dot(Gamma_Old)
                    m2 = Gamma_Old.T.dot(X[:, t])
                    F_New[:, t] = np.squeeze(np.linalg.solve(m1, m2.reshape((-1, 1))))
        else:
            F_New = np.full((K, T), np.nan)

        # ALS Step 2
        Numer = np.full((L*Ktilde, 1), 0)
        Denom = np.full((L*Ktilde, L*Ktilde), 0)

        if UsePreSpecFactors:
            if K > 0:
                for t in range(T):
                    Numer = Numer + np.kron(X[:, t].reshape((-1, 1)), np.vstack((F_New[:, t].reshape((-1, 1)), PSF[:, t].reshape((-1, 1)))))*np.sum(nan_mask[:, t])
                    Denom_temp = np.vstack((F_New[:, t].reshape((-1, 1)), PSF[:, t].reshape(-1, 1)))
                    Denom_temp = Denom_temp.dot(Denom_temp.T)*np.sum(nan_mask[:, t])
                    Denom = Denom + np.kron(W[:, :, t], Denom_temp)
            else:
                for t in range(T):
                    Numer = Numer + np.kron(X[:, t].reshape((-1, 1)), PSF[:, t].reshape((-1, 1)))*np.sum(nan_mask[:, t])
                    Denom = Denom + np.kron(W[:, :, t], PSF[:, t].reshape((-1, 1)).dot(PSF[:, t].reshape((-1, 1)).T))*np.sum(nan_mask[:, t])
        else:
            for t in range(T):
                Numer = Numer + np.kron(X[:, t].reshape((-1, 1)), F_New[:, t].reshape((-1, 1)))*np.sum(nan_mask[:, t])
                Denom = Denom + np.kron(W[:, :, t], F_New[:, t].reshape((-1, 1)).dot(F_New[:, t].reshape((1, -1))))*np.sum(nan_mask[:, t])

        Gamma_New_trans_vec = np.linalg.solve(Denom, Numer)
        Gamma_New = Gamma_New_trans_vec.reshape((L, Ktilde))

        # Enforce Orthogonality of Gamma_Beta and factors F
        if K > 0:
            R1 = np.linalg.cholesky(Gamma_New[:, :K].T.dot(Gamma_New[:, :K])).T
            R2, _, _ = np.linalg.svd(R1.dot(F_New).dot(F_New.T).dot(R1.T))
            Gamma_New[:, :K] = np.linalg.lstsq(Gamma_New[:, :K].T, R1.T)[0].dot(R2)
            F_New = np.linalg.solve(R2, R1.dot(F_New))

        # Enforce sign convention for Gamma_Beta and F_New
        if K > 0:
            sg = np.sign(np.mean(F_New, axis=1)).reshape((-1, 1))
            sg[sg == 0] = 1
            Gamma_New[:, :K] = np.multiply(Gamma_New[:, :K], sg.T)
            F_New = np.multiply(F_New, sg)

        return Gamma_New, F_New

    def _nan_check(self, Z, Y):
        """This function checks whether an element in the pair of Z[n,:,t] and Y[n,t]
        is missing and returns a matrix of dimension (n_samples, n_time) containing
        boolean values. The output is False whenever there is a missing value in the
        pair.
        ----------
        Parameters
        Z: array-like of shape (n_samples,n_characts,n_time)
        Y: array_like of shape (n_samples,n_time)
        ----------
        """
        n_time = np.size(Z, axis=2)
        n_samples = np.size(Z, axis=0)

        # Handle missing observations
        nan_mask = np.full((n_samples, n_time), np.nan)
        for n in range(n_samples):
            for t in range(n_time):
                    nan_mask[n, t] = np.any(np.isnan(Z[n, :, t])) or np.any(np.isnan(Y[n, t]))
        nan_mask = np.logical_not(nan_mask)
        return nan_mask
