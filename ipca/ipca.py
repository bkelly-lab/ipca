import numpy as np
import progressbar


class IPCARegressor:
    """
    This class implements the IPCA algorithm by Kelly, Pruitt, Su (2017).

    Parameters
    ----------

    n_factors : int, default=3:
        The number of latent factors to estimate

    intercept : boolean, default=False
        Determines whether the model is estimated with or without an intercept

    max_iter : int, default=10000
        Maximum number of alternating least squares updates before the
        estimation is stopped

    iter_tol : float, default=10e-6
        Tolerance that determines the threshold for stopping the alternating
        least squares updates. If the aggregated changes in updated
        parameters drop below this threshold the procedure is stopped.
    """

    def __init__(self, n_factors=1, intercept=False, max_iter=10000,
                 iter_tol=10e-6):

        # paranoid parameter checking to make it easier for users to know when
        # they have gone awry and to make it safe to assume some variables can
        # only have certain settings
        if not isinstance(n_factors, int) or n_factors < 1:
            raise ValueError('n_factors must be an int greater / equal 1.')
        if not isinstance(intercept, bool):
            raise NotImplementedError('intercept must be  boolean')
        if not isinstance(iter_tol, float) or iter_tol >= 1:
            raise ValueError('Iteration tolerance must be smaller than 1.')

        # Save parameters to the object
        params = locals()
        for k, v in params.items():
            if k != 'self':
                setattr(self, k, v)

    def fit(self, data=None, PSF=None):
        """
        Fits the regressor to the data using an alternating least squares
        scheme.

        Parameters
        ----------

        data : array-like, panel of stacked data. Each row corresponds to an
            observation (i, t) where i denotes the entity index and t denotes
            the time index. The panel may be unbalanced. The number of unique
            entities is n_samples, the number of unique dates is n_time, and
            the number of characteristics used as instruments is n_characts.
            The columns of the panel are organized in the following order:
                COLUMN 1: entity id (i)
                COLUMN 2: time index (t)
                COLUMN 3: dependent variable corresponding to observation (i,t)
                COLUMN 4 to COLUMN 4+n_characts: characteristics.
            The panel is rearranged into a tensor of dimensions
                (n_samples, n_characts, n_time)

        PSF : optional, array-like of shape (n_PSF, n_time),
            i.e. pre-specified factors

        Returns
        -------

        Gamma : array-like with dimensions (n_characts, n_factors). If there
            are n_prespec many pre-specified factors in the model then the
            matrix returned is of dimension (n_characts, (n_factors+n_PSF)).
            If an intercept is included in the model, its loadings are returned
            in the last column of Gamma.

        Factors : array_like with dimensions (n_factors, n_time). If
            pre-specified factors were passed the returned matrix is
            of dimension ((n_factors - n_PSF), n_time), corresponding to the
            n_factors - n_PSF many factors estimated on top of the pre-
            specified ones.
        """

        if data is None:
            raise ValueError('Must pass panel data.')

        # Unpack the Panel
        Z, Y = self._unpack_panel(data)

        # Run IPCA
        Gamma, Factors = self._fit_ipca(Z=Z, Y=Y, PSF=PSF)
        self.Gamma_Est = Gamma
        self.Factors_Est = Factors

        return Gamma, Factors

    def _fit_ipca(self, Z=None, Y=None, PSF=None):
        """
        Fits the regressor to the data using an alternating least squares
        scheme.

        Parameters
        ----------

        Z : array-like of shape (n_samples,n_characts,n_time),
            i.e. characteristics

        Y : array_like of shape (n_samples,n_time), i.e dependent variables

        PSF : optional, array-like of shape (n_PSF, n_time), i.e.
            pre-specified factors

        Returns
        -------

        Gamma : array-like with dimensions (n_characts, n_factors). If there
            are n_prespec many pre-specified factors in the model then the
            matrix returned is of dimension (n_characts, (n_factors+n_PSF)).
            If an intercept is included in the model, its loadings are returned
            in the last column of Gamma.

        Factors : array_like with dimensions (n_factors, n_time). If
            pre-specified factors were passed the returned matrix is
            of dimension ((n_factors - n_PSF), n_time), corresponding to the
            n_factors - n_PSF many factors estimated on top of the pre-
            specified ones.
        """

        if np.size(Z, axis=0) != np.size(Y, axis=0):
            raise ValueError('Number of entities in Z and Y' +
                             'must be identical.')
        if np.size(Y, axis=1) != np.size(Z, axis=2):
            raise ValueError('Number of samples in Z and Y must be identical.')
        if np.size(Z, axis=1) < self.n_factors:
            raise ValueError('Number of factors exceeds number' +
                             'of characteristics.')
        if PSF is not None:
            if np.size(PSF, 1) != np.size(Y, axis=1):
                raise ValueError('Number of samples in PSF' +
                                 'does not match Z or Y')

        # Establish dimensions
        n_time = np.size(Z, axis=2)
        n_characts = np.size(Z, axis=1)
        n_samples = np.size(Z, axis=0)

        # Handle pre-specified factors
        if PSF is not None:
            UsePreSpecFactors = True
        else:
            UsePreSpecFactors = False

        # Handle intercept, effectively treating it as a prespecified factor
        if self.intercept:
            self.n_factors = self.n_factors + 1
            if PSF is not None:
                PSF = np.concatenate((PSF, np.ones((1, n_time))), axis=0)
            elif PSF is None:
                UsePreSpecFactors = True
                PSF = np.ones((1, n_time))

        # Check whether elements are missing
        nan_mask = self._nan_check(Z, Y)
        # Define characteristics weighted matrices
        X = np.full((n_characts, n_time), np.nan)
        for t in range(n_time):
            X[:, t] = np.transpose(Z[nan_mask[:, t], :, t])\
                                   .dot(Y[nan_mask[:, t], t])\
                                   / np.sum(nan_mask[:, t])
        # Define W matrix
        W = np.full((n_characts, n_characts, n_time), np.nan)
        for t in range(n_time):
            W[:, :, t] = np.transpose(Z[nan_mask[:, t], :, t])\
                                      .dot(Z[nan_mask[:, t], :, t])\
                                      / np.sum(nan_mask[:, t])

        # Initialize the Alternating Least Squares Procedure
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

            if UsePreSpecFactors:
                Gamma_New, Factor_New = self._ALS_fit(Gamma_Old, W, X,
                                                      nan_mask, PSF=PSF)
                tol_current = np.amax(Gamma_New.reshape((-1, 1))
                                      - Gamma_Old.reshape((-1, 1)))
            else:
                Gamma_New, Factor_New = self._ALS_fit(Gamma_Old, W, X,
                                                      nan_mask)
                tol_current = np.amax(np.vstack((Gamma_New.reshape((-1, 1))
                                      - Gamma_Old.reshape((-1, 1)),
                                      Factor_New.reshape((-1, 1))
                                      - Factor_Old.reshape((-1, 1)))))

            # Compute update size
            Factor_Old = Factor_New
            Gamma_Old = Gamma_New
            iter += 1
            print(tol_current)

        return Gamma_New, Factor_New

    def _ALS_fit(self, Gamma_Old, W, X, nan_mask, **kwargs):
        """The alternating least squares procedure switches back and forth
        between evaluating the first order conditions for Gamma_Beta, and the
        factors until convergence is reached. This function carries out one
        complete update procedure and will need to be called repeatedly using
        the updated Gamma's and factors as inputs.
        """

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
                    m1 = Gamma_Old[:, :K].T.dot(W[:, :, t])\
                        .dot(Gamma_Old[:, :K])
                    m2 = Gamma_Old[:, :K].T.dot(X[:, t])-Gamma_Old[:, :K].T\
                        .dot(W[:, :, t]).dot(Gamma_Old[:, K:Ktilde])\
                        .dot(PSF[:, t])
                    F_New[:, t] = np.squeeze(np.linalg.solve(
                                                    m1, m2.reshape((-1, 1))))
            else:
                F_New = np.full((K, T), np.nan)
                for t in range(T):
                    m1 = Gamma_Old.T.dot(W[:, :, t]).dot(Gamma_Old)
                    m2 = Gamma_Old.T.dot(X[:, t])
                    F_New[:, t] = np.squeeze(np.linalg.solve(
                                                    m1, m2.reshape((-1, 1))))
        else:
            F_New = np.full((K, T), np.nan)

        # ALS Step 2
        Numer = np.full((L*Ktilde, 1), 0)
        Denom = np.full((L*Ktilde, L*Ktilde), 0)

        if UsePreSpecFactors:
            if K > 0:
                for t in range(T):
                    Numer = Numer + np.kron(X[:, t].reshape((-1, 1)),
                                            np.vstack(
                                            (F_New[:, t].reshape((-1, 1)),
                                             PSF[:, t].reshape((-1, 1)))))\
                                             * np.sum(nan_mask[:, t])
                    Denom_temp = np.vstack((F_New[:, t].reshape((-1, 1)),
                                           PSF[:, t].reshape(-1, 1)))
                    Denom_temp = Denom_temp.dot(Denom_temp.T) \
                        * np.sum(nan_mask[:, t])
                    Denom = Denom + np.kron(W[:, :, t], Denom_temp)
            else:
                for t in range(T):
                    Numer = Numer + np.kron(X[:, t].reshape((-1, 1)),
                                            PSF[:, t].reshape((-1, 1)))\
                                            * np.sum(nan_mask[:, t])
                    Denom = Denom + np.kron(W[:, :, t],
                                            PSF[:, t].reshape((-1, 1))
                                            .dot(PSF[:, t].reshape((-1, 1)).T))\
                        * np.sum(nan_mask[:, t])
        else:
            for t in range(T):
                Numer = Numer + np.kron(X[:, t].reshape((-1, 1)),
                                        F_New[:, t].reshape((-1, 1)))\
                                        * np.sum(nan_mask[:, t])
                Denom = Denom + np.kron(W[:, :, t],
                                        F_New[:, t].reshape((-1, 1))
                                        .dot(F_New[:, t].reshape((1, -1)))) \
                    * np.sum(nan_mask[:, t])

        Gamma_New_trans_vec = np.linalg.solve(Denom, Numer)
        Gamma_New = Gamma_New_trans_vec.reshape((L, Ktilde))

        # Enforce Orthogonality of Gamma_Beta and factors F
        if K > 0:
            R1 = np.linalg.cholesky(Gamma_New[:, :K].T.dot(Gamma_New[:, :K])).T
            R2, _, _ = np.linalg.svd(R1.dot(F_New).dot(F_New.T).dot(R1.T))
            Gamma_New[:, :K] = np.linalg.lstsq(Gamma_New[:, :K].T,
                                               R1.T, rcond=None)[0]\
                .dot(R2)
            F_New = np.linalg.solve(R2, R1.dot(F_New))

        # Enforce sign convention for Gamma_Beta and F_New
        if K > 0:
            sg = np.sign(np.mean(F_New, axis=1)).reshape((-1, 1))
            sg[sg == 0] = 1
            Gamma_New[:, :K] = np.multiply(Gamma_New[:, :K], sg.T)
            F_New = np.multiply(F_New, sg)

        return Gamma_New, F_New

    def _nan_check(self, Z, Y):
        """This function checks whether an element in the pair of
        Z[n,:,t] and Y[n,t] is missing and returns a matrix of dimension
        (n_samples, n_time) containing boolean values. The output is False
        whenever there is a missing value in the pair and true otherwise

        Parameters
        ----------

        Z: array-like of shape (n_samples,n_characts,n_time)

        Y: array_like of shape (n_samples,n_time)
        ----------
        """
        n_time = np.size(Z, axis=2)
        n_samples = np.size(Z, axis=0)
        # Handle missing observations
        Z_nan = np.isnan(Z)
        Y_nan = np.isnan(Y)
        nan_mask = np.full((n_samples, n_time), np.nan)
        # ProgressBar
        bar = progressbar.ProgressBar(maxval=n_samples,
                                      widgets=[progressbar.Bar('=', '[', ']'),
                                               ' ', progressbar.Percentage()])
        print("Obtaining NaN Locations...")
        bar.start()
        for n in range(n_samples):
            for t in range(n_time):
                    nan_mask[n, t] = ~np.any(Z_nan[n, :, t]) \
                        and ~np.any(Y_nan[n, t])
            bar.update(n)
        bar.finish()
        print("Done!")

        return np.array(nan_mask, dtype=bool)

    def _unpack_panel(self, P):
        """ Converts a stacked panel of data where each row corresponds to an
        observation (i, t) into a tensor of dimensions (N, L, T) where N is the
        number of unique entities, L is the number of characteristics and T is
        the number of unique dates

        Parameters
        ----------

        P: Panel of data. Each row corresponds to an observation (i, t). The
            columns are ordered in the following manner:
                COLUMN 1: entity id (i)
                COLUMN 2: time index (t)
                COLUMN 3: depdent variable Y(i,t)
                COLUMN 4 and following: L characteristics

        Returns
        -------
        Z: array-like, tensor of dimensions (N, L, T), containing
            the characteristics

        Y: array-like, matrix of dimension, containing the dependent variable.
        """

        dates = np.unique(P[:, 1])
        ids = np.unique(P[:, 0])
        T = np.size(dates, axis=0)
        N = np.size(ids, axis=0)
        L = np.size(P, axis=1) - 3
        print('The panel dimensions are:')
        print('n_samples:', N, ', n_characts:', L, ', n_time:', T)

        # Construct Z, Y
        Z = np.full((N, L, T), np.nan)
        Y = np.full((N, T), np.nan)

        bar = progressbar.ProgressBar(maxval=N,
                                      widgets=[progressbar.Bar('=', '[', ']'),
                                               ' ', progressbar.Percentage()])
        print("Unpacking Panel...")
        bar.start()
        temp = np.full((1, L+3), np.nan)
        for n_i, n in enumerate(ids):
            ixd = np.isin(dates, P[P[:, 0] == n, 1])
            temp_n = np.full((T, L+3), np.nan)
            temp_n[:, 0] = n
            temp_n[:, 1] = dates
            temp_n = temp_n[np.invert(ixd), :]
            temp = np.append(temp, temp_n, axis=0)
            bar.update(n)
        bar.finish()
        temp = temp[1:, :]  # get rid of row used for initialization

        # Append the missing observations to create balanced panel
        P = np.append(P, temp, axis=0)
        # Sort observations such that T observations for each n in N are
        # stacked vertically
        ind = np.lexsort((P[:, 1], P[:, 0]))
        P = P[ind, :]
        # Reshape the panel into Z (N, L, T) and Y(N, T)
        Z = np.dstack(np.split(P[:, 3:], N, axis=0))
        Z = np.swapaxes(Z, 0, 2)
        Y = np.reshape(P[:, 2], (N, T))
        return Z, Y
