import numpy as np
import progressbar
import warnings


class IPCARegressor:
    """
    This class implements the IPCA algorithm by Kelly, Pruitt, Su (2017).

    Parameters
    ----------

    n_factors : int, default=1
        The number of latent factors to estimate. Note, the number of
        estimated factors is automatically reduced by the number of
        pre-specified factors. For example, if n_factors = 2 and one
        pre-specified factor is passed, then IPCARegressor will estimate
        one factor estimated in addition to the pre-specified factor.

    intercept : boolean, default=False
        Determines whether the model is estimated with or without an intercept

    max_iter : int, default=10000
        Maximum number of alternating least squares updates before the
        estimation is stopped

    iter_tol : float, default=10e-6
        Tolerance threshold for stopping the alternating least squares procedure
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
        data :  numpy array
            Panel of stacked data. Each row corresponds to an observation
            (i, t) where i denotes the entity index and t denotes
            the time index. The panel may be unbalanced. The number of unique
            entities is n_samples, the number of unique dates is n_time, and
            the number of characteristics used as instruments is n_characts.
            The columns of the panel are organized in the following order:

            - Column 1: entity id (i)
            - Column 2: time index (t)
            - Column 3: dependent variable corresponding to observation (i,t)
            - Column 4 to column 4+n_characts: characteristics.

        PSF : numpy array, optional
            Set of pre-specified factors as matrix of dimension (n_PSF, n_time)

        Returns
        -------

        Gamma : numpy array
            Array with dimensions (n_characts, n_factors) containing the
            mapping between characteristics and factors loadings. If there
            are n_PSF many pre-specified factors in the model then the
            matrix returned is of dimension (n_characts, (n_factors+n_PSF)).
            If an intercept is included in the model, its loadings are returned
            in the last column of Gamma.

        Factors : numpy array
            Array with dimensions (n_factors, n_time) containing the estimated
            factors. If pre-specified factors were passed the returned
            array is of dimension ((n_factors - n_PSF), n_time),
            corresponding to the n_factors - n_PSF many factors estimated on
            top of the pre-specified ones.

        """

        if data is None:
            raise ValueError('Must pass panel data.')

        # Handle pre-specified factors
        if PSF is not None:
            self.UsePreSpecFactors = True
        else:
            self.UsePreSpecFactors = False

        # Unpack the Panel
        Z, Y = self._unpack_panel_XY(data)

        if self.UsePreSpecFactors:
            if np.size(PSF, axis=0) == self.n_factors:
                warnings.warn("The number of factors (n_factors) to be "
                              "estimated matches, the number of "
                              "pre-specified factors. No additional factors "
                              "will be estimated. To estimate additional "
                              "factors increase n_factors.")

        # Run IPCA
        Gamma, Factors = self._fit_ipca(Z=Z, Y=Y, PSF=PSF)

        # Store estimates
        self.Gamma_Est = Gamma
        self.Factors_Est = Factors

        if self.UsePreSpecFactors:
            if self.intercept and PSF is not None:
                PSF = np.concatenate((PSF, np.ones((1, len(self.dates)))), axis=0)
            else:
                PSF = np.ones((1, len(self.dates)))
        if self.UsePreSpecFactors:
            self.Factors_Est = np.concatenate((Factors, PSF), axis=0)

        # Compute goodness of fit measures
        data_x = np.delete(data, 2, axis=1)
        Ypred = self.predict(data_x, mean_factor=False)
        Ytrue = data[:, 2]
        self.r2_total = 1-np.nansum((Ypred-Ytrue)**2)/np.nansum(Ytrue**2)
        Ypred = self.predict(data_x, mean_factor=True)
        self.r2_pred = 1-np.nansum((Ypred-Ytrue)**2)/np.nansum(Ytrue**2)

        return Gamma, Factors

    def predict(self, data=None, mean_factor=False):
        """
        Predicts fitted values for a previously fitted regressor

        Parameters
        ----------
        data :  numpy array
            Panel of stacked data. Each row corresponds to an observation
            (i, t) where i denotes the entity index and t denotes
            the time index. The panel may be unbalanced. If an observation
            contains missing data NaN will be returned. Note that the
            number of passed characteristics n_characts must match the
            number of characteristics used when fitting the regressor.
            The columns of the panel are organized in the following order:

            - Column 1: entity id (i)
            - Column 2: time index (t)
            - Column 3 to column 3+n_characts: characteristics.

        mean_factor: boolean
            If true, the estimated factors are averaged in the time-series
            before prediction.


        Returns
        -------
        Ypred : array-like. The length of the returned array matches the
            the length of data. A nan will be returned if there is missing
            characteristics information.
        """

        if data is None:
            raise ValueError('A panel of characteristics data must be provided.')

        n_obs = np.size(data, axis=0)
        Ypred = np.full((n_obs), np.nan)

        for i in range(n_obs):
            if np.any(np.isnan(data[i,:])):
                Ypred[i, 0] = np.nan
            else:
                if mean_factor:
                    Ypred[i] = data[i, 2:].dot(self.Gamma_Est)\
                        .dot(np.mean(self.Factors_Est, axis=1))
                else:
                    Ypred[i] = data[i, 2:].dot(self.Gamma_Est)\
                        .dot(self.Factors_Est[:, self.dates == data[i, 1]])
        return Ypred

    def predictOOS(self, data=None, mean_factor=False):
        """
        Predicts time t+1 observation using an out-of-sample design.

        Parameters
        ----------
        data :  numpy array
            Panel of stacked data. Each row corresponds to an observation
            (i,t) where i denotes the entity index and t denotes
            the time index. All data must correspond to time t, i.e. all
            observations occur on the same date.
            If an observation contains missing data NaN will be returned.
            Note that the number of characteristics (n_characts) passed,
            has to match the number of characteristics used when fitting
            the regressor.
            The columns of the panel are organized in the following order:

            - Column 1: entity id (i)
            - Column 2: time index (t)
            - Column 3: dependent variable corresponding to observation (i,t)
            - Column 4 to column 4+n_characts: characteristics.

        mean_factor: boolean
            If true, the estimated factors are averaged in the time-series
            before prediction.


        Returns
        -------
        Ypred : array-like. The length of the returned array matches the
            the length of data. A nan will be returned if there is missing
            characteristics information.
        """

        if data is None:
            raise ValueError('A panel of characteristics data must be provided.')

        if len(np.unique(data[:,1])) > 1:
            raise ValueError('The panel must only have a single timestamp.')

        n_obs = np.size(data, axis=0)
        Ypred = np.full((n_obs), np.nan)

        Z = data[:, 3:]
        Y = data[:, 2]

        # Compute realized factor returns
        Numer = self.Gamma_Est.T.dot(Z.T).dot(Y)
        Denom = self.Gamma_Est.T.dot(Z.T).dot(Z).dot(self.Gamma_Est)
        Factor_OOS = np.linalg.solve(Denom, Numer.reshape((-1, 1)))
        for i in range(n_obs):
            if np.any(np.isnan(data[i,:])):
                Ypred[i, 0] = np.nan
            else:
                if mean_factor:
                    Ypred[i] = Z[i,:].dot(self.Gamma_Est)\
                        .dot(np.mean(self.Factors_Est, axis=1))
                else:
                    Ypred[i] = Z[i,:].dot(self.Gamma_Est)\
                        .dot(Factor_OOS)
        return Ypred

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
        # n_samples = np.size(Z, axis=0)

        # Handle intercept, effectively treating it as a prespecified factor
        if self.intercept:
            n_factors = self.n_factors + 1
            if PSF is not None:
                PSF = np.concatenate((PSF, np.ones((1, n_time))), axis=0)
            elif PSF is None:
                self.UsePreSpecFactors = True
                PSF = np.ones((1, n_time))
        else:
            n_factors = self.n_factors

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
        Gamma_Old = Gamma_Old[:, :n_factors]
        s = s[:n_factors]
        v = v.T
        v = v[:, :n_factors]
        Factor_Old = np.diag(s).dot(v.T)

        # Estimation Step
        tol_current = 1
        iter = 0
        while((iter <= self.max_iter) and (tol_current > self.iter_tol)):

            if self.UsePreSpecFactors:
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
            print('Step', iter, '- Aggregate Update:', tol_current)
        print('-- Convergence Reached --')
        return Gamma_New, Factor_New

    def _ALS_fit(self, Gamma_Old, W, X, nan_mask, **kwargs):
        """The alternating least squares procedure switches back and forth
        between evaluating the first order conditions for Gamma_Beta, and the
        factors until convergence is reached. This function carries out one
        complete update procedure and will need to be called repeatedly using
        the updated Gamma's and factors as inputs.
        """

        # Determine whether any per-specified factors were passed

        if self.UsePreSpecFactors:
            PSF = kwargs.get("PSF")
            K_PSF, T_PSF = np.shape(PSF)


        # Determine number of factors to be estimated
        T = np.size(nan_mask, axis=1)
        if self.UsePreSpecFactors:
            L, Ktilde = np.shape(Gamma_Old)
            K = Ktilde - K_PSF

        else:
            L, K = np.shape(Gamma_Old)
            Ktilde = K

        # ALS Step 1
        F_New = np.nan
        if K > 0:
            if self.UsePreSpecFactors:
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

        if self.UsePreSpecFactors:
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

    def _unpack_panel_XY(self, P):
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

        self.ids = ids
        self.dates = dates

        return Z, Y
