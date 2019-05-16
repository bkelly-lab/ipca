import numpy as np
import scipy as sp
import progressbar
import warnings
from numba import jit
import time
from joblib import Parallel, delayed

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

    def fit(self, P=None, PSF=None, refit=False):
        """
        Fits the regressor to the data using an alternating least squares
        scheme.

        Parameters
        ----------
        P :  numpy array
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

        refit : boolean, optional
            Indicates whether the regressor should be re-fit. If set to True
            the function will skip unpacking the panel into a tensor and
            instead use the stored values from the previous fit. Note, it is
            still necessary to pass the previously used P.

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
        # Check re-fitting is valied
        if refit:
            try:
                self.X
            except AttributeError:
                raise ValueError('Refit only possible after initial fit.')

        # Handel paenl
        if P is None:
            raise ValueError('Must pass panel data.')
        else:
            # remove panel rows containing missing obs
            P = P[~np.any(np.isnan(P), axis=1)]

        # Unpack the Panel
        if not refit:
            X, W, val_obs = self._unpack_panel(P)

        # Handle pre-specified factors
        if PSF is not None:
            if np.size(PSF, axis=1) != np.size(np.unique(P[:, 1])):
                raise ValueError("""Number of PSF observations must match
                                 number of unique dates in panel P""")
            self.UsePreSpecFactors = True
        else:
            self.UsePreSpecFactors = False

        if self.UsePreSpecFactors:
            if np.size(PSF, axis=0) == self.n_factors:
                warnings.warn("The number of factors (n_factors) to be "
                              "estimated matches, the number of "
                              "pre-specified factors. No additional factors "
                              "will be estimated. To estimate additional "
                              "factors increase n_factors.")

        # Handle intercept
        if self.intercept and (self.n_factors == np.size(P, axis=1)-3):
            raise ValueError("""Number of factors + intercept higher than
                                number of characteristics. Reduce number of
                                factors or set intercept to false.""")

        #  Treating intercept it as a prespecified factor
        if self.intercept:
            self.n_factors_eff = self.n_factors + 1
            if PSF is not None:
                PSF = np.concatenate((PSF, np.ones((1, self.T))), axis=0)
            elif PSF is None:
                PSF = np.ones((1, self.T))
        else:
            self.n_factors_eff = self.n_factors

        # Check enough features provided
        if np.size(P, axis=1) - 3 < self.n_factors_eff:
            raise ValueError('The number of factors requested exceeds number of features')

        # Determine fit case - if intercept or PSF or both use PSFcase fitting
        # Note PSFcase in contrast to UsePreSpecFactors is only indicating
        # that the IPCA fitting is carried out as if PSF were passed even if
        # only an intercept was passed.
        if self.UsePreSpecFactors or self.intercept:
            self.PSFcase = True
        else:
            self.PSFcase = False

        # Run IPCA
        if not refit:
            Gamma, Factors = self._fit_ipca(X=X, W=W, PSF=PSF,
                                            val_obs=val_obs)
        else:
            Gamma, Factors = self._fit_ipca(X=self.X, W=self.W, PSF=PSF,
                                            val_obs=self.val_obs)

        # Store estimates
        if self.PSFcase:
            if self.intercept and self.UsePreSpecFactors:
                PSF = np.concatenate((PSF, np.ones((1, len(self.dates)))), axis=0)
            elif self.intercept:
                PSF = np.ones((1, len(self.dates)))

            Factors = np.concatenate((Factors, PSF), axis=0)

        self.Gamma_Est = Gamma
        self.Factors_Est = Factors


        # Save unpacked panel for Re-fitting
        if not refit:
            self.PSF = PSF
            self.X = X
            self.W = W
            self.val_obs = val_obs

        # Compute Goodness of Fit
        self.r2_total, self.r2_pred, self.r2_total_x, self.r2_pred_x = \
            self._R2_comps(P=P)

        return self.Gamma_Est, self.Factors_Est

    def predict(self, P=None, mean_factor=False):
        """
        Predicts fitted values for a previously fitted regressor

        Parameters
        ----------
        P :  numpy array
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

        Ypred : numpy array
            The length of the returned array matches the
            the length of data. A nan will be returned if there is missing
            characteristics information.
        """

        if P is None:
            raise ValueError('A panel of characteristics data must be provided.')

        if np.any(np.isnan(P)):
            raise ValueError('Cannot contain missing observations / nan values.')
        n_obs = np.size(P, axis=0)
        Ypred = np.full((n_obs), np.nan)

        mean_Factors_Est = np.mean(self.Factors_Est, axis=1).reshape((-1, 1))

        if mean_factor:
            Ypred[:] = np.squeeze(P[:, 2:].dot(self.Gamma_Est)\
                .dot(mean_Factors_Est))
        else:

            for t_i, t in enumerate(self.dates):
                ix = (P[:, 1] == t)
                Ypred[ix] = np.squeeze(P[ix, 2:].dot(self.Gamma_Est)\
                    .dot(self.Factors_Est[:, t_i]))
        return Ypred

    def BS_Walpha(self, ndraws=1000, n_jobs=-1, backend='loky'):
        """
        Bootstrap inference on the hypotheses Gamma_alpha = 0

        Parameters
        ----------

        ndraws  : integer, default=1000
            Number of bootstrap draws and re-estimations to be performed

        backend : optional

        Returns
        -------

        pval : float
            P-value from the hypothesis test H0: Gamma_alpha=0
        """

        if not self.intercept:
            raise ValueError('Need to fit model with intercept first.')

        # Compute Walpha
        Walpha = self.Gamma_Est[-1, :].T.dot(self.Gamma_Est[-1, :])

        # Compute residuals
        d = np.full((self.L, self.T), np.nan)

        for t_i, t in enumerate(self.dates):
            d[:, t_i] = self.X[:, t_i]-self.W[:, :, t_i].dot(self.Gamma_Est)\
                .dot(self.Factors_Est[:, t_i])

        print("Starting Bootstrap...")
        Walpha_b = Parallel(n_jobs=n_jobs, backend=backend, verbose=10)(
            delayed(_BS_Walpha_sub)(self, n, d) for n in range(ndraws))
        print("Done!")

        # print(Walpha_b, Walpha)
        pval = np.sum(Walpha_b > Walpha)/ndraws
        return pval

    def predictOOS(self, P=None, mean_factor=False):
        """
        Predicts time t+1 observation using an out-of-sample design.

        Parameters
        ----------
        P :  numpy array
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

        Ypred : numpy array
            The length of the returned array matches the
            the length of data. A nan will be returned if there is missing
            characteristics information.
        """

        if P is None:
            raise ValueError('A panel of characteristics data must be provided.')

        if len(np.unique(P[:, 1])) > 1:
            raise ValueError('The panel must only have a single timestamp.')

        n_obs = np.size(P, axis=0)
        Ypred = np.full((n_obs), np.nan)

        # Unpack the panel into Z, Y
        Z = P[:, 3:]
        Y = P[:, 2]

        # Compute realized factor returns
        Numer = self.Gamma_Est.T.dot(Z.T).dot(Y)
        Denom = self.Gamma_Est.T.dot(Z.T).dot(Z).dot(self.Gamma_Est)
        Factor_OOS = np.linalg.solve(Denom, Numer.reshape((-1, 1)))

        if mean_factor:
            Ypred = np.squeeze(Z.dot(self.Gamma_Est)\
                    .dot(np.mean(self.Factors_Est, axis=1).reshape((-1, 1))))
        else:
            Ypred = np.diag(Z.dot(self.Gamma_Est).dot(Factor_OOS))

        return Ypred

    def _fit_ipca(self, X=None, W=None, val_obs=None, PSF=None, quiet=False):
        """
        Fits the regressor to the data using an alternating least squares
        scheme.

        Parameters
        ----------
        X : array-like of shape (n_characts,n_time),
            i.e. characteristics weighted portfolios

        W : array_like of shape (n_characts, n_characts,n_time),

        nan_mask : array, boolean
            The value at nan_mask[n,t] is True if no nan values are contained
            in Z[n,:,t] or Y[n,t] and False otherwise.

        PSF : optional, array-like of shape (n_PSF, n_time), i.e.
            pre-specified factors

        quiet   : optional, bool
            If true no text output will be produced


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

        # Initialize the Alternating Least Squares Procedure
        Gamma_Old, s, v = np.linalg.svd(X)
        Gamma_Old = Gamma_Old[:, :self.n_factors_eff]
        s = s[:self.n_factors_eff]
        v = v.T
        v = v[:, :self.n_factors_eff]
        Factor_Old = np.diag(s).dot(v.T)

        # Estimation Step
        tol_current = 1

        iter = 0

        while((iter <= self.max_iter) and (tol_current > self.iter_tol)):

            if self.PSFcase:
                Gamma_New, Factor_New = self._ALS_fit(Gamma_Old, W, X,
                                                      val_obs, PSF=PSF)
                tol_current = np.max(np.abs(Gamma_New.reshape((-1, 1))
                                      - Gamma_Old.reshape((-1, 1))))
            else:
                Gamma_New, Factor_New = self._ALS_fit(Gamma_Old, W, X,
                                                      val_obs)
                tol_current = np.max(np.abs(np.vstack((Gamma_New.reshape((-1, 1))
                                      - Gamma_Old.reshape((-1, 1)),
                                      Factor_New.reshape((-1, 1))
                                      - Factor_Old.reshape((-1, 1))))))

            # Compute update size
            Factor_Old = Factor_New
            Gamma_Old = Gamma_New


            iter += 1
            if not quiet:
                print('Step', iter, '- Aggregate Update:', tol_current)

        if not quiet:
            print('-- Convergence Reached --')

        return Gamma_New, Factor_New


    def _ALS_fit(self, Gamma_Old, W, X, val_obs, **kwargs):
        """The alternating least squares procedure switches back and forth
        between evaluating the first order conditions for Gamma_Beta, and the
        factors until convergence is reached. This function carries out one
        complete update procedure and will need to be called repeatedly using
        the updated Gamma's and factors as inputs.
        """
        T = self.T

        # Determine whether any per-specified factors were passed
        if self.PSFcase:
            PSF = kwargs.get("PSF")
            K_PSF, _ = np.shape(PSF)

        # Determine number of factors to be estimated
        if self.PSFcase:
            L, Ktilde = np.shape(Gamma_Old)
            K = Ktilde - K_PSF
        else:
            L, K = np.shape(Gamma_Old)
            Ktilde = K

        # ALS Step 1
        F_New = np.nan
        if K > 0:
            if self.PSFcase:
                F_New = np.full((K, T), np.nan)
                for t in range(T):
                    m1 = Gamma_Old[:, :K].T.dot(W[:, :, t])\
                        .dot(Gamma_Old[:, :K])
                    m2 = Gamma_Old[:, :K].T.dot(X[:, t])-Gamma_Old[:, :K].T\
                        .dot(W[:, :, t]).dot(Gamma_Old[:, K:Ktilde])\
                        .dot(PSF[:, t])
                    F_New[:, t] = np.squeeze(self._numba_solve(
                                                    m1, m2.reshape((-1, 1))))
            else:
                F_New = np.full((K, T), np.nan)
                for t in range(T):
                    m1 = Gamma_Old.T.dot(W[:, :, t]).dot(Gamma_Old)
                    m2 = Gamma_Old.T.dot(X[:, t])
                    F_New[:, t] = np.squeeze(self._numba_solve(
                                                    m1, m2.reshape((-1, 1))))
        else:
            F_New = np.full((K, T), np.nan)


        # ALS Step 2
        Numer = self._numba_full((L*Ktilde, 1), 0.0)
        Denom = self._numba_full((L*Ktilde, L*Ktilde), 0.0)
        #t1 = time.time()
        if self.PSFcase:
            if K > 0:
                for t in range(T):
                    Numer += self._numba_kron(X[:, t].reshape((-1, 1)),
                                            np.vstack(
                                            (F_New[:, t].reshape((-1, 1)),
                                             PSF[:, t].reshape((-1, 1)))))\
                                             * val_obs[t]
                    Denom_temp = np.vstack((F_New[:, t].reshape((-1, 1)),
                                           PSF[:, t].reshape((-1, 1))))
                    Denom += self._numba_kron(W[:, :, t], Denom_temp.dot(Denom_temp.T)
                                              * val_obs[t])
            else:
                for t in range(T):
                    Numer += self._numba_kron(X[:, t].reshape((-1, 1)),
                                              PSF[:, t].reshape((-1, 1)))\
                                              * val_obs[t]
                    Denom += self._numba_kron(W[:, :, t],
                                              PSF[:, t].reshape((-1, 1))
                                              .dot(PSF[:, t].T)) * val_obs[t]
        else:
            for t in range(T):

                Numer = Numer + self._numba_kron(X[:, t].reshape((-1, 1)),
                                          F_New[:, t].reshape((-1, 1)))\
                                          * val_obs[t]
                Denom = Denom + self._numba_kron(W[:, :, t],
                                          F_New[:, t].reshape((-1, 1))
                                          .dot(F_New[:, t].reshape((1, -1)))) \
                                          * val_obs[t]

        Gamma_New = self._numba_solve(Denom, Numer).reshape((L, Ktilde))
        #print('ALS2', time.time()-t1)

        # Enforce Orthogonality of Gamma_Beta and factors F
        if K > 0:
            R1 = self._numba_chol(Gamma_New[:, :K].T.dot(Gamma_New[:, :K])).T
            R2, _, _ = self._numba_svd(R1.dot(F_New).dot(F_New.T).dot(R1.T))
            Gamma_New[:, :K] = self._numba_lstsq(Gamma_New[:, :K].T,
                                                 R1.T)[0]\
                .dot(R2)
            F_New = self._numba_solve(R2, R1.dot(F_New))

        # Enforce sign convention for Gamma_Beta and F_New
        if K > 0:
            sg = np.sign(np.mean(F_New, axis=1)).reshape((-1, 1))
            sg[sg == 0] = 1
            Gamma_New[:, :K] = np.multiply(Gamma_New[:, :K], sg.T)
            F_New = np.multiply(F_New, sg)

        return Gamma_New, F_New

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
        P: array-like, tensor of dimensions (N, L, T), containing
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

        bar = progressbar.ProgressBar(maxval=N,
                                      widgets=[progressbar.Bar('=', '[', ']'),
                                               ' ', progressbar.Percentage()])

        bar.start()
        temp = []
        for n_i, n in enumerate(ids):
            ixd = np.isin(dates, P[P[:, 0] == 1, 1])
            temp_n = np.full((T, L+3), np.nan)
            temp_n[:, 0] = n
            temp_n[:, 1] = dates
            temp_n[ixd, :] = P[P[:, 0] == n, :]
            if np.size(temp_n, axis=0) == 0:
                continue
            temp.append(temp_n)
            bar.update(n_i)
        bar.finish()

        # Append the missing observations to create balanced panel
        if len(temp) > 0:
            P = np.concatenate(temp, axis=0)
        temp = []
        Y = np.reshape(P[:, 2], (N, T))
        nan_mask = ~np.any(np.isnan(P), axis=1)
        nan_mask = np.reshape(nan_mask, (N, T))
        # Reshape the panel into P (N, L, T) and Y(N, T)
        P = np.dstack(np.split(P[:, 3:], N, axis=0))
        P = np.swapaxes(P, 0, 2)

        self.ids = ids
        self.dates = dates
        print("Done!")
        return P, Y, nan_mask

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
        X: array-like
            matrix of dimensions (L, T), containing the characteristics
            weighted portfolios

        W: array-like
            matrix of dimension (L, L, T)

        val_obs: array-like
            matrix of dimension (T), containting the number of non missing
            observations at each point in time
        """

        dates = np.unique(P[:, 1])
        ids = np.unique(P[:, 0])
        T = np.size(dates, axis=0)
        N = np.size(ids, axis=0)
        L = np.size(P, axis=1) - 3
        print('The panel dimensions are:')
        print('n_samples:', N, ', n_characts:', L, ', n_time:', T)

        bar = progressbar.ProgressBar(maxval=T,
                                      widgets=[progressbar.Bar('=', '[', ']'),
                                               ' ', progressbar.Percentage()])
        bar.start()
        X = np.full((L, T), np.nan)
        W = np.full((L, L, T), np.nan)
        val_obs = np.full((T), np.nan)
        for t_i, t in enumerate(dates):
            ixt = (P[:, 1] == t)
            val_obs[t_i] = np.sum(ixt)
            # Define characteristics weighted matrices
            X[:, t_i] = np.transpose(P[ixt, 3:]).dot(P[ixt, 2])/val_obs[t_i]
            W[:, :, t_i] = np.transpose(P[ixt, 3:]).dot(P[ixt, 3:])/val_obs[t_i]
            bar.update(t_i)
        bar.finish()

        # Store panel dimensions
        self.ids = ids
        self.dates = dates
        self.T = T
        self.N = N
        self.L = L

        return X, W, val_obs

    def _R2_comps(self, P=None):
        """
        Computes the goodness of fit measures both at the entity level
        and at the managed portfolio level. Requires the estimator to be
        fitted previously.

        Parameters
        ----------
        P   :   Panel of stacked data. Each row corresponds to an observation
                (i, t) where i denotes the entity index and t denotes
                the time index. The panel may be unbalanced. The number of unique
                entities is n_samples, the number of unique dates is n_time, and
                the number of characteristics used as instruments is n_characts.
                The columns of the panel are organized in the following order:

                - Column 1: entity id (i)
                - Column 2: time index (t)
                - Column 3: dependent variable corresponding to observation (i,t)
                - Column 4 to column 4+n_characts: characteristics.

        """

        # Compute goodness of fit measures, entity level
        Ytrue = P[:, 2]

        # R2 Total
        Ypred = self.predict(np.delete(P, 2, axis=1), mean_factor=False)
        r2_total = 1-np.nansum((Ypred-Ytrue)**2)/np.nansum(Ytrue**2)

        # R2 Pred
        Ypred = self.predict(np.delete(P, 2, axis=1), mean_factor=True)
        r2_pred = 1-np.nansum((Ypred-Ytrue)**2)/np.nansum(Ytrue**2)

        # Compute goodness of fit measures, portfolio level
        Num_tot = 0
        Denom_tot = 0
        Num_pred = 0
        Denom_pred = 0

        mean_Factors_Est = np.mean(self.Factors_Est, axis=1).reshape((-1, 1))

        for t_i, t in enumerate(self.dates):
            Ytrue = self.X[:, t_i]
            # R2 Total
            Ypred = self.W[:, :, t_i].dot(self.Gamma_Est)\
                .dot(self.Factors_Est[:, t_i])
            Num_tot += np.transpose((Ytrue-Ypred)).dot((Ytrue-Ypred))
            Denom_tot += Ytrue.T.dot(Ytrue)

            # R2 Pred
            Ypred = self.W[:, :, t_i].dot(self.Gamma_Est).dot(mean_Factors_Est)
            Ypred = np.squeeze(Ypred)
            Num_pred += np.transpose((Ytrue-Ypred)).dot((Ytrue-Ypred))
            Denom_pred += Ytrue.T.dot(Ytrue)

        r2_total_x = 1-Num_tot/Denom_tot
        r2_pred_x = 1-Num_pred/Denom_pred

        return r2_total, r2_pred, r2_total_x, r2_pred_x

    @staticmethod
    @jit(nopython=True)
    def _numba_solve(m1, m2):
        return np.linalg.solve(np.ascontiguousarray(m1), np.ascontiguousarray(m2))

    @staticmethod
    @jit(nopython=True)
    def _numba_lstsq(m1, m2):
        return np.linalg.lstsq(np.ascontiguousarray(m1), np.ascontiguousarray(m2))

    @staticmethod
    @jit(nopython=True)
    def _numba_kron(m1, m2):
        return np.kron(np.ascontiguousarray(m1), np.ascontiguousarray(m2))

    @staticmethod
    @jit(nopython=True)
    def _numba_chol(m1):
        return np.linalg.cholesky(np.ascontiguousarray(m1))

    @staticmethod
    @jit(nopython=True)
    def _numba_svd(m1):
        return np.linalg.svd(np.ascontiguousarray(m1))

    @staticmethod
    @jit(nopython=True)
    def _numba_full(m1, m2):
        return np.full(m1, m2)


def _BS_Walpha_sub(self, n, d):
    X_b = np.full((self.L, self.T), np.nan)
    np.random.seed(n)
    for t in range(self.T):
        d_temp = np.random.standard_t(5)*d[:, np.random.randint(0, high=self.T)]
        X_b[:, t] = self.W[:, :, t].dot(self.Gamma_Est[:, :-1])\
            .dot(self.Factors_Est[:-1, t]) + d_temp

    # Re-estimate unrestricted model
    Gamma, Factors = self._fit_ipca(X=X_b, W=self.W, PSF=self.PSF,
                                    val_obs=self.val_obs, quiet=True)

    # Compute and store Walpha_b
    Walpha_b = Gamma[-1, :].T.dot(Gamma[-1, :])

    return Walpha_b
