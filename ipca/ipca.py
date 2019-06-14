from sklearn.linear_model import ElasticNet
from joblib import Parallel, delayed
from numba import jit
import numpy as np
import scipy as sp
import progressbar
import warnings
import time

class IPCARegressor:
    """
    This class implements the IPCA algorithm by Kelly, Pruitt, Su (2017).

    Parameters
    ----------

    n_factors : int, default=1
        The total number of factors to estimate. Note, the number of
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
        Tolerance threshold for stopping the alternating least squares
        procedure
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


    def fit(self, Panel=None, PSF=None, refit=False, alpha=0., l1_ratio=1.,
            **kwargs):
        """
        Fits the regressor to the data using an alternating least squares
        scheme.

        Parameters
        ----------
        Panel :  numpy array
            Panel of stacked data. Each row corresponds to an observation
            (i, t) where i denotes the entity index and t denotes
            the time index. The panel may be unbalanced. The number of unique
            entities is n_samples, the number of unique dates is T, and
            the number of characteristics used as instruments is L.
            The columns of the panel are organized in the following order:

            - Column 1: entity id (i)
            - Column 2: time index (t)
            - Column 3: dependent variable corresponding to observation (i,t)
            - Column 4 to column 4+L: characteristics.

        PSF : numpy array, optional
            Set of pre-specified factors as matrix of dimension (M, T)

        refit : boolean, optional
            Indicates whether the regressor should be re-fit. If set to True
            the function will skip unpacking the panel into a tensor and
            instead use the stored values from the previous fit. Note, it is
            still necessary to pass the previously used P.

        alpha : scalar
            Regularizing constant for Gamma estimation.  If this is set to
            zero then the estimation defaults to non-regularized.

        l1_ratio : scalar
            Ratio of l1 and l2 penalties for elastic net Gamma fit.

        Returns
        -------

        Gamma : numpy array
            Array with dimensions (L, n_factors) containing the
            mapping between characteristics and factors loadings. If there
            are M many pre-specified factors in the model then the
            matrix returned is of dimension (L, (n_factors+M)).
            If an intercept is included in the model, its loadings are
            returned in the last column of Gamma.

        Factors : numpy array
            Array with dimensions (n_factors, T) containing the estimated
            factors. If pre-specified factors were passed the returned
            array is of dimension ((n_factors - M), T),
            corresponding to the n_factors - M many factors estimated on
            top of the pre-specified ones.

        """
        # Check re-fitting is valid
        if refit:
            try:
                self.X
            except AttributeError:
                raise ValueError('Refit only possible after initial fit.')

        # Check panel input
        if Panel is None:
            raise ValueError('Must pass panel input data.')
        else:
            # remove panel rows containing missing obs
            Panel = Panel[~np.any(np.isnan(Panel), axis=1)]

        # Unpack the Panel
        if not refit:
            X, W, val_obs = self._unpack_panel(Panel)
        else:
            Panel, X, W, val_obs = self.Panel, self.X, self.W, self.val_obs

        # Handle pre-specified factors
        if PSF is not None:
            if np.size(PSF, axis=1) != np.size(np.unique(Panel[:, 1])):
                raise ValueError("""Number of PSF observations must match
                                 number of unique dates in panel P""")
            self.has_PSF = True
        else:
            self.has_PSF = False

        if self.has_PSF:
            if np.size(PSF, axis=0) == self.n_factors:
                print("""Note: The number of factors (n_factors) to be
                      estimated matches the number of
                      pre-specified factors. No additional factors
                      will be estimated. To estimate additional
                      factors increase n_factors.""")

        #  Treating intercept as if was a prespecified factor
        if self.intercept:
            self.n_factors_eff = self.n_factors + 1
            if PSF is not None:
                PSF = np.concatenate((PSF, np.ones((1, self.T))), axis=0)
            elif PSF is None:
                PSF = np.ones((1, self.T))
        else:
            self.n_factors_eff = self.n_factors

        # Check that enough features provided
        if np.size(Panel, axis=1) - 3 < self.n_factors_eff:
            raise ValueError("""The number of factors requested exceeds number
                              of features""")

        # Determine fit case - if intercept or PSF or both use PSFcase fitting
        # Note PSFcase in contrast to has_PSF is only indicating
        # that the IPCA fitting is carried out as if PSF were passed even if
        # only an intercept was passed.
        self.PSFcase = True if self.has_PSF or self.intercept else False

        # Run IPCA
        Gamma, Factors = self._fit_ipca(X, W, val_obs, Panel=Panel, PSF=PSF,
                                        alpha=alpha, l1_ratio=l1_ratio,
                                        **kwargs)

        # Store estimates
        if self.PSFcase:
            if self.intercept and self.has_PSF:
                PSF = np.concatenate((PSF, np.ones((1, len(self.dates)))),
                                     axis=0)
            elif self.intercept:
                PSF = np.ones((1, len(self.dates)))
            if Factors is not None:
                Factors = np.concatenate((Factors, PSF), axis=0)
            else:
                Factors = PSF

        self.Gamma_Est, self.Factors_Est = Gamma, Factors

        # Save unpacked panel for Re-fitting
        if not refit:
            self.Panel = Panel
            self.PSF = PSF
            self.X = X
            self.W = W
            self.val_obs = val_obs

        # Compute Goodness of Fit
        self.r2_total, self.r2_pred, self.r2_total_x, self.r2_pred_x = \
            self._R2_comps(Panel=Panel)

        return self.Gamma_Est, self.Factors_Est


    def predict(self, Panel=None, mean_factor=False):
        """
        Predicts fitted values for a previously fitted regressor

        Parameters
        ----------
        Panel :  numpy array
            Panel of stacked data. Each row corresponds to an observation
            (i, t) where i denotes the entity index and t denotes
            the time index. The panel may be unbalanced. If an observation
            contains missing data NaN will be returned. Note that the
            number of passed characteristics L must match the
            number of characteristics used when fitting the regressor.
            The columns of the panel are organized in the following order:

            - Column 1: entity id (i)
            - Column 2: time index (t)
            - Column 3 to column 3+L: characteristics.

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

        if Panel is None:
            raise ValueError("""A panel of characteristics data must be
                              provided.""")

        if np.any(np.isnan(Panel)):
            raise ValueError("""Cannot contain missing observations / nan
                              values.""")
        N = np.size(Panel, axis=0)
        Ypred = np.full((N), np.nan)

        mean_Factors_Est = np.mean(self.Factors_Est, axis=1).reshape((-1, 1))

        if mean_factor:
            Ypred[:] = np.squeeze(Panel[:, 2:].dot(self.Gamma_Est)\
                .dot(mean_Factors_Est))
        else:

            for t_i, t in enumerate(self.dates):
                ix = (Panel[:, 1] == t)
                Ypred[ix] = np.squeeze(Panel[ix, 2:].dot(self.Gamma_Est)\
                    .dot(self.Factors_Est[:, t_i]))
        return Ypred


    def BS_Walpha(self, ndraws=1000, n_jobs=1, backend='loky'):
        """
        Bootstrap inference on the hypothesis Gamma_alpha = 0

        Parameters
        ----------

        ndraws  : integer, default=1000
            Number of bootstrap draws and re-estimations to be performed

        backend : optional
            Value is either 'loky' or 'multiprocessing'

        n_jobs  : integer
            Number of workers to be used. If -1, all available workers are
            used.

        Returns
        -------

        pval : float
            P-value from the hypothesis test H0: Gamma_alpha=0
        """

        if not self.intercept:
            raise ValueError('Need to fit model with intercept first.')

        # Compute Walpha
        Walpha = self.Gamma_Est[:, -1].T.dot(self.Gamma_Est[:, -1])

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


    def BS_Wbeta(self, l, ndraws=1000, n_jobs=1, backend='loky'):
        """
        Test of instrument significance.
        Bootstrap inference on the hypothesis  l-th column of Gamma_beta = 0.

        Parameters
        ----------

        l   : integer
            Position of the characteristics for which the bootstrap is to be
            carried out. For example, if there are 10 characteristics, l is in
            the range 0 to 9 (left-/right-inclusive).

        ndraws  : integer, default=1000
            Number of bootstrap draws and re-estimations to be performed

        n_jobs  : integer
            Number of workers to be used for multiprocessing.
            If -1, all available Workers are used.

        backend : optional

        Returns
        -------

        pval : float
            P-value from the hypothesis test H0: Gamma_alpha=0
        """

        if self.PSFcase:
            raise ValueError('Need to fit model without intercept first.')

        # Compute Wbeta_l if l-th characteristics is set to zero
        Wbeta_l = self.Gamma_Est[l, :].dot(self.Gamma_Est[l, :].T)
        Wbeta_l = np.trace(Wbeta_l)
        # Compute residuals
        d = np.full((self.L, self.T), np.nan)
        for t_i, t in enumerate(self.dates):
            d[:, t_i] = self.X[:, t_i]-self.W[:, :, t_i].dot(self.Gamma_Est)\
                .dot(self.Factors_Est[:, t_i])

        print("Starting Bootstrap...")
        Wbeta_l_b = Parallel(n_jobs=n_jobs, backend=backend, verbose=10)(
            delayed(_BS_Wbeta_sub)(self, n, d, l) for n in range(ndraws))
        print("Done!")

        pval = np.sum(Wbeta_l_b > Wbeta_l)/ndraws
        # print(Wbeta_l_b, Wbeta_l)

        return pval


    def predictOOS(self, Panel=None, mean_factor=False):
        """
        Predicts time t+1 observation using an out-of-sample design.

        Parameters
        ----------
        Panel :  numpy array
            Panel of stacked data. Each row corresponds to an observation
            (i,t) where i denotes the entity index and t denotes
            the time index. All data must correspond to time t, i.e. all
            observations occur on the same date.
            If an observation contains missing data NaN will be returned.
            Note that the number of characteristics (L) passed,
            has to match the number of characteristics used when fitting
            the regressor.
            The columns of the panel are organized in the following order:

            - Column 1: entity id (i)
            - Column 2: time index (t)
            - Column 3: dependent variable corresponding to observation (i,t)
            - Column 4 to column 4+L: characteristics.

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

        if Panel is None:
            raise ValueError("""A panel of characteristics data must be
                              provided.""")

        if len(np.unique(Panel[:, 1])) > 1:
            raise ValueError('The panel must only have a single timestamp.')

        N = np.size(Panel, axis=0)
        Ypred = np.full((N), np.nan)

        # Unpack the panel into Z, Y
        Z, Y = Panel[:, 3:], Panel[:, 2]

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


    def _unpack_panel(self, Panel):
        """ Converts a stacked panel of data where each row corresponds to an
        observation (i, t) into a tensor of dimensions (N, L, T) where N is the
        number of unique entities, L is the number of characteristics and T is
        the number of unique dates

        Parameters
        ----------

        Panel : Panel of data. Each row corresponds to an observation (i, t).
                The columns are ordered in the following manner:
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

        dates = np.unique(Panel[:, 1])
        ids = np.unique(Panel[:, 0])
        T = np.size(dates, axis=0)
        N = np.size(ids, axis=0)
        L = np.size(Panel, axis=1) - 3
        print('The panel dimensions are:')
        print('n_samples:', N, ', L:', L, ', T:', T)

        bar = progressbar.ProgressBar(maxval=T,
                                      widgets=[progressbar.Bar('=', '[', ']'),
                                               ' ', progressbar.Percentage()])
        bar.start()
        X = np.full((L, T), np.nan)
        W = np.full((L, L, T), np.nan)
        val_obs = np.full((T), np.nan)
        for t_i, t in enumerate(dates):
            ixt = (Panel[:, 1] == t)
            val_obs[t_i] = np.sum(ixt)
            # Define characteristics weighted matrices
            X[:, t_i] = Panel[ixt, 3:].T.dot(Panel[ixt, 2])/val_obs[t_i]
            W[:, :, t_i] = Panel[ixt, 3:].T.dot(Panel[ixt, 3:])/val_obs[t_i]
            bar.update(t_i)
        bar.finish()

        # Store panel dimensions
        self.ids, self.dates, self.T, self.N, self.L = ids, dates, T, N, L

        return X, W, val_obs


    def _fit_ipca(self, X, W, val_obs, Panel=None, PSF=None, quiet=False,
                  **kwargs):
        """
        Fits the regressor to the data using alternating least squares

        Parameters
        ----------
        X : array-like of shape (L,T),
            i.e. characteristics weighted portfolios

        W : array_like of shape (L, L,T),


        val_obs: array-like
            matrix of dimension (T), containting the number of non missing
            observations at each point in time

        Panel : optional, Panel of data.

                Each row corresponds to an observation (i, t).
                The columns are ordered in the following manner:
                COLUMN 1: entity id (i)
                COLUMN 2: time index (t)
                COLUMN 3: depdent variable Y(i,t)
                COLUMN 4 and following: L characteristics

        PSF : optional, array-like of shape (M, T), i.e.
            pre-specified factors

        quiet   : optional, bool
            If true no text output will be produced

        Returns
        -------
        Gamma : array-like with dimensions (L, n_factors). If there
            are n_prespec many pre-specified factors in the model then the
            matrix returned is of dimension (L, (n_factors+M)).
            If an intercept is included in the model, its loadings are returned
            in the last column of Gamma.

        Factors : array_like with dimensions (n_factors, T). If
            pre-specified factors were passed the returned matrix is
            of dimension ((n_factors - M), T), corresponding to the
            n_factors - M many factors estimated on top of the pre-
            specified ones.
        """

        # Initialize the Alternating Least Squares Procedure
        Gamma_Old, s, v = np.linalg.svd(X)
        Gamma_Old = Gamma_Old[:, :self.n_factors_eff]
        s = s[:self.n_factors_eff]
        v = v[:self.n_factors_eff, :]
        Factor_Old = np.diag(s).dot(v)

        # Estimation Step
        tol_current = 1

        iter = 0

        while((iter <= self.max_iter) and (tol_current > self.iter_tol)):

            Gamma_New, Factor_New = self._ALS_fit(Gamma_Old, W, X, val_obs,
                                                  Panel=Panel, PSF=PSF,
                                                  **kwargs)
            if self.PSFcase:
                tol_current = np.max(np.abs(Gamma_New - Gamma_Old))
            else:
                tol_current_G = np.max(np.abs(Gamma_New - Gamma_Old))
                tol_current_F = np.max(np.abs(Factor_New - Factor_Old))
                tol_current = max(tol_current_G, tol_current_F)

            # Update factors and loadings
            Factor_Old, Gamma_Old = Factor_New, Gamma_New

            iter += 1
            if not quiet:
                print('Step', iter, '- Aggregate Update:', tol_current)

        if not quiet:
            print('-- Convergence Reached --')

        return Gamma_New, Factor_New


    def _ALS_fit(self, Gamma_Old, W, X, val_obs, Panel=None, PSF=None,
                 n_jobs=1, backend="loky", alpha=0., l1_ratio=1., **kwargs):
        """Alternating least squares procedure to fit params

        The alternating least squares procedure switches back and forth
        between evaluating the first order conditions for Gamma_Beta, and the
        factors until convergence is reached. This function carries out one
        complete update procedure and will need to be called repeatedly using
        the updated Gamma's and factors as inputs.
        """

        T = self.T

        if PSF is None:
            L, K = np.shape(Gamma_Old)
            Ktilde = K
        else:
            L, Ktilde = np.shape(Gamma_Old)
            K_PSF, _ = np.shape(PSF)
            K = Ktilde - K_PSF

        # ALS Step 1
        if K > 0:

            # case with no observed factors
            if PSF is None:
                if n_jobs > 1:
                    F_New = Parallel(n_jobs=n_jobs, backend=backend)(
                                delayed(_Ft_fit)(
                                    Gamma_Old, W[:,:,t], X[:,t])
                                for t in range(T))
                    F_New = np.stack(F_New, axis=1)

                else:
                    F_New = np.full((K, T), np.nan)
                    for t in range(T):
                        F_New[:,t] = _Ft_fit(Gamma_Old, W[:,:,t], X[:,t])

            # observed factors+latent factors case
            else:
                if n_jobs > 1:
                    F_New = Parallel(n_jobs=n_jobs, backend=backend)(
                                delayed(_Ft_PSF_fit)(
                                    Gamma_Old, W[:,:,t], X[:,t], PSF[:,t],
                                    K, Ktilde)
                                for t in range(T))
                    F_New = np.stack(F_New, axis=1)

                else:
                    F_New = np.full((K, T), np.nan)
                    for t in range(T):
                        F_New[:,t] = _Ft_PSF_fit(Gamma_Old, W[:,:,t], X[:,t],
                                                 PSF[:,t], K, Ktilde)

        else:
            F_New = None

        # ALS Step 2

        if Panel is None:
            Gamma_New = _Gamma_portfolio_fit(F_New, X, W, val_obs, PSF, L, K,
                                             Ktilde, T)
        else:
            Gamma_New = _Gamma_panel_fit(F_New, Panel, PSF, L, Ktilde, alpha,
                                         l1_ratio, **kwargs)

        # condition checks

        # Enforce Orthogonality of Gamma_Beta and factors F
        if K > 0:
            R1 = _numba_chol(Gamma_New[:, :K].T.dot(Gamma_New[:, :K])).T
            R2, _, _ = _numba_svd(R1.dot(F_New).dot(F_New.T).dot(R1.T))
            Gamma_New[:, :K] = _numba_lstsq(Gamma_New[:, :K].T,
                                            R1.T)[0].dot(R2)
            F_New = _numba_solve(R2, R1.dot(F_New))

        # Enforce sign convention for Gamma_Beta and F_New
        if K > 0:
            sg = np.sign(np.mean(F_New, axis=1)).reshape((-1, 1))
            sg[sg == 0] = 1
            Gamma_New[:, :K] = np.multiply(Gamma_New[:, :K], sg.T)
            F_New = np.multiply(F_New, sg)

        return Gamma_New, F_New


    def _R2_comps(self, Panel=None):
        """
        Computes the goodness of fit measures both at the entity level
        and at the managed portfolio level. Requires the estimator to be
        fitted previously.

        Parameters
        ----------
        Panel   :   Panel of stacked data. Each row corresponds to an
                    observation (i, t) where i denotes the entity index and t
                    denotes the time index. The panel may be unbalanced. The
                    number of unique entities is n_samples, the number of
                    unique dates is T, and the number of characteristics used
                    as instruments is L. The columns of the panel are
                    organized in the following order:

                - Column 1: entity id (i)
                - Column 2: time index (t)
                - Column 3: dependent variable corresponding to observation
                            (i,t)
                - Column 4 to column 4+L: characteristics.

        """

        # Compute goodness of fit measures, entity level
        Ytrue = Panel[:, 2]

        # R2 Total
        Ypred = self.predict(np.delete(Panel, 2, axis=1), mean_factor=False)
        r2_total = 1-np.nansum((Ypred-Ytrue)**2)/np.nansum(Ytrue**2)

        # R2 Pred
        Ypred = self.predict(np.delete(Panel, 2, axis=1), mean_factor=True)
        r2_pred = 1-np.nansum((Ypred-Ytrue)**2)/np.nansum(Ytrue**2)

        # Compute goodness of fit measures, portfolio level
        Num_tot, Denom_tot, Num_pred, Denom_pred = 0, 0, 0, 0

        mean_Factors_Est = np.mean(self.Factors_Est, axis=1).reshape((-1, 1))

        for t_i, t in enumerate(self.dates):
            Ytrue = self.X[:, t_i]
            # R2 Total
            Ypred = self.W[:, :, t_i].dot(self.Gamma_Est)\
                .dot(self.Factors_Est[:, t_i])
            Num_tot += (Ytrue-Ypred).T.dot((Ytrue-Ypred))
            Denom_tot += Ytrue.T.dot(Ytrue)

            # R2 Pred
            Ypred = self.W[:, :, t_i].dot(self.Gamma_Est).dot(mean_Factors_Est)
            Ypred = np.squeeze(Ypred)
            Num_pred += (Ytrue-Ypred).T.dot((Ytrue-Ypred))
            Denom_pred += Ytrue.T.dot(Ytrue)

        r2_total_x = 1-Num_tot/Denom_tot
        r2_pred_x = 1-Num_pred/Denom_pred

        return r2_total, r2_pred, r2_total_x, r2_pred_x



def _Ft_fit(Gamma_Old, W_t, X_t):
    """helper func to parallelize F ALS fit"""

    m1 = Gamma_Old.T.dot(W_t).dot(Gamma_Old)
    m2 = Gamma_Old.T.dot(X_t)

    return np.squeeze(_numba_solve(m1, m2.reshape((-1, 1))))


def _Ft_PSF_fit(Gamma_Old, W_t, X_t, PSF_t, K, Ktilde):
    """helper func to parallelize F ALS fit with observed factors"""

    m1 = Gamma_Old[:,:K].T.dot(W_t).dot(Gamma_Old[:,:K])
    m2 = Gamma_Old[:,:K].T.dot(X_t)
    m2 -= Gamma_Old[:,:K].T.dot(W_t).dot(Gamma_Old[:,K:Ktilde]).dot(PSF_t)

    return np.squeeze(_numba_solve(m1, m2.reshape((-1, 1))))


def _Gamma_portfolio_fit(F_New, X, W, val_obs, PSF, L, K, Ktilde, T):
    """helper function for fitting gamma without panel"""

    Numer = _numba_full((L*Ktilde, 1), 0.0)
    Denom = _numba_full((L*Ktilde, L*Ktilde), 0.0)

    # no observed factors
    if PSF is None:
        for t in range(T):

            Numer += _numba_kron(X[:, t].reshape((-1, 1)),
                                 F_New[:, t].reshape((-1, 1)))\
                                 * val_obs[t]
            Denom += _numba_kron(W[:, :, t],
                                 F_New[:, t].reshape((-1, 1))
                                 .dot(F_New[:, t].reshape((1, -1)))) \
                                 * val_obs[t]

    # observed+latent factors
    elif K > 0:
        for t in range(T):
            Numer += _numba_kron(X[:, t].reshape((-1, 1)),
                                 np.vstack(
                                 (F_New[:, t].reshape((-1, 1)),
                                 PSF[:, t].reshape((-1, 1)))))\
                                 * val_obs[t]
            Denom_temp = np.vstack((F_New[:, t].reshape((-1, 1)),
                                   PSF[:, t].reshape((-1, 1))))
            Denom += _numba_kron(W[:, :, t], Denom_temp.dot(Denom_temp.T)
                                 * val_obs[t])

    # only observed factors
    else:
        for t in range(T):
            Numer += _numba_kron(X[:, t].reshape((-1, 1)),
                                 PSF[:, t].reshape((-1, 1)))\
                                 * val_obs[t]
            Denom += _numba_kron(W[:, :, t],
                                 PSF[:, t].reshape((-1, 1))
                                 .dot(PSF[:, t].reshape((-1, 1)).T))\
                                 * val_obs[t]

    Gamma_New = _numba_solve(Denom, Numer).reshape((L, Ktilde))

    return Gamma_New


def _Gamma_panel_fit(F_New, Panel, PSF, L, Ktilde, alpha, l1_ratio, **kwargs):
    """helper function for estimating vectorized Gamma with panel"""

    # join observed factors with latent factors and map to panel
    if PSF is None:
        F = F_New
    else:
        if F_New is None:
            F = PSF
        else:
            F = np.vstack((F_New, PSF))
    F = F[:,np.unique(Panel[:,1], return_inverse=True)[1]]

    # interact factors and characteristics
    ZkF = np.hstack((F[k,:,None] * Panel[:,3:] for k in range(Ktilde)))

    # elastic net fit
    if alpha:
        mod = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **kwargs)
        mod.fit(ZkF, Panel[:,2])
        gamma = mod.coef_

    # OLS fit
    else:
        gamma = _numba_lstsq(ZkF, Panel[:,2])[0]

    gamma = gamma.reshape((Ktilde, L)).T

    return gamma


def _BS_Walpha_sub(model, n, d):
    X_b = np.full((model.L, model.T), np.nan)
    np.random.seed(n)


    # Re-estimate unrestricted model
    Gamma = None
    while Gamma is None:
        try:
            for t in range(model.T):
                d_temp = np.random.standard_t(5)*d[:,np.random.randint(0,high=model.T)]
                X_b[:, t] = model.W[:, :, t].dot(model.Gamma_Est[:, :-1])\
                    .dot(model.Factors_Est[:-1, t]) + d_temp
            Gamma, Factors = model._fit_ipca(X=X_b, W=model.W, val_obs=model.val_obs,
                                             PSF=model.PSF, quiet=True)
        except np.linalg.LinAlgError:
            warnings.warn("Encountered singularity in bootstrap iteration. Observation discarded.")
            pass


    # Compute and store Walpha_b
    Walpha_b = Gamma[:, -1].T.dot(Gamma[:, -1])

    return Walpha_b


def _BS_Wbeta_sub(model, n, d, l):
    X_b = np.full((model.L, model.T), np.nan)
    np.random.seed(n)
    #Modify Gamma_beta such that its l-th row is zero
    Gamma_beta_l = np.copy(model.Gamma_Est)
    Gamma_beta_l[l, :] = 0

    Gamma = None
    while Gamma is None:
        try:
            for t in range(model.T):
                d_temp = np.random.standard_t(5)*d[:,np.random.randint(0,high=model.T)]
                X_b[:, t] = model.W[:, :, t].dot(Gamma_beta_l)\
                    .dot(model.Factors_Est[:, t]) + d_temp

            # Re-estimate unrestricted model
            Gamma, Factors = model._fit_ipca(X=X_b, W=model.W, val_obs=model.val_obs,
                                             PSF=model.PSF, quiet=True)
        except np.linalg.LinAlgError:
            warnings.warn("Encountered singularity in bootstrap iteration. Observation discarded.")
            pass

    # Compute and store Walpha_b
    Wbeta_l_b = Gamma[l, :].dot(Gamma[l, :].T)
    Wbeta_l_b = np.trace(Wbeta_l_b)
    return Wbeta_l_b


@jit(nopython=True)
def _numba_solve(m1, m2):
    return np.linalg.solve(np.ascontiguousarray(m1), np.ascontiguousarray(m2))

@jit(nopython=True)
def _numba_lstsq(m1, m2):
    return np.linalg.lstsq(np.ascontiguousarray(m1), np.ascontiguousarray(m2))

@jit(nopython=True)
def _numba_kron(m1, m2):
    return np.kron(np.ascontiguousarray(m1), np.ascontiguousarray(m2))

@jit(nopython=True)
def _numba_chol(m1):
    return np.linalg.cholesky(np.ascontiguousarray(m1))

@jit(nopython=True)
def _numba_svd(m1):
    return np.linalg.svd(np.ascontiguousarray(m1))

@jit(nopython=True)
def _numba_full(m1, m2):
    return np.full(m1, m2)
