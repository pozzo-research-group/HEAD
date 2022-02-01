import fdasrsf as fs
import numpy as np
from numpy.linalg import norm, inv
from _DP import ffi, lib
import _DP
import numba
from numba.core.typing import cffi_utils
import fdasrsf.utility_functions as uf
from scipy.integrate import trapz, cumtrapz
from scipy.interpolate import interp1d, UnivariateSpline
from joblib import Parallel, delayed
import time
import fdasrsf.utility_functions as uf
import fdasrsf.geometry as geo
from scipy.integrate import trapz, cumtrapz
from scipy.interpolate import interp1d
from numpy.linalg import norm, inv

DP = lib.DP
cffi_utils.register_module(_DP)


@numba.jit()
def _grad(f, binsize):
    n = f.shape[0]
    g = np.zeros(n)
    h = binsize*np.arange(1,n+1)
    g[0] = (f[1] - f[0])/(h[1]-h[0])
    g[-1] = (f[-1] - f[(-2)])/(h[-1]-h[-2])

    h = h[2:]-h[0:-2]
    g[1:-1] = (f[2:]-f[0:-2])/h[0]

    return g

@numba.njit()
def _warp(q1, q2):
    M = q1.shape[0]
    disp = 0
    n1 = 1
    lam = 0.0
    gam = np.zeros(M)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    q1 = np.ascontiguousarray(q1)
    q2 = np.ascontiguousarray(q2)
    gam = np.ascontiguousarray(gam)
    q2i = ffi.from_buffer(q2)
    q1i = ffi.from_buffer(q1)
    gami = ffi.from_buffer(gam)
    DP(q2i,q1i,n1,M,lam,disp,gami)

    return gam

@numba.njit()
def _amplitude_distance(q1, q2):
    """"
    calculates the distances between two curves, where
    q2 is aligned to q1. In other words calculates the elastic distances/
    This metric is set up for use with UMAP or t-sne from scikit-learn
    :param q1: vector of size N
    :param q2: vector of size N
    :rtype: scalar
    :return dist: amplitude distance
    """
    tst = q1-q2
    if tst.sum() == 0:
        dist = 0
    else:
        q1 = q1.astype(np.double)
        q2 = q2.astype(np.double)
        gam = _warp(q1, q2)
        M = q1.shape[0]
        time = np.linspace(0,1,q1.shape[0])
        gam = (gam - gam[0]) / (gam[-1] - gam[0])
        gam_dev = _grad(gam, 1 / np.double(M - 1))
        tmp = np.interp((time[-1] - time[0]) * gam + time[0], time, q2)

        qw = tmp * np.sqrt(gam_dev)

        y = (qw - q1) ** 2
        tmp = np.diff(time)*(y[0:-1]+y[1:])/2
        dist = np.sqrt(tmp.sum())

    return dist

@numba.njit()
def _phase_distance(q1, q2):
    """"
    calculates the phase distances between two curves, where
    q2 is aligned to q1. In other words calculates the elastic distances/
    This metric is set up for use with UMAP or t-sne from scikit-learn
    :param q1: vector of size N
    :param q2: vector of size N
    :rtype: scalar
    :return dist: amplitude distance
    """
    tst = np.abs(q1-q2)
    if tst.sum()<1e-3:
        dist = 0
    else:
        q1 = q1.astype(np.double)
        q2 = q2.astype(np.double)
        gam = _warp(q1, q2)
        M = q1.shape[0]
        time = np.linspace(0,1,q1.shape[0])
        gam = (gam - gam[0]) / (gam[-1] - gam[0])
        gam_dev = _grad(gam, 1 / np.double(M - 1))
        theta = np.trapz(np.sqrt(gam_dev),x=time)
        if theta > 1:
            theta = 1
        elif theta < -1:
            theta = -1
        dist = np.arccos(theta)

    return dist

def AmplitudePhaseDistance(point_a, point_b, time):
    curves = np.zeros((len(point_a), 2))
    curves[...,0] = point_a.astype('double')
    curves[...,1] = point_b.astype('double')
    obj = fdawarp(curves.astype('double'), time.astype('double'))
    obj.srsf_align(parallel=True, MaxItr=50)
    dp = _phase_distance(obj.qn[...,0], obj.qn[...,1]) #.astype('double')
    da = _amplitude_distance(obj.qn[...,0], obj.qn[...,1]) #.astype('double')

    return da, dp


class fdawarp:
    """
    This class provides alignment methods for functional data using the SRVF framework
    Usage:  obj = fdawarp(f,t)

    :param f: (M,N): matrix defining N functions of M samples
    :param time: time vector of length M
    :param fn: aligned functions
    :param qn: aligned srvfs
    :param q0: initial srvfs
    :param fmean: function mean
    :param mqn: mean srvf
    :param gam: warping functions
    :param psi: srvf of warping functions
    :param stats: alignment statistics
    :param qun: cost function
    :param lambda: lambda
    :param method: optimization method
    :param gamI: inverse warping function
    :param rsamps: random samples
    :param fs: random aligned functions
    :param gams: random warping functions
    :param ft: random warped functions
    :param qs: random aligned srvfs
    :param type: alignment type
    :param mcmc: mcmc output if bayesian

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  15-Mar-2018
    """

    def __init__(self, f, time, verbose=False):
        """
        Construct an instance of the fdawarp class
        :param f: numpy ndarray of shape (M,N) of N functions with M samples
        :param time: vector of size M describing the sample points
        """
        a = time.shape[0]

        if f.shape[0] != a:
            raise Exception('Columns of f and time must be equal')

        self.f = f
        self.time = time
        self.rsamps = False
        self.verbose = verbose


    def srsf_align(self, method="mean", omethod="DP2", center=True,
                   smoothdata=False, MaxItr=20, parallel=False, lam=0.0,
                   cores=-1, grid_dim=7):
        """
        This function aligns a collection of functions using the elastic
        square-root slope (srsf) framework.
        :param method: (string) warp calculate Karcher Mean or Median
                       (options = "mean" or "median") (default="mean")
        :param omethod: optimization method (DP, DP2, RBFGS) (default = DP2)
        :param center: center warping functions (default = T)
        :param smoothdata: Smooth the data using a box filter (default = F)
        :param MaxItr: Maximum number of iterations (default = 20)
        :param parallel: run in parallel (default = F)
        :param lam: controls the elasticity (default = 0)
        :param cores: number of cores for parallel (default = -1 (all))
        :param grid_dim: size of the grid, for the DP2 method only (default = 7)
        :type lam: double
        :type smoothdata: bool
        Examples
        >>> import tables
        >>> fun=tables.open_file("../Data/simu_data.h5")
        >>> f = fun.root.f[:]
        >>> f = f.transpose()
        >>> time = fun.root.time[:]
        >>> obj = fs.fdawarp(f,time)
        >>> obj.srsf_align()
        """
        M = self.f.shape[0]
        N = self.f.shape[1]
        self.lam = lam

        if M > 500:
            parallel = True
        elif N > 100:
            parallel = True

        eps = np.finfo(np.double).eps
        f0 = self.f
        self.method = omethod

        methods = ["mean", "median"]
        self.type = method

        # 0 mean, 1-median
        method = [i for i, x in enumerate(methods) if x == method]
        if len(method) == 0:
            method = 0
        else:
            method = method[0]

        # Compute SRSF function from data
        f, g, g2 = uf.gradient_spline(self.time, self.f, smoothdata)
        q = g / np.sqrt(abs(g) + eps)

        if self.verbose:
            print("Initializing...")
        mnq = q.mean(axis=1)
        a = mnq.repeat(N)
        d1 = a.reshape(M, N)
        d = (q - d1) ** 2
        dqq = np.sqrt(d.sum(axis=0))
        min_ind = dqq.argmin()
        mq = q[:, min_ind]
        mf = f[:, min_ind]

        if parallel:
            out = Parallel(n_jobs=cores)(delayed(uf.optimum_reparam)(mq, self.time,
                                    q[:, n], omethod, lam, grid_dim) for n in range(N))
            gam = np.array(out)
            gam = gam.transpose()
        else:
            gam = np.zeros((M,N))
            for k in range(0,N):
                gam[:,k] = uf.optimum_reparam(mq,self.time,q[:,k],omethod,lam,grid_dim)

        gamI = uf.SqrtMeanInverse(gam)
        mf = np.interp((self.time[-1] - self.time[0]) * gamI + self.time[0], self.time, mf)
        mq = uf.f_to_srsf(mf, self.time)

        # Compute Karcher Mean
        if self.verbose:
            if method == 0:
                print("Compute Karcher Mean of %d function in SRSF space..." % N)
            if method == 1:
                print("Compute Karcher Median of %d function in SRSF space..." % N)

        ds = np.repeat(0.0, MaxItr + 2)
        ds[0] = np.inf
        qun = np.repeat(0.0, MaxItr + 1)
        tmp = np.zeros((M, MaxItr + 2))
        tmp[:, 0] = mq
        mq = tmp
        tmp = np.zeros((M, MaxItr+2))
        tmp[:,0] = mf
        mf = tmp
        tmp = np.zeros((M, N, MaxItr + 2))
        tmp[:, :, 0] = self.f
        f = tmp
        tmp = np.zeros((M, N, MaxItr + 2))
        tmp[:, :, 0] = q
        q = tmp

        for r in range(0, MaxItr):
            if self.verbose:
                print("updating step: r=%d" % (r + 1))
                if r == (MaxItr - 1):
                    print("maximal number of iterations is reached")

            # Matching Step
            if parallel:
                out = Parallel(n_jobs=cores)(delayed(uf.optimum_reparam)(mq[:, r],
                                        self.time, q[:, n, 0], omethod, lam, grid_dim) for n in range(N))
                gam = np.array(out)
                gam = gam.transpose()
            else:
                for k in range(0,N):
                    gam[:,k] = uf.optimum_reparam(mq[:, r], self.time, q[:, k, 0],
                            omethod, lam, grid_dim)

            gam_dev = np.zeros((M, N))
            vtil = np.zeros((M,N))
            dtil = np.zeros(N)
            for k in range(0, N):
                f[:, k, r + 1] = np.interp((self.time[-1] - self.time[0]) * gam[:, k]
                                        + self.time[0], self.time, f[:, k, 0])
                q[:, k, r + 1] = uf.f_to_srsf(f[:, k, r + 1], self.time)
                gam_dev[:, k] = np.gradient(gam[:, k], 1 / float(M - 1))
                v = q[:, k, r + 1] - mq[:,r]
                d = np.sqrt(trapz(v*v, self.time))
                vtil[:,k] = v/d
                dtil[k] = 1.0/d

            mqt = mq[:, r]
            a = mqt.repeat(N)
            d1 = a.reshape(M, N)
            d = (q[:, :, r + 1] - d1) ** 2
            if method == 0:
                d1 = sum(trapz(d, self.time, axis=0))
                d2 = sum(trapz((1 - np.sqrt(gam_dev)) ** 2, self.time, axis=0))
                ds_tmp = d1 + lam * d2
                ds[r + 1] = ds_tmp

                # Minimization Step
                # compute the mean of the matched function
                qtemp = q[:, :, r + 1]
                ftemp = f[:, :, r + 1]
                mq[:, r + 1] = qtemp.mean(axis=1)
                mf[:, r + 1] = ftemp.mean(axis=1)

                qun[r] = norm(mq[:, r + 1] - mq[:, r]) / norm(mq[:, r])

            if method == 1:
                d1 = np.sqrt(sum(trapz(d, self.time, axis=0)))
                d2 = sum(trapz((1 - np.sqrt(gam_dev)) ** 2, self.time, axis=0))
                ds_tmp = d1 + lam * d2
                ds[r + 1] = ds_tmp

                # Minimization Step
                # compute the mean of the matched function
                stp = .3
                vbar = vtil.sum(axis=1)*(1/dtil.sum())
                qtemp = q[:, :, r + 1]
                ftemp = f[:, :, r + 1]
                mq[:, r + 1] = mq[:,r] + stp*vbar
                tmp = np.zeros(M)
                tmp[1:] = cumtrapz(mq[:, r + 1] * np.abs(mq[:, r + 1]), self.time)
                mf[:, r + 1] = np.median(f0[1, :])+tmp

                qun[r] = norm(mq[:, r + 1] - mq[:, r]) / norm(mq[:, r])

            if qun[r] < 1e-2 or r >= MaxItr:
                break

        # Last Step with centering of gam

        if center:
            r += 1
            if parallel:
                out = Parallel(n_jobs=cores)(delayed(uf.optimum_reparam)(mq[:, r], self.time,
                    q[:, n, 0], omethod, lam, grid_dim) for n in range(N))
                gam = np.array(out)
                gam = gam.transpose()
            else:
                for k in range(0,N):
                    gam[:,k] = uf.optimum_reparam(mq[:, r], self.time, q[:, k, 0], omethod,
                            lam, grid_dim)

            gam_dev = np.zeros((M, N))
            for k in range(0, N):
                gam_dev[:, k] = np.gradient(gam[:, k], 1 / float(M - 1))

            gamI = uf.SqrtMeanInverse(gam)
            gamI_dev = np.gradient(gamI, 1 / float(M - 1))
            time0 = (self.time[-1] - self.time[0]) * gamI + self.time[0]
            mq[:, r + 1] = np.interp(time0, self.time, mq[:, r]) * np.sqrt(gamI_dev)

            for k in range(0, N):
                q[:, k, r + 1] = np.interp(time0, self.time, q[:, k, r]) * np.sqrt(gamI_dev)
                f[:, k, r + 1] = np.interp(time0, self.time, f[:, k, r])
                gam[:, k] = np.interp(time0, self.time, gam[:, k])
        else:
            gamI = uf.SqrtMeanInverse(gam)
            gamI_dev = np.gradient(gamI, 1 / float(M - 1))

        # Aligned data & stats
        self.fn = f[:, :, r + 1]
        self.qn = q[:, :, r + 1]
        self.q0 = q[:, :, 0]
        mean_f0 = f0.mean(axis=1)
        std_f0 = f0.std(axis=1)
        mean_fn = self.fn.mean(axis=1)
        std_fn = self.fn.std(axis=1)
        self.gam = gam
        self.mqn = mq[:, r + 1]
        tmp = np.zeros(M)
        tmp[1:] = cumtrapz(self.mqn * np.abs(self.mqn), self.time)
        self.fmean = np.mean(f0[1, :]) + tmp

        fgam = np.zeros((M, N))
        for k in range(0, N):
            time0 = (self.time[-1] - self.time[0]) * gam[:, k] + self.time[0]
            fgam[:, k] = np.interp(time0, self.time, self.fmean)

        var_fgam = fgam.var(axis=1)
        self.orig_var = trapz(std_f0 ** 2, self.time)
        self.amp_var = trapz(std_fn ** 2, self.time)
        self.phase_var = trapz(var_fgam, self.time)

        return
