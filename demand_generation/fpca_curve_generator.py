import fdasrsf as fs
import fdasrsf.utility_functions as uf
from scipy.linalg import norm
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import math

from copula import pyCopula

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

class fpca_curve_generator:
    
    def __init__(self, input_data):
        f = input_data
        f = np.float64(f)
        time = np.arange(1,f.shape[0]+1)
        time = np.float64(time)
        
        self.time = time
        self.obj = fs.fdawarp(f, time)
        
    def calc_fpca(self, is_smooth, is_parallel, maxiter, ncompv, ncomph):
        self.obj.srsf_align(smoothdata=is_smooth,parallel=is_parallel,MaxItr=maxiter)
        
        self.vpca = fs.fdavpca(self.obj)
        self.vpca.calc_fpca(no=ncompv)
        self.hpca = fs.fdahpca(self.obj)
        self.hpca.calc_fpca(no=ncomph)
        
        self.vpca_U = self.vpca.U
        self.vpca_mqn = self.vpca.mqn
        self.vpca_coef = self.vpca.coef
        self.hpca_U = self.hpca.U
        self.hpca_coef = self.hpca.coef
        self.hpca_mu = self.hpca.psi_mu
        
    def distribution_est(self, in_kernel):
        self.vkde_list = []
        opt_bw_v = []
        self.hkde_list = []
        opt_bw_h = []
        for i in range(0,self.vpca_coef.shape[1]):
            X = np.ndarray(shape=(self.vpca_coef.shape[0],1))
            X[:,0] = self.vpca_coef[:,i]
            bandwidth = np.arange(0.05, 2, 0.05)
            kde = KernelDensity(kernel=in_kernel)
            grid = GridSearchCV(kde, {'bandwidth': bandwidth})
            grid.fit(X)
            kde = grid.best_estimator_
            opt_bw_v.append(kde.bandwidth)
            self.vkde_list.append(kde)
            #plt.figure()
            #X_plot = np.linspace(math.floor(min(X)),math.ceil(max(X)),1000)[:,np.newaxis]
            #log_dens = kde.score_samples(X_plot)
            #plt.plot(X_plot, np.exp(log_dens))
            #plt.hist(X, density=True, bins=40);
        for i in range(0,self.hpca_coef.shape[1]):
            X = np.ndarray(shape=(self.hpca_coef.shape[0],1))
            X[:,0] = self.hpca_coef[:,i]
            bandwidth = np.arange(0.05, 2, 0.05)
            kde = KernelDensity(kernel=in_kernel)
            grid = GridSearchCV(kde, {'bandwidth': bandwidth})
            grid.fit(X)
            kde = grid.best_estimator_
            opt_bw_h.append(kde.bandwidth)
            self.hkde_list.append(kde)
        
    def curve_gen(self, num):
        gen_vcoef = np.ndarray(shape=(num,self.vpca_coef.shape[1]))
        gen_qn = np.ndarray(shape=(len(self.vpca_mqn), num))
        gen_fn = np.ndarray(shape=(len(self.vpca_mqn)-1, num))
    
        gen_hcoef = np.ndarray(shape=(num,self.hpca_coef.shape[1]))
        gen_v = np.ndarray(shape=(self.hpca_U.shape[0], num))
        gen_psi = np.ndarray(shape=(self.hpca_U.shape[0], num))
        gen_gam = np.ndarray(shape=(self.hpca_U.shape[0], num))
    
        gen_f = np.ndarray(shape=(self.hpca_U.shape[0], num))
        
        midid = int(np.round(len(self.time)/2))

        for i in range(0,self.vpca_coef.shape[1]):
            X = np.ndarray(shape=(self.vpca_coef.shape[0],1))
            X[:,0] = self.vpca_coef[:,i]
            kde = self.vkde_list[i]
            if abs(min(X)) < 1e-8 and abs(max(X)) < 1e-8 :
                gen_vcoef[:,i] = 0
            else:
                gen_vcoef[:,i] = np.transpose(kde.sample(n_samples = num))
        #Generate qn and fn
        for i in range(0, num):
            gen_qn[:,i] = self.vpca_mqn
            for j in range(0, self.vpca_coef.shape[1]):
                gen_qn[:,i] = gen_qn[:,i] + gen_vcoef[i,j]*self.vpca_U[:,j]
        for i in range(0, num):
            gen_fn[:,i] = uf.cumtrapzmid(self.time, gen_qn[0:(len(self.vpca_mqn)-1),i] * np.abs(gen_qn[0:(len(self.vpca_mqn)-1),i]), np.sign(gen_qn[(len(self.vpca_mqn)-1),i]) * (gen_qn[(len(self.vpca_mqn)-1),i] ** 2), midid+1)    
            
        for i in range(0,self.hpca_coef.shape[1]):
            X = np.ndarray(shape=(self.hpca_coef.shape[0],1))
            X[:,0] = self.hpca_coef[:,i]
            kde = self.hkde_list[i]
            if abs(min(X)) < 1e-8 and abs(max(X)) < 1e-8 :
                gen_hcoef[:,i] = 0
            else:
                gen_hcoef[:,i] = np.transpose(kde.sample(n_samples = num))
            
        #Generate gamma
        for i in range(0, num):
            gen_v[:,i] = 0
            for j in range(0, gen_hcoef.shape[1]):
                gen_v[:,i] = gen_v[:,i] + gen_hcoef[i,j]*self.hpca_U[:,j]
            vn = norm(gen_v[:,i]) / np.sqrt(self.hpca_U.shape[0])
            if vn < 0.0001:
                gen_psi[:,i] = self.hpca_mu
            else:
                gen_psi[:,i] = np.cos(vn)*self.hpca_mu + np.sin(vn)*gen_v[:,i]/vn
            tmp = cumtrapz(gen_psi[:,i]*gen_psi[:,i], np.linspace(0,1,self.hpca_U.shape[0]), initial=0)
            gen_gam[:,i] = (tmp - tmp[0]) / (tmp[-1] - tmp[0])
            
        #Generate f
        for i in range(0, num):
            gen_f[:,i] = uf.warp_f_gamma(gen_gam[:,i], gen_fn[:,i], np.linspace(0,1,len(gen_fn[:,i])))
            
        return gen_f

    def correlated_curve_gen(self, num):
        gen_vcoef = self.gen_correlated_vcoef
        gen_qn = np.ndarray(shape=(len(self.vpca_mqn), num))
        gen_fn = np.ndarray(shape=(len(self.vpca_mqn)-1, num))
    
        gen_hcoef = self.gen_correlated_hcoef
        gen_v = np.ndarray(shape=(self.hpca_U.shape[0], num))
        gen_psi = np.ndarray(shape=(self.hpca_U.shape[0], num))
        gen_gam = np.ndarray(shape=(self.hpca_U.shape[0], num))
    
        gen_f = np.ndarray(shape=(self.hpca_U.shape[0], num))
        
        midid = int(np.round(len(self.time)/2))

        #Generate qn and fn
        for i in range(0, num):
            gen_qn[:,i] = self.vpca_mqn
            for j in range(0, self.vpca_coef.shape[1]):
                gen_qn[:,i] = gen_qn[:,i] + gen_vcoef[i,j]*self.vpca_U[:,j]
        for i in range(0, num):
            gen_fn[:,i] = uf.cumtrapzmid(self.time, gen_qn[0:(len(self.vpca_mqn)-1),i] * np.abs(gen_qn[0:(len(self.vpca_mqn)-1),i]), np.sign(gen_qn[(len(self.vpca_mqn)-1),i]) * (gen_qn[(len(self.vpca_mqn)-1),i] ** 2), midid+1)    
            
        #Generate gamma
        for i in range(0, num):
            gen_v[:,i] = 0
            for j in range(0, gen_hcoef.shape[1]):
                gen_v[:,i] = gen_v[:,i] + gen_hcoef[i,j]*self.hpca_U[:,j]
            vn = norm(gen_v[:,i]) / np.sqrt(self.hpca_U.shape[0])
            if vn < 0.0001:
                gen_psi[:,i] = self.hpca_mu
            else:
                gen_psi[:,i] = np.cos(vn)*self.hpca_mu + np.sin(vn)*gen_v[:,i]/vn
            tmp = cumtrapz(gen_psi[:,i]*gen_psi[:,i], np.linspace(0,1,self.hpca_U.shape[0]), initial=0)
            gen_gam[:,i] = (tmp - tmp[0]) / (tmp[-1] - tmp[0])
            
        #Generate f
        for i in range(0, num):
            gen_f[:,i] = uf.warp_f_gamma(gen_gam[:,i], gen_fn[:,i], np.linspace(0,1,len(gen_fn[:,i])))
            
        return gen_f


    def distribution_est_piecewise(self, in_kernel, disp):
        self.vkde_pw_list = []
        #opt_bw_v = []
        self.hkde_pw_list = []
        #opt_bw_h = []
        for j in range(0, len(disp)-1):
            id_start = disp[j]
            id_end = disp[j+1]
            id_length = id_end - id_start
            tmp_vkde_list = []
            tmp_hkde_list = []
            for i in range(0,self.vpca_coef.shape[1]):
                X = np.ndarray(shape=(id_length, 1))
                X[:,0] = self.vpca_coef[id_start:id_end,i]
                bandwidth = np.arange(0.05, 2, 0.05)
                kde = KernelDensity(kernel=in_kernel)
                grid = GridSearchCV(kde, {'bandwidth': bandwidth})
                grid.fit(X)
                kde = grid.best_estimator_
                #opt_bw_v.append(kde.bandwidth)
                tmp_vkde_list.append(kde)
            for i in range(0,self.hpca_coef.shape[1]):
                X = np.ndarray(shape=(id_length,1))
                X[:,0] = self.hpca_coef[id_start:id_end,i]
                bandwidth = np.arange(0.05, 2, 0.05)
                kde = KernelDensity(kernel=in_kernel)
                grid = GridSearchCV(kde, {'bandwidth': bandwidth})
                grid.fit(X)
                kde = grid.best_estimator_
                #opt_bw_h.append(kde.bandwidth)
                tmp_hkde_list.append(kde)
            self.vkde_pw_list.append(tmp_vkde_list)
            self.hkde_pw_list.append(tmp_hkde_list)
    

    def curve_gen_piecewise(self, disp):

        gen_f_total = np.ndarray(shape=(self.time.shape[0], disp[-1]))

        for m in range(0, len(disp)-1):
            id_start = disp[m]
            id_end = disp[m+1]
            id_length = id_end - id_start
            
            vkde_pw = self.vkde_pw_list[m]
            hkde_pw = self.hkde_pw_list[m]
            
            gen_vcoef = np.ndarray(shape=(id_length,self.vpca_coef.shape[1]))
            gen_qn = np.ndarray(shape=(len(self.vpca_mqn), id_length))
            gen_fn = np.ndarray(shape=(len(self.vpca_mqn)-1, id_length))
             
            gen_hcoef = np.ndarray(shape=(id_length,self.hpca_coef.shape[1]))
            gen_v = np.ndarray(shape=(self.hpca_U.shape[0], id_length))
            gen_psi = np.ndarray(shape=(self.hpca_U.shape[0], id_length))
            gen_gam = np.ndarray(shape=(self.hpca_U.shape[0], id_length))
             
            gen_f = np.ndarray(shape=(self.hpca_U.shape[0], id_length))
                 
            midid = int(np.round(len(self.time)/2))

            for i in range(0,self.vpca_coef.shape[1]):
                X = np.ndarray(shape=(id_length,1))
                X[:,0] = self.vpca_coef[id_start:id_end,i]
                kde = vkde_pw[i]
                if abs(min(X)) < 1e-8 and abs(max(X)) < 1e-8 :
                    gen_vcoef[:,i] = 0
                else:
                    gen_vcoef[:,i] = np.transpose(kde.sample(n_samples = id_length))
            #Generate qn and fn
            for i in range(0, id_length):
                gen_qn[:,i] = self.vpca_mqn
                for j in range(0, self.vpca_coef.shape[1]):
                    gen_qn[:,i] = gen_qn[:,i] + gen_vcoef[i,j]*self.vpca_U[:,j]
            for i in range(0, id_length):
                gen_fn[:,i] = uf.cumtrapzmid(self.time, gen_qn[0:(len(self.vpca_mqn)-1),i] * np.abs(gen_qn[0:(len(self.vpca_mqn)-1),i]), np.sign(gen_qn[(len(self.vpca_mqn)-1),i]) * (gen_qn[(len(self.vpca_mqn)-1),i] ** 2), midid+1)    
            
            for i in range(0,self.hpca_coef.shape[1]):
                X = np.ndarray(shape=(self.hpca_coef.shape[0],1))
                X[:,0] = self.hpca_coef[:,i]
                kde = hkde_pw[i]
                if abs(min(X)) < 1e-8 and abs(max(X)) < 1e-8 :
                 gen_hcoef[:,i] = 0
                else:
                 gen_hcoef[:,i] = np.transpose(kde.sample(n_samples = id_length))
            
            #Generate gamma
            for i in range(0, id_length):
                gen_v[:,i] = 0
                for j in range(0, gen_hcoef.shape[1]):
                    gen_v[:,i] = gen_v[:,i] + gen_hcoef[i,j]*self.hpca_U[:,j]
                vn = norm(gen_v[:,i]) / np.sqrt(self.hpca_U.shape[0])
                if vn < 0.0001:
                    gen_psi[:,i] = self.hpca_mu
                else:
                    gen_psi[:,i] = np.cos(vn)*self.hpca_mu + np.sin(vn)*gen_v[:,i]/vn
                tmp = cumtrapz(gen_psi[:,i]*gen_psi[:,i], np.linspace(0,1,self.hpca_U.shape[0]), initial=0)
                gen_gam[:,i] = (tmp - tmp[0]) / (tmp[-1] - tmp[0])
            
            #Generate f
            for i in range(0, id_length):
                gen_f[:,i] = uf.warp_f_gamma(gen_gam[:,i], gen_fn[:,i], np.linspace(0,1,len(gen_fn[:,i])))
            
            gen_f_total[:, id_start:id_end] = gen_f
            
        return gen_f_total
    
#dg = fpca_demand_generator('hourly_demand/RESIDENT.csv', 365, 24)
#dg.calc_fpca(False, True, 100, 5, 5)
#dg.distribution_est('gaussian')
#dg.calc_total_demand(365)


def correlated_ceof_gen(pca_1, pca_2, num):
    vpca_coef_list = []
    for i in range(0, pca_1.vpca_coef.shape[0]):
        vpca_coef_list.append(pca_1.vpca_coef[i,:].tolist()+pca_2.vpca_coef[i,:].tolist())
    vpca_cop = pyCopula.Copula(vpca_coef_list)
    vpca_samples_list = vpca_cop.gendata(num)
    gen_vcoef = np.asarray(vpca_samples_list)
    pca_1.gen_correlated_vcoef = gen_vcoef[:, 0:pca_1.vpca_coef.shape[1]]
    pca_2.gen_correlated_vcoef = gen_vcoef[:, pca_1.vpca_coef.shape[1]:(2*pca_1.vpca_coef.shape[1])]

    hpca_coef_list = []
    for i in range(0, pca_1.hpca_coef.shape[0]):
        hpca_coef_list.append(pca_1.hpca_coef[i,:].tolist()+pca_2.hpca_coef[i,:].tolist())
    hpca_cop = pyCopula.Copula(hpca_coef_list)
    hpca_samples_list = hpca_cop.gendata(num)
    gen_hcoef = np.asarray(hpca_samples_list)
    pca_1.gen_correlated_hcoef = gen_hcoef[:, 0:pca_1.hpca_coef.shape[1]]
    pca_2.gen_correlated_hcoef = gen_hcoef[:, pca_1.hpca_coef.shape[1]:(2*pca_1.hpca_coef.shape[1])]



def mv_kl_div(x, y):
  """Compute the Kullback-Leibler divergence between two multivariate samples.
  Parameters
  ----------
  x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
  y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
  Returns
  -------
  out : float
    The estimated Kullback-Leibler divergence D(P||Q).
  References
  ----------
  Pérez-Cruz, F. Kullback-Leibler divergence estimation of
continuous distributions IEEE International Symposium on Information
Theory, 2008.
  """
  from scipy.spatial import cKDTree as KDTree

  # Check the dimensions are consistent
  x = np.atleast_2d(x)
  y = np.atleast_2d(y)

  n,d = x.shape
  m,dy = y.shape

  assert(d == dy)


  # Build a KD tree representation of the samples and find the nearest neighbour
  # of each point in x.
  xtree = KDTree(x)
  ytree = KDTree(y)

  # Get the first two nearest neighbours for x, since the closest one is the
  # sample itself.
  r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
  s = ytree.query(x, k=1, eps=.01, p=2)[0]

  # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
  # on the first term of the right hand side.
  return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))

