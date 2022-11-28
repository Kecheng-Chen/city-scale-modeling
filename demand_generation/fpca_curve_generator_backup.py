import fdasrsf as fs
import fdasrsf.utility_functions as uf
from scipy.linalg import norm
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import math

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


	def distribution_est_piecewise(self, in_kernel, disp):
		self.vkde_pw_list = []
		tmp_vkde_list = []
		#opt_bw_v = []
		self.hkde_pw_list = []
		tmp_hkde_list = []
		#opt_bw_h = []
		for j in range(0, len(disp)-1):
			id_start = disp[j]
			id_end = disp[j+1]
			id_length = id_end - id_start
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

		gen_f_total = np.ndarray(shape=(self.time, disp[-1]))

		for m in range(0, len(disp)-1):
			id_start = disp[m]
			id_end = disp[m+1]
			id_length = id_end - id_start

			vkde_pw = vkde_pw_list[m]
			hkde_pw = hkde_pw_list[m]

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
