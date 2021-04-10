import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.gridspec as gridspec
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import GPy

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()

'''
Package for Bayesian Optimization.
    - Bayesian Optimization 
    - Contextual Bayesian Optimization
    - Constrained Bayesian Optimization

Args:
        X_sample - Action X used to build the statistical model
        Y_sample - f(X)
        bounds - for the action space 
        kernal - GPy.kern
        optimize - optimize hyperparameters? True/False


Requires GPy package for Gaussian Process Regression 
 https://sheffieldml.github.io/GPy/
 
    written by: Dinesh Krishnamoorthy, July 2020
'''
class bayesian_optimization:
    # Written by: Dinesh Krishnamoorthy, July 2020

    def __init__(self,X_sample,Y_sample,bounds,
                          kernel,
                          mf = None,
                          X_grid = np.linspace(0,1,100),
                          obj_fun = None,):
        self.X_sample = X_sample
        self.Y_sample = Y_sample
        self.obj_fun = obj_fun
        self.bounds = bounds
        self.kernel = kernel
        self.mf = mf
        self.X_grid = X_grid
        self.grid_size = self.X_grid.shape[0]
        self.nX = self.X_sample.shape[1] 
        

    def fit_gp(self):
        if self.mf is not None:
            self.m = GPy.models.GPRegression(self.X_sample,self.Y_sample,self.kernel,mean_function = self.mf)
        else:
            self.m = GPy.models.GPRegression(self.X_sample,self.Y_sample,self.kernel)
        return self
    
    def optimize_fit(self):
        return self.m.optimize()
    
    def get_dim(self):  
        return self.X_sample.shape[1] 
    
    # Acquisition functions:

    def LCB(self,X):
        '''
        Acquisition function: Lower confidence bound
        '''
        mu, sigma = self.m.predict(X.reshape(-1,self.nX))
        return mu - 2*sigma
    
    def EI0(self,X,xi=0.01):
        '''
        Acquisition function: Expected improvement
        '''
        mu, sigma = self.m.predict(X.reshape(-1,self.nX))
        mu_sample,si = self.m.predict(self.X_sample)
        f_best = np.max(-mu_sample) # incumbent
        with np.errstate(divide='warn'):
            imp = -mu - f_best - xi
            Z = imp/sigma
            EI = (imp*norm.cdf(Z) + sigma*norm.pdf(Z))
            EI[sigma == 0.0] = 0.0
        return EI

    def PI0(self,X,xi=0.01):
        '''
        Acquisition function: Probability of improvement
        '''
        mu, sigma = self.m.predict(X.reshape(-1,self.nX))
        mu_sample,si = self.m.predict(self.X_sample)
        f_best = np.max(-mu_sample) # incumbent
        with np.errstate(divide='warn'):
            imp = -mu - f_best - xi
            Z = imp/sigma
            PI = norm.cdf(Z)
            PI[sigma == 0.0] = 0.0
        return PI

    def EI(self,X,xi=0.01):
        '''
        Acquisition function: Expected improvement with aumented terms
        '''
        mu, sigma = self.m.predict(X.reshape(-1,self.nX))
        if self.Aug is not None:
            assert callable(self.Aug),'self.Aug must be a callable function of X'
            mu += self.Aug(X,*self.args)

        mu_sample,si = self.m.predict(self.X_sample)
        if self.Aug is not None:
            assert callable(self.Aug),'self.Aug must be a callable function of X'
            mu_sample += self.Aug(self.X_sample,*self.args)
            
        f_best = np.max(-mu_sample) # incumbent
        with np.errstate(divide='warn'):
            imp = -mu - f_best - xi
            Z = imp/sigma
            EI = (imp*norm.cdf(Z) + sigma*norm.pdf(Z))
            EI[sigma == 0.0] = 0.0
        return EI

    def PI(self,X,xi=0.01):
        '''
        Acquisition function: Probability of improvement
        '''
        mu, sigma = self.m.predict(X.reshape(-1,self.nX))
        if self.Aug is not None:
            assert callable(self.Aug),'self.Aug must be a callable function of X'
            mu += self.Aug(X,*self.args)

        mu_sample,si = self.m.predict(self.X_sample)
        if self.Aug is not None:
            assert callable(self.Aug),'self.Aug must be a callable function of X'
            mu_sample += self.Aug(self.X_sample,*self.args)
            
        f_best = np.max(-mu_sample) # incumbent
        with np.errstate(divide='warn'):
            imp = -mu - f_best - xi
            Z = imp/sigma
            PI = norm.cdf(Z)
            PI[sigma == 0.0] = 0.0
        return PI

    def greedy(self,X,epsilon=0.1):
        '''
        Acquisition function:  epsilon-greedy
        '''
        p = np.random.uniform(0,1)
        if p<epsilon:
            mu = np.random.uniform(self.bounds[:,0],self.bounds[:,1])
        else:
            mu, sigma = self.m.predict(X.reshape(-1,self.nX))
        return mu

    def query_next(self,acquisition='EI',epsilon=0,xi=0,Aug=None,args=()): 
        '''
        Function that computes the next query point.

        Inputs:
            - acquisition: PI, EI, LCB, TS, greedy
            - epsilon: used for epsilon greedy policy (default = 0)
            - xi: Exploration factor used in PI and EI policies (deafult = 0)
            - Aug: Function delta(X) to augment the acqusition function with additional terms (default = None)
            - args: additional arguments for the augmented function (default = None)

        Output:
            - self.X_next
        '''
        self.acquisition = acquisition
        self.Aug = Aug
        self.args = args
        self.epsilon = epsilon
        self.xi = xi
        nX = self.X_sample.shape[1] 
        min_val = -1e-5
        min_x = self.X_sample[-1,:]
        n_restarts=25

        def min_obj(X,self):
            if self.acquisition=='LCB' or self.acquisition=='UCB' :
                alpha = self.LCB(X)
                if self.Aug is not None:
                    assert callable(self.Aug),'self.Aug must be a callable function of X'
                    alpha += self.Aug(X,*self.args)
            if self.acquisition == 'EI':
                alpha = -self.EI(X,xi = self.xi)
            if self.acquisition == 'PI':
                alpha = -self.PI(X,xi = self.xi)
            if self.acquisition == 'greedy':
                alpha = self.greedy(X,epsilon = self.epsilon)
                if self.Aug is not None:
                    assert callable(self.Aug),'self.Aug must be a callable function of X'
                    alpha += self.Aug(X,*self.args)
            return alpha 

        if self.acquisition == 'TS':
            
            self.posterior_sample = self.m.posterior_samples_f(self.X_grid,full_cov=True,size = 1).reshape(-1,nX)
            if self.Aug is not None:
                assert callable(self.Aug),'self.Aug must be a callable function of X'
                self.posterior_sample_aug =  self.posterior_sample + self.Aug(self.X_grid,*self.args)
                self.min_index = np.argmin(self.posterior_sample_aug)
            else:
                self.posterior_sample_aug =  self.posterior_sample
                self.min_index = np.argmin(self.posterior_sample)
            
            self.X_next = self.X_grid[self.min_index].reshape(-1, nX) 
            self.min_val = np.min(self.posterior_sample)
        else:
            # Find the best optimum by starting from n_restart different random points.
            for x0 in np.random.uniform(self.bounds[:, 0].T, self.bounds[:, 1].T,
                                        size=(n_restarts, nX)):
                res = minimize(min_obj, x0=x0, args = (self),
                            bounds=self.bounds, method='L-BFGS-B')        
                if res.fun < min_val:
                    min_val = res.fun[0]
                    min_x = res.x         
                
            self.X_next =  min_x.reshape(-1, nX)
            self.min_val = min_val
        return self
    
    def acq(self):
        if self.acquisition=='LCB' or self.acquisition=='UCB':
            self.acq_fn = self.LCB(self.X_grid)
            self.acq_fn0 = self.LCB(self.X_grid)
        if self.acquisition=='EI':
            self.acq_fn = -self.EI(self.X_grid)
            self.acq_fn0 = -self.EI0(self.X_grid)
        if self.acquisition=='PI':
            self.acq_fn = -self.PI(self.X_grid)
            self.acq_fn0 = -self.PI0(self.X_grid)
        if self.acquisition == 'TS':
            self.acq_fn = self.posterior_sample_aug
            self.acq_fn0 = self.posterior_sample
        if self.acquisition == 'greedy':
            self.acq_fn = self.greedy(self.X_grid)
            self.acq_fn0 = self.greedy(self.X_grid)
            
        if self.Aug is not None:
            assert callable(self.Aug),'self.Aug must be a callable function of X'
            self.acq_fn += self.Aug(self.X_grid,*self.args)
        return self

    def plot(self,plot_acq = True,plot_ideal=False,fig_name=''):
        assert self.X_sample.shape[1]==1, "X dimension must be 1 for this function"
        mu, sigma = self.m.predict(self.X_grid)

        if plot_acq:
            self.acq()
            fig = plt.figure(constrained_layout=True)
            spec = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)
            ax1 = fig.add_subplot(spec[0:2, 0])
            ax2 = fig.add_subplot(spec[2, 0])
            ax1.fill_between(self.X_grid.ravel(), 
                        mu.ravel() + 2 * sigma.ravel(), 
                        mu.ravel() - 2 * sigma.ravel(), 
                        alpha=0.1,label='Confidence') 
            ax1.plot(self.X_grid,mu+2*sigma,color=(0,0.4,0.7,0.1),linewidth=0.3)
            ax1.plot(self.X_grid,mu-2*sigma,color=(0,0.4,0.7,0.1),linewidth=0.3)
            ax1.plot(self.X_grid,mu,color=(0,0.4,0.7),linewidth=2.4,label='Mean')
            ax1.plot(self.X_sample,self.Y_sample,'kx',label='Data')
            if plot_ideal:
                ax1.plot(self.X_grid,self.obj_fun(self.X_grid),'--',color=(0.6,0.6,0.6),linewidth=1,label='Ground truth')
        
            
            ax2.plot(self.X_grid,self.acq_fn,color = (0,0.7,0,0.8),linewidth=2)
            if self.Aug is not None:
                ax2.plot(self.X_grid,self.acq_fn0,'--',color = (0,0.7,0,0.8),linewidth=1)
            ax2.plot(self.X_next,self.min_val,'ro',label='$x_{next}$')
            ax2.fill_between(self.X_grid.ravel(), 
                        np.max(self.acq_fn)+0*self.X_grid.ravel(), 
                        self.acq_fn.ravel(),color='g', 
                        alpha=0.1)
            ax1.set_xlabel('$x$')
            ax1.set_ylabel('Objective $f(x)$')
            ax2.set_ylabel('Acqusition fn')
            ax2.set_xlabel('$x$')
            ax1.legend()
            ax2.legend()
            plt.show()
        else:
            plt.fill_between(self.X_grid.ravel(), 
                        mu.ravel() + 2 * sigma.ravel(), 
                        mu.ravel() - 2 * sigma.ravel(), 
                        alpha=0.1,label='Confidence') 
            plt.plot(self.X_grid,mu+2*sigma,color=(0,0.4,0.7,0.1),linewidth=0.3)
            plt.plot(self.X_grid,mu-2*sigma,color=(0,0.4,0.7,0.1),linewidth=0.3)
            plt.plot(self.X_grid,mu,color=(0,0.4,0.7),linewidth=2.4,label='Mean')
            plt.plot(self.X_sample,self.Y_sample,'kx',label='Data')
            if plot_ideal:
                plt.plot(self.X_grid,self.obj_fun(self.X_grid),'--',color=(0.6,0.6,0.6),linewidth=1,label='Ground truth')
            plt.xlabel('$x$')
            plt.ylabel('Objective $f(x)$')
            plt.legend()
            plt.show()
        if fig_name:
            fig.savefig(fig_name+'.pdf',bbox_inches='tight') 

    def plot2d(self,plot_ideal=False,fig_name=''):
        lb = self.bounds[:,0]
        ub = self.bounds[:,1]
        X,Y = np.mgrid[lb[0]:ub[0]:100j, lb[1]:ub[1]:100j]
        X_grid = np.mgrid[lb[0]:ub[0]:100j, lb[1]:ub[1]:100j].reshape(2,-1).T
        mu, sigma = self.m.predict(X_grid)
        Z = mu.reshape(100,100)
        Z0 = self.obj_fun(X_grid).reshape(100,100)  # Ground truth

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        if plot_ideal:
                ax.plot_surface(X,Y,Z0,rstride=8, cstride=8, alpha=0.1,label='Ground Truth')
        ax.scatter(self.X_sample[:,0], self.X_sample[:,1], self.Y_sample,label='Data')
        ax.scatter(self.X_next[:,0], self.X_next[:,1], 0,'rx')
        ax.plot_surface(X,Y,Z,rstride=8, cstride=8, alpha=0.3,label='Posterior mean')
        cset = ax.contour(X, Y, Z, zdir='z', offset=np.min(Z)-1, cmap=cm.coolwarm)
        #cset = ax.contour(X, Y, Z, zdir='x', offset=lb[0], cmap=cm.coolwarm)
        #cset = ax.contour(X, Y, Z, zdir='y', offset=ub[1], cmap=cm.coolwarm)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$f(x1,x2)$')
        #plt.legend()
        plt.show()
        if fig_name:
            fig.savefig(fig_name+'.pdf',bbox_inches='tight') 


class contextual_bayesian_optimization:
    # Written by: Dinesh Krishnamoorthy, July 2020

    def __init__(self,X_sample,Y_sample,context,bounds,
                          kernel,
                          mf = None,
                          X_grid = np.linspace(0,1,100),
                          obj_fun = None):
        self.X_sample = X_sample  # X - combined action-context space
        self.Y_sample = Y_sample  # Observed objective function
        self.context = context  # New context
        self.obj_fun = obj_fun 
        self.bounds = bounds
        self.kernel = kernel
        self.mf = mf
        self.X_grid = X_grid
        self.grid_size = self.X_grid.shape[0]
        self.nX = self.X_sample.shape[1] 
        self.nC = self.context.shape[1]
        self.nU = self.nX - self.nC
        
        
    def fit_gp(self):
        if self.mf is not None:
            self.m = GPy.models.GPRegression(self.X_sample,self.Y_sample,self.kernel,mean_function = self.mf)
        else:
            self.m = GPy.models.GPRegression(self.X_sample,self.Y_sample,self.kernel)
        return self
    
    def optimize_fit(self):
        return self.m.optimize()
    
    def extract_action_space(self):
        nU = self.nX-self.nC
        return np.array(self.X_sample[:,0:nU+1]).reshape(-1,nU)
    
    def query_next_TS(self): # Thompson Sampling
        C_grid = self.context*np.ones([self.grid_size,1])#.reshape(-1,nC)
        testX = np.concatenate((self.X_grid.reshape(-1,self.nU),C_grid),axis=1)
        posterior_sample = self.m.posterior_samples_f(testX,full_cov=True,size = 1).reshape(-1,1)
        
        min_index = np.argmin(posterior_sample)
        U_next = self.X_grid[min_index]
        self.X_next = np.concatenate((U_next.reshape(-1,self.nU),self.context),axis=1) #np.array([U_next,self.context]).ravel()
        return self
    
    def query_next_UCB(self): #Upper confidence bound 
        C_grid = self.context*np.ones([self.grid_size,1])#.reshape(-1,nC)
        testX = np.concatenate((self.X_grid.reshape(-1,self.nU),C_grid),axis=1)
        mu, sigma = self.m.predict(testX)
        self.LCB = mu - 2*sigma
        min_index = np.argmin(self.LCB)
        U_next = self.X_grid[min_index]
        self.X_next = np.concatenate((U_next.reshape(-1,self.nU),self.context),axis=1) #np.array([U_next,self.context]).ravel()
        return self

    def LCB(self,X):  
        testX = np.concatenate((X.reshape(-1,self.nU),self.context),axis=1)
        mu, sigma = self.m.predict(testX.reshape(-1,self.nX))
        return mu - 2*sigma

    def expect(self,X):
        testX = np.concatenate((X.reshape(-1,self.nU),self.context),axis=1)
        mu, sigma = self.m.predict(testX.reshape(-1,self.nX))
        return mu

    def query_next(self,acquisition='LCB'):

        self.acquisition = acquisition
        min_val = -1e-5
        min_x = self.X_sample[-1,0] # Latext X
        n_restarts=25

        if self.acquisition == 'TS':
            C_grid = self.context*np.ones([self.grid_size,1])#.reshape(-1,nC)
            testX = np.concatenate((self.X_grid.reshape(-1,self.nU),C_grid),axis=1)
            self.posterior_sample = self.m.posterior_samples_f(testX,full_cov=True,size = 1).reshape(-1,1)
            self.min_index = np.argmin(self.posterior_sample)
            
            U_next = self.X_grid[self.min_index]
            self.X_next = np.concatenate((U_next.reshape(-1,self.nU),self.context),axis=1)
            self.min_val = np.min(self.posterior_sample)
        else:
            def min_obj(X,self):
                if acquisition=='LCB' or acquisition=='UCB' :
                    alpha = self.LCB(X)
                elif acquisition == 'expect':
                    alpha = self.expect(X)
                return alpha

            # Find the best optimum by starting from n_restart different random points.
            for x0 in np.random.uniform(self.bounds[:, 0].T, self.bounds[:, 1].T,
                                        size=(n_restarts, self.nU)):
                res = minimize(min_obj, x0=x0, args = (self),
                            bounds=self.bounds, method='L-BFGS-B')        
                if res.fun < min_val:
                    min_val = res.fun[0]
                    min_x = res.x           

            U_next = min_x.reshape(-1, self.nU)
            self.X_next = np.concatenate((U_next,self.context),axis=1)
            self.min_val = min_val
        return self

    def plot_action_context_space(self,action_space=np.linspace(0,1,10),
                                    context_space=np.linspace(0,1,10),
                                    projection = '2d',
                                    plot_ideal=False,
                                    ideal_c = None,
                                    ideal_x = None,
                                    xlabel = 'Action $x$',
                                    ylabel = 'Context $d$',
                                    zlabel = 'Objective $f(x,d)$',
                                    fig_name=''):

        nX = action_space.shape[0]
        nC = context_space.shape[0]
        X,C = np.meshgrid(action_space,context_space)
        X_grid = np.vstack((X.flatten(), C.flatten())).T
        mu, sigma = self.m.predict(X_grid)
        Z = mu.reshape(nC,nX)

        C_now = self.context*np.ones([self.grid_size,1])
        testX = np.concatenate((action_space,C_now),axis=1)
        mu1, sigma1 = self.m.predict(testX)

        fig = plt.figure()
        if projection == '2d':
            #plt.scatter(self.X_sample[:,0], self.X_sample[:,1], self.Y_sample,label='Data')
            plt.plot(self.X_sample[:,0], self.X_sample[:,1],'ko',label='Data')
            plt.contour(X, C, Z,cmap=cm.coolwarm)
            plt.plot(action_space.reshape(self.grid_size,), C_now.reshape(self.grid_size,),'--',color=(0.4,0.4,0.4),label='Current context')
            plt.plot(self.X_next[:,0], self.context,'rv',label='$x_{next}$')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if plot_ideal:
                assert ideal_c is not None, 'Missing argument: ideal_c'
                assert ideal_x is not None, 'Missing argument: ideal_x'
                plt.plot(ideal_x,ideal_c,'--',color=(1,0,0,0.5),label='True optimum')  
            plt.legend()
        elif projection == '3d':
            ax = fig.gca(projection='3d')
            ax.scatter(self.X_sample[:,0], self.X_sample[:,1], self.Y_sample,label='Data')
            
            ax.plot_surface(X,C,Z,rstride=8, cstride=8, alpha=0.3,label='Posterior mean')
            cset = ax.contour(X,C,Z, zdir='z', offset=np.min(Z)-1, cmap=cm.coolwarm)
            ax.plot(action_space.reshape(self.grid_size,), C_now.reshape(self.grid_size,), mu1.reshape(self.grid_size,),color=(0,0,0.7))
            ax.plot(action_space.reshape(self.grid_size,), C_now.reshape(self.grid_size,), 0*mu1.reshape(self.grid_size,),'--',color=(0.4,0.4,0.4))
            ax.scatter(self.X_next[:,0], self.context, 0,marker="v", color='r')
            if plot_ideal:
                assert ideal_c is not None, 'Missing argument: ideal_c'
                assert ideal_x is not None, 'Missing argument: ideal_x'
                plt.plot(ideal_x,ideal_c,'--',color=(1,0,0,0.5),zdir='z', zs=0,label='True optimum')  
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
            #plt.legend()
        plt.show()
        if fig_name:
            fig.savefig(fig_name+'.pdf',bbox_inches='tight') 
  
        
class constrained_bayesian_optimization:
    '''
    Constrained Bayesian Optimization Class:
    
            max_X {f(X)|c(X)>=0}
    
    Use the same class for the objective function and the constraints. 
    
    Args:
        X_sample - Action X used to build the statistical model
        Y_sample - f(X)
        bounds - for the action space 
        kernal - GPy.kern
        optimize - optimize hyperparameters? True/False
        
    Requires GPy package: https://sheffieldml.github.io/GPy/
    written by: Dinesh Krishnamoorthy, July 2020
    '''
    
    def __init__(self,X_sample,Y_sample,bounds,kernel,
                            mf = None,
                            X_grid = np.linspace(0,1,100),
                            obj_fun = None):
        self.X_sample = X_sample
        self.Y_sample = Y_sample
        self.bounds = bounds
        self.kernel = kernel
        self.X_grid = X_grid
        self.grid_size = self.X_grid.shape[0]
        self.nX = self.X_sample.shape[1] 
        self.mf = mf
        
    def fit_gp(self):
        if self.mf is not None:
            self.m = GPy.models.GPRegression(self.X_sample,self.Y_sample,self.kernel,mean_function = self.mf)
        else:
            self.m = GPy.models.GPRegression(self.X_sample,self.Y_sample,self.kernel)
        return self

    def optimize_fit(self):
        return self.m.optimize()
    
    def query_next(self,acquisition='EI',constraint = None):
        self.acquisition = acquisition
        self.constraint = constraint
        nX = self.X_sample.shape[1] 
        
        min_val = -1e-5
        min_x = self.X_sample[-1,:]
        n_restarts=25
        
        if constraint is not None:
            self.PF_grid(constraint)   
        else:
            self.PF_grid = 1 

        def min_obj(X,self):

            if self.constraint is not None:
                PF = self.PF(self.constraint,np.array(X).reshape(-1,1))
            else:
                PF = 1

            if self.acquisition=='LCB' or self.acquisition=='UCB' :
                alpha = self.LCB(X)
            elif self.acquisition == 'EI':
                alpha = -self.EI(X)
            elif self.acquisition == 'PI':
                alpha = -self.PI(X)
            elif self.acquisition == 'expect':
                alpha = self.expect(X)
            return PF*alpha # standard acq_fn multiplied with the probability of feasibility

        if self.acquisition == 'TS':  # Thompson sampling
            self.posterior_sample = self.m.posterior_samples_f(self.X_grid,full_cov=True,size = 1).reshape(-1,nX)
            self.min_index = np.argmin(self.PF_grid*self.posterior_sample)
            
            self.X_next = self.X_grid[self.min_index].reshape(-1, nX) 
            self.min_val = np.min(self.posterior_sample)
        else:
            # Find the best optimum by starting from n_restart different random points.
            for x0 in np.random.uniform(self.bounds[:, 0].T, self.bounds[:, 1].T,
                                        size=(n_restarts, nX)):
                res = minimize(min_obj, x0=x0, args = (self),
                            bounds=self.bounds, method='L-BFGS-B')        
                if res.fun < min_val:
                    min_val = res.fun[0]
                    min_x = res.x         
            
            self.X_next =  min_x.reshape(-1, nX)
            self.min_val = min_val
        return self
    
    

    def LCB(self,X):
        mu, sigma = self.m.predict(X.reshape(-1,self.nX))
        return mu - 2*sigma
    
    def EI(self,X,xi=0.01):
        mu, sigma = self.m.predict(X.reshape(-1,self.nX))
        mu_sample,si = self.m.predict(self.X_sample)
        f_best = np.max(-mu_sample) # incumbent
        with np.errstate(divide='warn'):
            imp = -mu - f_best - xi
            Z = imp/sigma
            EI = (imp*norm.cdf(Z) + sigma*norm.pdf(Z))
            EI[sigma == 0.0] = 0.0
        return EI

    def PI(self,X,xi=0.01):
        mu, sigma = self.m.predict(X.reshape(-1,self.nX))
        mu_sample,si = self.m.predict(self.X_sample)
        f_best = np.max(-mu_sample) # incumbent
        with np.errstate(divide='warn'):
            imp = -mu - f_best - xi
            Z = imp/sigma
            PI = norm.cdf(Z)
            PI[sigma == 0.0] = 0.0
        return PI
    
    def expect(self,X):
        mu, sigma = self.m.predict(X.reshape(-1,self.nX))
        return mu

    def PF(self,constraint,X):
        '''
        Probability of feasibility evaluated at X
        '''
        mu, sigma = constraint.m.predict(X)
        return norm.cdf(0,-mu,sigma) # Probability of feasibility

    def PF_grid(self,constraint):
        '''
        Probability of feasibility function
        '''
        mu, sigma = constraint.m.predict(self.X_grid)
        self.PF_grid = norm.cdf(0,-mu,sigma) # Probability of feasibility
        return self

    def acq(self):
        if self.acquisition=='LCB' or self.acquisition=='UCB':
            self.acq_fn = self.LCB(self.X_grid)
        if self.acquisition=='EI':
            self.acq_fn = -self.EI(self.X_grid)
        if self.acquisition=='PI':
            self.acq_fn = -self.PI(self.X_grid)
        if self.acquisition == 'TS':
            self.acq_fn = self.posterior_sample
        if self.acquisition == 'expect':
            self.acq_fn = self.expect(self.X_grid)
        return self

    def plot(self,plot_acq = True,plot_ideal=False,fig_name=''):
        assert self.X_sample.shape[1]==1, "X dimension must be 1 for this function"
        mu, sigma = self.m.predict(self.X_grid)

        if plot_acq:
            self.acq()
            fig = plt.figure(constrained_layout=True)
            spec = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)
            ax1 = fig.add_subplot(spec[0:2, 0])
            ax2 = fig.add_subplot(spec[2, 0])
            ax1.fill_between(self.X_grid.ravel(), 
                        mu.ravel() + 2 * sigma.ravel(), 
                        mu.ravel() - 2 * sigma.ravel(), 
                        alpha=0.1,label='Confidence') 
            ax1.plot(self.X_grid,mu+2*sigma,color=(0,0.4,0.7,0.1),linewidth=0.3)
            ax1.plot(self.X_grid,mu-2*sigma,color=(0,0.4,0.7,0.1),linewidth=0.3)
            ax1.plot(self.X_grid,mu,color=(0,0.4,0.7),linewidth=2.4,label='Mean')
            ax1.plot(self.X_sample,self.Y_sample,'kx',label='Data')
            if plot_ideal:
                ax1.plot(self.X_grid,self.obj_fun(self.X_grid),'--',color=(0.6,0.6,0.6),linewidth=1,label='Ground truth')
        
            
            ax2.plot(self.X_grid,self.acq_fn*self.PF_grid,color = (0,0.7,0,0.8),linewidth=2,label='constrained')
            if self.constraint is not None:
                ax2.plot(self.X_grid,self.acq_fn,'--',color = (0,0.7,0,0.4),linewidth=2,label='unconstrained')
            ax2.plot(self.X_next,self.min_val,'ro',label='$x_{next}$')
            ax2.fill_between(self.X_grid.ravel(), 
                        np.max(self.acq_fn)+0*self.X_grid.ravel(), 
                        self.acq_fn.ravel()*self.PF_grid.ravel(),color='g', 
                        alpha=0.1)
            ax1.set_xlabel('$x$')
            ax1.set_ylabel('Objective $f(x)$')
            ax2.set_ylabel('Acqusition fn')
            ax2.set_xlabel('$x$')
            ax1.legend()
            ax2.legend()
            plt.show()
        else:
            plt.fill_between(self.X_grid.ravel(), 
                        mu.ravel() + 2 * sigma.ravel(), 
                        mu.ravel() - 2 * sigma.ravel(), 
                        alpha=0.1,label='Confidence') 
            plt.plot(self.X_grid,mu+2*sigma,color=(0,0.4,0.7,0.1),linewidth=0.3)
            plt.plot(self.X_grid,mu-2*sigma,color=(0,0.4,0.7,0.1),linewidth=0.3)
            plt.plot(self.X_grid,mu,color=(0,0.4,0.7),linewidth=2.4,label='Mean')
            plt.plot(self.X_sample,self.Y_sample,'kx',label='Data')
            if plot_ideal:
                plt.plot(self.X_grid,self.obj_fun(self.X_grid),'--',color=(0.6,0.6,0.6),linewidth=1,label='Ground truth')
            plt.xlabel('$x$')
            plt.ylabel('Objective $f(x)$')
            plt.legend()
            plt.show()
        if fig_name:
            fig.savefig(fig_name+'.pdf',bbox_inches='tight')

    def plot_constraint(self,constraint=None,C_sample=None,fig_name=''):
        assert constraint is not None, 'Constraint GP model missing'
        assert C_sample is not None, 'Constraint data missing'

        mu, sigma = constraint.m.predict(self.X_grid)

        fig = plt.figure(constrained_layout=True)
        spec = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)
        ax1 = fig.add_subplot(spec[0:2, 0])
        ax2 = fig.add_subplot(spec[2, 0])
        ax1.fill_between(self.X_grid.ravel(), 
                    mu.ravel() + 2 * sigma.ravel(), 
                    mu.ravel() - 2 * sigma.ravel(), 
                    color=(0.7,0,0,0.1),label='Confidence') 
        ax1.plot(self.X_grid,mu+2*sigma,color=(0,0.4,0.7,0.1),linewidth=0.3)
        ax1.plot(self.X_grid,mu-2*sigma,color=(0,0.4,0.7,0.1),linewidth=0.3)
        ax1.plot(self.X_sample,C_sample,'kx',label='Constraint Data')
        ax1.plot(self.X_grid,mu,color=(0.7,0.0,0.0),linewidth=2.4,label='Constraint Mean')
        ax2.plot(self.X_grid,self.PF_grid,color = (0.2,0.2,0.2,0.8),linewidth=2,label='Probability of Feasibility')
        ax1.set_xlabel('$x$')
        ax1.set_ylabel('Constraint $c(x)\geq0$')
        ax2.set_ylabel('Prob. Feas.')
        ax2.set_xlabel('$x$')
        ax1.legend()
        ax2.legend()
        plt.legend()
        plt.show()
        if fig_name:
            fig.savefig(fig_name+'.pdf',bbox_inches='tight')


class exploit:
    def __init__(self,model,context,bounds):
        self.model = model
        self.context = context
        self.bounds = bounds
        
    
    def compute_optimum(self):
        nC = self.context.shape[1]    
        nU = self.bounds.shape[0] 
        
        min_val = 100000000
        min_x = None
        n_restarts=25
        
        def min_obj(X,self):
            mu, sigma = self.model.predict(np.array([X,self.context]).T)
            return mu
        
        # Find the best optimum by starting from n_restart different random points.
        for x0 in np.random.uniform(self.bounds[:, 0].T, self.bounds[:, 1].T,
                                    size=(n_restarts, nU)):
            res = minimize(min_obj, x0=x0, args = (self),
                           bounds=self.bounds, method='L-BFGS-B')        
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x           

        U_next = min_x.reshape(-1, nU)
        return U_next
    
    def compute_optimum_1(self):
        nC = self.context.shape[1]    
        nU = self.bounds.shape[0] 

        U_grid = np.linspace(self.bounds[:, 0].T, self.bounds[:, 1].T, 100).reshape(-1,nU)
        C_grid = self.context*np.ones([100,1])
        testX = np.concatenate((U_grid,C_grid),axis=1)

        mu, sigma = self.model.predict(testX)
        GP_mean = mu
        min_index = np.argmin(GP_mean)
        U_next = U_grid[min_index]

        return U_next
        
# ---- OLD: DELETE everything below -----
class contextual_bayesian_optimization_1D:
    # Written by: Dinesh Krishnamoorthy, July 2020

    def __init__(self,X_sample,Y_sample,context,obj_fun,bounds,mf,
                          kernel, grid_size= 100,optimize = False, exploit = False):
        self.X_sample = X_sample
        self.Y_sample = Y_sample
        self.context = context  # New context
        self.obj_fun = obj_fun
        self.bounds = bounds
        self.kernel = kernel
        self.mf = mf
        self.grid_size = grid_size
        self.optimize = optimize  # not used for anything. Should remove it in the final version
        self.exploit = False
        
    def fit_gp(self):
        self.m = GPy.models.GPRegression(self.X_sample,self.Y_sample,self.kernel,mean_function = self.mf)
        return self
    
    def optimize_fit(self):
        return self.m.optimize()
    
    def get_dim(self):
        nC = self.context.shape[1]    
        nX = self.X_sample.shape[1] 
        nU = nX - nC
        return nU, nC, nX
    
    def extract_action_space(self):
        nU,nC,nX = self.get_dim()
        return np.array(self.X_sample[:,0:nU+1]).reshape(-1,nU)
        
    def draw_sample_posterior(self):
        nU,nC,nX = self.get_dim()
        U_grid = np.linspace(self.bounds[:, 0].T, self.bounds[:, 1].T, self.grid_size).reshape(-1,nU)
        C_grid = self.context*np.ones([self.grid_size,1]).reshape(-1,nC)
        testX = np.concatenate((U_grid,C_grid),axis=1)
        posterior_sample = self.m.posterior_samples_f(testX,full_cov=True,size = 1).reshape(-1,1)
        return posterior_sample,Ugrid,Cgrid
    
    def query_next_TS(self): # Thompson Sampling
        nU,nC,nX = self.get_dim()
        U_grid = np.linspace(self.bounds[:, 0].T, self.bounds[:, 1].T, self.grid_size).reshape(-1,nU)
        C_grid = self.context*np.ones([self.grid_size,1])#.reshape(-1,nC)
        testX = np.concatenate((U_grid,C_grid),axis=1)
        posterior_sample = self.m.posterior_samples_f(testX,full_cov=True,size = 1).reshape(-1,1)
        
        min_index = np.argmin(posterior_sample)
        U_next = U_grid[min_index]
        self.X_next = np.concatenate((U_next.reshape(-1,1),self.context),axis=1) #np.array([U_next,self.context]).ravel()
        return self
    
    def query_next_UCB(self): #Upper confidence bound
        nU,nC,nX = self.get_dim()
        U_grid = np.linspace(self.bounds[:, 0].T, self.bounds[:, 1].T, self.grid_size).reshape(-1,nU)
        C_grid = self.context*np.ones([self.grid_size,1])#.reshape(-1,nC)
        testX = np.concatenate((U_grid,C_grid),axis=1)
        mu, sigma = self.m.predict(testX)
        GP_UCB = mu - 2*sigma
        min_index = np.argmin(GP_UCB)
        U_next = U_grid[min_index]
        self.X_next = np.concatenate((U_next.reshape(-1,1),self.context),axis=1) #np.array([U_next,self.context]).ravel()
        return self
    
    def query_next(self):
        nU,nC,nX = self.get_dim()
        
        min_val = 100000000
        min_x = None
        n_restarts=25

        def min_obj(X,self):
            mu, sigma = self.m.predict(np.array([X,self.context]).T)
            if self.exploit:
                alpha = mu
            else:
                alpha = mu - 2*sigma
            return alpha

        # Find the best optimum by starting from n_restart different random points.
        for x0 in np.random.uniform(self.bounds[:, 0].T, self.bounds[:, 1].T,
                                    size=(n_restarts, nU)):
            res = minimize(min_obj, x0=x0, args = (self),
                           bounds=self.bounds, method='L-BFGS-B')        
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x           

        U_next = min_x.reshape(-1, nU)

        
        self.X_next = np.array([U_next,self.context]).ravel()
        return self
    
    def observe_obj(self):
        Y_next = self.obj_fun([self.X_next]).tolist()
        X_sample = np.vstack((self.X_sample, self.X_next))
        Y_sample = np.vstack((self.Y_sample, Y_next))
        
        return X_sample, Y_sample
    
    def observe_obj_init(self):
        Y_next = self.obj_fun([self.X_next]).tolist()
        X_sample = np.array((self.X_next))
        Y_sample = np.array((Y_next))
        
        return X_sample, Y_sample

       
class constrained_bayesian_optimization_old:
    '''
    Constrained Bayesian Optimization Class:
    
            max_X {f(X)|c(X)>=0}
    
    Use the same class for the objective function and the constraints. 
    
    Args:
        X_sample - Action X used to build the statistical model
        Y_sample - f(X)
        my_fun - objective/constraint function 
        bounds - for the action space 
        kernal - GPy.kern
        optimize - optimize hyperparameters? True/False
        
    Requires GPy package: https://sheffieldml.github.io/GPy/
    written by: Dinesh Krishnamoorthy, July 2020
    '''
    
    def __init__(self,X_sample,Y_sample,my_fun,bounds,kernel,optimize=False):
        self.X_sample = X_sample
        self.Y_sample = Y_sample
        self.my_fun = my_fun # cost or constraint function
        self.bounds = bounds
        self.kernel = kernel
        self.optimize = optimize
        
    def fit_gp(self):
        self.m = GPy.models.GPRegression(self.X_sample,self.Y_sample,self.kernel)
        if self.optimize:
            self.m.optimize()
        return self
    
    def query_next(self,constraint,acquisition='UCB'):
        if acquisition == 'UCB':
            return self.query_next_UCB(constraint)
        else:
            return self.query_next_EI(constraint)
    
    def EI(self,constraint,X,xi=0.01):
        '''
        Constrained Expected improvement activation function
        '''
        mu, sigma = self.m.predict(X)
        mu_sample = self.m.predict(self.X_sample)

        sigma = sigma.reshape(-1, 1)
        mu_incumbent = np.max(mu_sample)  # f_best
        
        mu_c, sigma_c = constraint.m.predict(X)
        PF = norm.cdf(0,mu_c,sigma_c) # Probability of feasibility

        with np.errstate(divide='warn'):
            imp = mu - mu_incumbent - xi # \mu(x) - f_best
            Z = imp / sigma  # (\mu(x) - f_best)/sigma(x)
            ei = PF*(imp * norm.cdf(Z) + sigma * norm.pdf(Z)) 
            ei[sigma == 0.0] = 0.0   
        return ei
    
    def query_next_EI(self,constraint):
        '''
        Query next point using the constrained EI activation function
        '''
        nX = self.X_sample.shape[1]
        min_val = 1
        min_x = None
        n_restarts=25

        def min_obj(X,self,constraint):
            return -self.EI(constraint,X.reshape(-1,1))

        # Find the best optimum by starting from n_restart different random points.
        for x0 in np.random.uniform(self.bounds[:, 0].T, self.bounds[:, 1].T,
                                    size=(n_restarts, nX)):
            res = minimize(min_obj, x0=x0, args = (self,constraint),
                           bounds=self.bounds, method='L-BFGS-B')        
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x           

        self.X_next = min_x.reshape(-1, nX)
        return self
    
    def UCB(self):
        nX = self.X_sample.shape[1]
        testX = np.linspace(self.bounds[:, 0].T, self.bounds[:, 1].T, 100).reshape(-1,nX)
        
        mu, sigma = self.m.predict(testX)
        return mu + 2*sigma
    
    def PF(self,constraint):
        nX = self.X_sample.shape[1]
        testX = np.linspace(self.bounds[:, 0].T, self.bounds[:, 1].T, 100).reshape(-1,nX)
        
        mu, sigma = constraint.m.predict(testX)
        PF = norm.cdf(0,-mu,sigma) # Probability of feasibility
        return PF
    
    def query_next_UCB(self,constraint):
        '''
        Compute the next query point using GP-UCB as the acquisition function
        '''
        nX = self.X_sample.shape[1]
        testX = np.linspace(self.bounds[:, 0].T, self.bounds[:, 1].T, 100).reshape(-1,nX)
        
        GP_UCB = self.UCB()
        
        PF = self.PF(constraint)
        
        max_index = np.argmax(PF*GP_UCB)
        self.X_next = testX[max_index]
        return self
    
    
    def observe_obj(self):
        '''
        Implement next query point, and observe the objective function.
        Returns the updated dataset X_sample and Y_sample. 
        '''
        Y_next = self.my_fun(np.array([self.X_next])).reshape(-1,1)
        X_sample = np.vstack((self.X_sample, self.X_next))
        Y_sample = np.vstack((self.Y_sample, Y_next))
        return X_sample, Y_sample
    
    def observe_constraint(self,X_next):
        '''
        Implement next query point, and observe the constraint.
        Returns the updated dataset c_sample. 
        '''
        c_next = self.my_fun(np.array([X_next])).reshape(-1,1)
        return np.vstack((self.Y_sample, c_next))


