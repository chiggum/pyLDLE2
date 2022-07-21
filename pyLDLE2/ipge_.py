import numpy as np
from scipy.stats.distributions import chi2

# Inner product of gradient of eigenfunctions
class IPGE:
    def __init__(self, debug=False):
        self.debug = debug
        self.Gtilde = None
        self.Atilde = None

    def fem(self, opts, print_prop = 0.25):
        epsilon = opts['epsilon']
        p = opts['p']
        d = opts['d']

        n, N = opts['phi'].shape
        print_freq = np.int(n*print_prop)

        # Compute G
        t = 0.5*((epsilon**2)/chi2.ppf(p, df=d))
        #t = 0.5*(np.dot(epsilon.T,epsilon))/chi2.ppf(p, df=d)
        G = np.exp(-opts['neigh_dist'][:,1:opts['k']-1]**2/(4*t))
        G = G/(np.sum(G,1)[:,np.newaxis])

        # Compute Gtilde (Gtilde_k = (1/t_k)[G_{k1},...,G_{kn}])
        Gtilde = G/(2*t) # Helps in correcting issues at the boundary (see page 5  of http://math.gmu.edu/~berry/Publications/VaughnBerryAntil.pdf)
        # Gtilde = G

        Atilde=np.zeros((n,N,N))

        print('FEM for Atilde.')
        for k in range(n):
            if print_freq and np.mod(k,print_freq)==0:
                print('Atilde: %d points processed...' % k)
            U_k = opts['neigh_ind'][k,1:opts['k']-1]
            dphi_k = opts['phi'][U_k,:]-opts['phi'][k,:]
            Atilde[k,:,:] = np.dot(dphi_k.T, dphi_k*(Gtilde[k,:][:,np.newaxis]))

        print('Atilde: all points processed...')
        if self.debug:
            self.Gtilde = Gtilde
        self.Atilde = Atilde

    def compute_gradient_using_LLR(self, X, phi, d_e, U, t, d, print_prop = 0.25):
        n,p = X.shape
        print_freq = np.int(n*print_prop)
        N = phi.shape[1]
        grad_phi = np.zeros((n,N,p))

        #Uh = d_e < np.sqrt(4*t)
        Uh = d_e < 4*t
        G = np.power(4*t,-d/2)*np.exp(-d_e**2/(4*t))*Uh

        for k in range(n):
            if print_freq and np.mod(k,print_freq)==0:
                print('Gradient computation done for %d points...' % k)

            U_k = U[k,:]
            n_U_k = np.sum(U_k)
            X_k = X[U_k,:]
            X_k_ = X_k - np.mean(X_k, axis=0)[np.newaxis,:]

            Sigma_k = np.dot(X_k_.T, X_k_)/n_U_k
            if p == d:
                _, B_k = eigh(Sigma_k)
            else:
                _, B_k = eigsh(Sigma_k, k=d, which='LM')

            Uh_k = Uh[k,:]
            n_Uh_k = np.sum(Uh_k)
            Xh_k = X[Uh_k,:]
            XX_k = np.zeros((n_Uh_k,d+1))
            XX_k[:,0] = 1
            XX_k[:,1:] = np.dot(Xh_k - X[[k],:], B_k)

            WW_k = G[k,Uh_k][np.newaxis,:].T

            Y = phi[Uh_k,:]
            temp = np.dot(XX_k.T,WW_k*XX_k)
            #print(k)
            #print(XX_k)
            #print(WW_k)
            #print(temp)
            bhat_k = np.dot(np.linalg.inv(temp),np.dot(XX_k.T, WW_k*Y))

            grad_phi[k,:,:] = np.dot(B_k,bhat_k[1:,:]).T

        return grad_phi

    def compute_Atilde_LDLE_2(X, L, phi0, phi, lmbda0, lmbda, d_e, U, epsilon, p, d, autotune, print_prop = 0.25):
        n, N = phi.shape
        print_freq = np.int(n*print_prop)

        L = L.copy()
        L = L/(autotune.toarray()+1e-12)

        lmbda = lmbda.copy()
        lmbda = lmbda.reshape(1,N)
        Atilde=np.zeros((n,N,N))

        # For computing derivative at t=0
        for k in range(n):
            if print_freq and np.mod(k,print_freq)==0:
                print('Atilde: : %d points processed...' % k)
            dphi_k = phi-phi[k,:]
            Atilde[k,:,:] = -0.5*np.dot(dphi_k.T, dphi_k*(L[k,:][:,np.newaxis]))
        print('Atilde_k, Atilde_k: all points processed...')
        return None, Atilde

    def llr(self, X, phi, d_e, U, epsilon, p, d, print_prop = 0.25):
        n, N = phi.shape
        print_freq = np.int(n*print_prop)

        # t = 0.5*((epsilon**2)/chi2.ppf(p, df=d))
        t = 0.5*((epsilon**2)*chi2.ppf(p, df=d))
        grad_phi = compute_gradient_using_LLR(X, phi, d_e, U, t, d, print_prop = print_prop)
        Atilde=np.zeros((n,N,N))

        for k in range(n):
            if print_freq and np.mod(k,print_freq)==0:
                print('A_k, Atilde_k: %d points processed...' % k)

            grad_phi_k = grad_phi[k,:,:]
            Atilde[k,:,:] = np.dot(grad_phi_k, grad_phi_k.T)

        print('Atilde_k, Atilde_k: all points processed...')
        return grad_phi, Atilde

    def compute_Atilde_LDLE_3(X, L, phi0, phi, lmbda0, lmbda, d_e, U, epsilon, p, d, autotune, print_prop = 0.25):
        n, N = phi.shape
        print_freq = np.int(n*print_prop)
        Atilde=np.zeros((n,N,N))

        temp1 = np.dot(lmbda*phi, phi.T)
        temp1 = temp1 + np.dot(lmbda0*phi0, phi0.T)
        temp1 = temp1/(autotune.toarray()+1e-12)

        for k in range(n):
            if print_freq and np.mod(k,print_freq)==0:
                print('Atilde: : %d eigenvectors processed...' % k)

            dphi_k = phi-phi[k,:]
            Atilde[k,:,:] = -0.5*np.dot(dphi_k.T, dphi_k*(temp1[k,:][:,np.newaxis]))

        print('Atilde_k, Atilde_k: all points processed...')
        return None, Atilde