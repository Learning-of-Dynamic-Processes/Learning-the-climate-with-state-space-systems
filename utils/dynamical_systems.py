from scipy.integrate import solve_ivp
import numpy as np

def iter_rk4(y, t, h, f, fargs=None):
    """
    Runge-Kutta 4th order (vectorized).
    y: (batch, n_dim)
    f: function f(t, y) that works with (batch, n_dim)
    """
    if fargs is None:
        k1 = f(t, y)                         # (batch, n_dim)
        k2 = f(t + 0.5*h, y + 0.5*h*k1)
        k3 = f(t + 0.5*h, y + 0.5*h*k2)
        k4 = f(t + h, y + h*k3)
    else:
        k1 = f(t, y, fargs)
        k2 = f(t + 0.5*h, y + 0.5*h*k1, fargs)
        k3 = f(t + 0.5*h, y + 0.5*h*k2, fargs)
        k4 = f(t + h, y + h*k3, fargs)

    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)  # (batch, n_dim)

def generate_points(batch, y0, sigma = 1, distribution = 'uniform', seed = None):
        rng = np.random.default_rng(seed)

        n_dim = y0.shape[0]

        y = np.array([y0 for _ in range(batch)])

        if distribution == "gaussian":
            y += rng.normal(0, sigma, (batch, n_dim))
            
        elif distribution == "uniform":
            y += rng.uniform(-sigma, sigma, (batch, n_dim))

        else:
            print('No such distribution')

        return y
    

class DS():
    def __init__(self, phi, n_dim, step = 1):
        self.phi = phi
        self.n_dim = n_dim
        self.step = step

    def integrate(self, y0, T = 1, t0 = 0):
        """
        Integrates forward over a batch for T time steps starting from t0
        y0 : (batch, n_dim)
        """
        batch = y0.shape[0]
        y_sol = np.zeros((batch, T, self.n_dim))
        y_step = y0 # (batch, n_dim)
        t_step = t0

        for t in range(T):
            y_step = self.phi(y_step, t_step)
            t_step += self.step
            y_sol[:,t,:] = y_step
        
        return y_sol


class DS_dudt(DS):
    def __init__(self, du_dt, n_dim, step, method='RK4'):
        self.du_dt = du_dt
        self.method = method

        if self.method == 'RK4':
            def phi(y,t):
                return iter_rk4(y, t, step, du_dt)
        else:
            def phi(y,t):
                """
                NOTE: solve_ivp not really batch-friendly
                y : (batch, n_dim)
                """
                
                y_new = []
                batch = y.shape[0]

                for i in range(batch):
                    sol = solve_ivp(self.du_dt, (t, t + step), 
                                    y[i], method=self.method, 
                                    t_eval=[t + step])
                    y_new.append(sol.y[:,-1])
                return np.stack(y_new, axis=0)

        super().__init__(phi, n_dim, step)

lor_args = (10, 8/3, 28)

class lorenz(DS_dudt):
    def __init__(self, step = 0.02, params = lor_args, method = 'RK4'):
        
        def lorenz_du_dt(t, Z):
            Z = np.atleast_2d(Z) # ensures shape (batch, 3)
            u, v, w = Z.T # (batch, 1)
            sig, beta, rho = params
            
            up = -sig*(u - v)
            vp = rho*u - v - u*w
            wp = -beta*w + u*v
            
            return np.stack([up, vp, wp], axis = 1) # (batch, 3)
        
        super().__init__(du_dt = lorenz_du_dt,
                                 n_dim = 3,
                                 step = step,
                                 method = method)
    