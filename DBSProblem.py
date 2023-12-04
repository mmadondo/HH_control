import os.path

import torch
import torch.nn as nn
from torch.nn.functional import pad
from utils import normpdf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import tikzplotlib
from AbstractOCProblem import AbstractOCProblem

class DBSProblem(AbstractOCProblem):
    """Definition of DBS Problem"""
    def __init__(self, z_star, dbs_params = [1.0, 380.0, 36.0, 0.3, 115.0, -12.0, 10.613]):
        super().__init__()

        # initial values for membrane potential/voltage and gating variables 
        Vm = 0.0 
        m = 0.0
        h = 0.0
        n = 0.0
        # membrane capacitance (per unit area)
        self.dbs_params = dbs_params
        self.Cm = dbs_params[0]

        # ionic conductance for sodium, potassium, and leakage channels [mS/cm^2]
        self.gNa = dbs_params[1] # pathological condition: 380, normal: 120
        self.gK = dbs_params[2] # pathological condition: 18.0, normal: 36
        self.gL = dbs_params[3]

        # reversal potential (mV)
        self.ENa = dbs_params[4]
        self.EK = dbs_params[5]
        self.EL = dbs_params[6]

        self.d = 4
        self.t = 0.0
        self.T = 30.0
        self.alpha = 0.5 # electrode impedance
        self.e1 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        self.xInit = torch.tensor([[Vm, m, n, h]], dtype=torch.float64)

        self.z_star = z_star
        self.xtarget = z_star[-1].view(1,-1)

    def _get_name(self):
        return 'DBSProblem'

    def __str__(self):
        s = "DBSProblem(d = {:},\n xInit={:},\n xtarget={:}\n params={:})".format(self.d, self.xInit, self.xtarget, self.dbs_params)
        return s

    def x_init(self,nex):
        return self.xInit + torch.cat([10*torch.randn(nex, self.d - 3), torch.zeros(nex, self.d-1)], dim=1) #torch.randn((nex, self.d))

    def f(self, s, z, u):
        # membrane potential/voltage and gating variables
        V = z[:, 0]
        m = z[:, 1]
        n = z[:, 2]
        h = z[:, 3]

        # z_0 at t+1: compute channel currents and add control
        i_Na = self.gNa * m**3 * h * (V - self.ENa)
        i_K = self.gK * n**4 * (V - self.EK)
        i_L = self.gL * (V - self.EL)
        # compute ion current
        I_c = i_Na + i_K + i_L

        # gating variables dynamics
        dz0 = (1/self.Cm)*(u - I_c.unsqueeze(1)).squeeze()
        dz1 = self.alpha_m(V)*(1-m) - self.beta_m(V)*m
        dz2 = self.alpha_n(V)*(1-n) - self.beta_n(V)*n
        dz3 = self.alpha_h(V)*(1-h) - self.beta_h(V)*h

        f = torch.cat((dz0.view(-1,1), dz1.view(-1,1), dz2.view(-1,1), dz3.view(-1,1)), dim=1)
        return f 

    def L(self, t, x, u, z_star, Q=200.0):
        # running cost
        if z_star is None:
            # z_star = x.clone()
            z_star = self.z_star[2]
        nex = x.shape[0]
        # idx = torch.linspace(self.t, self.T, nex+1, dtype=torch.int) # select corresponding samples in z_ref
        return self.alpha * torch.norm(u, dim=1, keepdim=True)**2 + Q*0.5*torch.norm(x - z_star, dim=1, keepdim=True)**2

    def g(self, z, alphaG=1.0):
        # terminal condition for value function
        res = z - self.xtarget
        G   = alphaG * 0.5 * torch.norm(res, dim=1, keepdim=True)**2
        return G, alphaG * res

    def u_star(self,s,z,p,clamp=False):
        u = - (1/(2*self.alpha) * p@self.e1).view(-1, 1)
        return u

    def calcGradpH(self, s, z, p, u = None):
        if u is None:
            u = self.u_star(s, z, p)
        return - self.f(s, z, u)

    def Hamiltonian(self, s, z, p, z_star, clamp=False, M=None):
        u =  self.u_star(s,z,p)
        H = - torch.sum(p*self.f(s,z,u),dim=1,keepdim=True) - self.L(s,z,u, z_star)
        gradpH = self.calcGradpH(s,z,p, u)
        return H, gradpH

    # Potassium ion-channel rate functions
    def alpha_n(self, V):
        """Computes the voltage-dependant alpha term for a gating variable n"""  
        return 0.01* (10.0 - V) / (torch.exp(1.0 - 0.1*V) - 1.0)

    def beta_n(self, V):
        """Computes the voltage-dependant beta term for a gating variable n"""
        return 0.125 * torch.exp(-V/80.0)

    # Sodium ion-channel rate functions
    def alpha_m(self, V):
        """Computes the voltage-dependant alpha term for a gating variable m"""
        return 0.1 * (25.0-V) / (torch.exp(0.1 * (25.0-V)) - 1.0)

    def beta_m(self, V):
        """Computes the voltage-dependant beta term for a gating variable m"""
        return 4.0 * torch.exp(-V/18.0)

    def alpha_h(self, V):
        """Computes the voltage-dependant alpha term for a gating variable h"""
        return 0.07 * torch.exp(-V/20.0)

    def beta_h(self, V):
        """Computes the voltage-dependant beta term for a gating variable h"""
        return 1.0 / (torch.exp(3.0 - 0.1*V) + 1.0)

    def render(self, s, z, gradPhiz, J, path):
        # Single plot code
        u = 0

        # ionic conductance for sodium, potassium, and leakage channels [mS/cm^2]
        gNa = self.gNa
        gK = self.gK
        gL = self.gL

        # reversal potential (mV)
        ENa = self.ENa
        EK = self.EK
        EL = self.EL

        controls = []
        voltage = []
        mgate = []
        ngate = []
        hgate=[]
        Na_current = []
        K_current = []
        L_current = []
        
        for i in range(len(z)):
            V = z[i][0][0]
            m = z[i].detach().numpy()[0][1]
            n = z[i].detach().numpy()[0][2]
            h = z[i].detach().numpy()[0][3]

            i_Na = gNa * m**3 * h * (V - ENa)
            i_K = gK * n**4 * (V - EK)
            i_L = gL * (V - EL)
            
            voltage.append(V.detach().numpy())
            mgate.append(m)
            ngate.append(n)
            hgate.append(h)
            Na_current.append(i_Na.detach().numpy())
            K_current.append(i_K.detach().numpy())
            L_current.append(i_L.detach().numpy())

            u = self.u_star(s, z, gradPhiz[i][0,:])
            controls.append(u.detach().numpy()[0][0])
        
        # plot membrane potential Vm
        fig = plt.figure()
        plt.plot(s, voltage, label='NN solution', c='b')
        plt.xlabel(r'Time ($ms$)')
        plt.ylabel(r'Voltage ($mV$)')
        # plt.axhline(y=self.xtarget[0][0].item(), label=r'target $V_m$', color='k', linestyle='dashed')
        plt.legend()
        plt.savefig(path+'voltage_time.pdf', dpi=1000)
        path_ions =  "./experiments/data/ion_channels"
        if not os.path.exists(path_ions):
            os.makedirs(path_ions)
        np.save(f"{path_ions}/{path[path.index('_m'):]}voltage.npy", np.array(voltage))
        plt.close()

        # plot controls
        fig = plt.figure()
        plt.plot(s, controls, label='NN control (stimulus)')
        plt.xlabel(r'Time ($ms$)')
        plt.ylabel(r'Control ($\mu A/cm^2$)')
        # plt.title('Cost = {:.3f}'.format(J))
        plt.legend()
        plt.savefig(path+'controls.pdf', dpi=1000)
        path_ctrl = "./experiments/data/controls"
        if not os.path.exists(path_ctrl):
            os.makedirs(path_ctrl)
        np.save(f"{path_ctrl}/{path[path.index('_m'):]}controls.npy", np.array(controls))
        plt.close()

        # plot gating variables
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('time step (t)')
        ax.set_ylabel('Gating variable')
        ax.plot(s, mgate, 'k', label='m')
        ax.plot(s, ngate, 'r', label='n')
        ax.plot(s, hgate, '#1f77b4', label='h')
        plt.legend()
        plt.savefig(path+'gating_variables.pdf', dpi=1000)
        path_gatv =  "./experiments/data/gate_variables"
        if not os.path.exists(path_gatv):
            os.makedirs(path_gatv)
        np.save(f"{path_gatv}/{path[path.index('_m'):]}m_gate.npy", np.array(mgate))
        np.save(f"{path_gatv}/{path[path.index('_m'):]}n_gate.npy", np.array(ngate))
        np.save(f"{path_gatv}/{path[path.index('_m'):]}h_gate.npy", np.array(hgate))
        plt.close()

        # plot ion channels
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel(r'Time ($ms$)')
        ax.set_ylabel('Ion Channel Current')
        ax.plot(s, Na_current, 'k', label='Na')
        ax.plot(s, K_current, 'r', label='K')
        ax.plot(s, L_current, 'g', label='Leak')
        plt.legend(loc='best')
        plt.savefig(path+'ion_channels.pdf', dpi=1000)

        np.save(f"./experiments/data/ion_channels/{path[path.index('_m'):]}Na_current.npy", np.array(Na_current))
        np.save(f"./experiments/data/ion_channels/{path[path.index('_m'):]}K_current.npy", np.array(K_current))
        np.save(f"./experiments/data/ion_channels/{path[path.index('_m'):]}L_current.npy", np.array(L_current))
        plt.close('all')

    def evolv_f(self, z, nt=3000):
        zall = z
        ds = 30/nt
        for i in range(nt):
            z = z + ds*self.f(i*ds, zall[i:i+1], 0.0)
            zall = torch.cat((zall, z), 0)
        return zall

if  __name__ == '__main__':
    d = 4
    z_true = torch.from_numpy(np.load("./target_solution/target_sol.npz")["target_sol"])
    z_star = torch.cat((z_true[::int(z_true.shape[0]/100)], z_true[-1:]), 0)
    prob = DBSProblem(z_star=z_star)

    print(prob)
    nex = 10
    s = 2.0

    torch.set_default_dtype(torch.float64)
    # test evolutuion of f
    nt = [1400, 1450, 1500]
    for i in nt:
        zall = prob.evolv_f(prob.xInit, i)
        plt.plot(range(i+1),zall[:,0])
        plt.title(f"{i}")
        plt.show()
    plt.plot(range(nt+1),zall[:,1],range(nt+1),zall[:,2],range(nt+1),zall[:,3]), plt.show()
    plt.show()
    # test problem setup
    z = torch.Tensor([[-50.0, 0.5, 0.5, 0]]) + torch.randn((nex,prob.d))
    # xtarget = torch.Tensor([2, 2, 2]).unsqueeze(0)
    # xtarget = pad(xtarget, [0, d - 3, 0, 0], value=0)
    p = torch.randn_like(z)
    tt = prob.test_u_star(s,z,p)
    print("feedback form test error: ", tt)
    prob.test_Hamiltonian(s,z,p)
    prob.test_g(z)
