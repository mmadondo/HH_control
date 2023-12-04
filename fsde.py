import torch
import numpy as np

class fwdEuler(torch.nn.Module):

    def __init__(self,Phi,prob,t,T,nt,z_star):
        super().__init__()
        self.t=t
        self.T=T
        self.nt=nt
        self.Phi = Phi
        self.prob = prob
        self.z_star = z_star

    def __str__(self):
        return "%s(t=%1.2f, T=%1.2f, nt=%d)" % (self._get_name(), self.t, self.T, self.nt)


    def forward(self, x, t=None, T=None, nt=None):

        if t is None: t= self.t
        if T is None: T = self.T
        if nt is None: nt = self.nt

        s = torch.linspace(t, T, nt + 1)
        z = [x]
        Phizi, gradPhizi, _ = self.Phi(s[0], z[0], do_gradient=True)
        Phiz = [Phizi]
        gradPhiz = [gradPhizi]
        H, gradpH = self.prob.Hamiltonian(s[0], z[0], gradPhiz[0], self.z_star[0])
        for i in range(nt):
            ds = s[i+1]-s[i]
            z.append(z[i] - gradpH * ds)
            Phizi, gradPhizi, _ = self.Phi(s[i+1], z[i+1], do_gradient=True)
            Phiz.append(Phizi)
            gradPhiz.append(gradPhizi)
            H, gradpH = self.prob.Hamiltonian(s[i+1], z[i+1], gradPhiz[i+1], self.z_star[i+1])
        return s, z, None, Phiz, gradPhiz

class rk4(torch.nn.Module):

    def __init__(self,Phi,prob,t,T,nt,z_star):
        super().__init__()
        self.t=t
        self.T=T
        self.nt=nt
        self.Phi = Phi
        self.prob = prob
        self.z_star = z_star

    def __str__(self):
        return "%s(t=%1.2f, T=%1.2f, nt=%d)" % (self._get_name(), self.t, self.T, self.nt)


    def forward(self, x, t=None, T=None, nt=None):

        if t is None: t= self.t
        if T is None: T = self.T
        if nt is None: nt = self.nt

        s = torch.linspace(t, T, nt + 1)
        z = [x]
        Phizi, gradPhizi, _ = self.Phi(s[0], z[0], do_gradient=True)
        Phiz = [Phizi]
        gradPhiz = [gradPhizi]
        H, gradpH = self.prob.Hamiltonian(s[0], z[0], -gradPhiz[0], self.z_star[0])

        for i in range(nt):
            ds = s[i+1]-s[i]
            ztmp = z[i]
            # k1 = gradpH
            ztmp = ztmp - (ds/6.0)*gradpH
            
            Phizi, gradPhizi, _ = self.Phi(s[i]+ds/2, z[i]-gradpH*ds/2, do_gradient=True)
            H, gradpH = self.prob.Hamiltonian(s[i]+ds/2, z[i]-gradpH*ds/2, -gradPhizi, self.z_star[i])
            ztmp = ztmp - (2.0*ds/6.0)*gradpH
            
            Phizi, gradPhizi, _ = self.Phi(s[i]+ds/2, z[i]-gradpH*ds/2, do_gradient=True)
            H, gradpH = self.prob.Hamiltonian(s[i]+ds/2, z[i]-gradpH*ds/2, -gradPhizi, self.z_star[i])
            ztmp = ztmp- (2.0*ds/6.0)*gradpH
            
            Phizi, gradPhizi, _ = self.Phi(s[i]+ds, z[i]-gradpH*ds, do_gradient=True)
            H, gradpH = self.prob.Hamiltonian(s[i]+ds, z[i]-gradpH*ds, -gradPhizi, self.z_star[i])
            ztmp = ztmp- (1.0*ds/6.0)*gradpH
            
            # m = (k1+2*k2+2*k3+k4)/6
            
            z.append(ztmp)

            Phizi, gradPhizi, _ = self.Phi(s[i+1], z[i+1], do_gradient=True)
            Phiz.append(Phizi); gradPhiz.append(gradPhizi)

            H, gradpH = self.prob.Hamiltonian(s[i+1], z[i+1], -gradPhiz[i+1], self.z_star[i+1])

        return s, z, None, Phiz, gradPhiz


if __name__ == '__main__':
    sampler = fwdEuler(0.1,0.8,10)
    print(sampler)
    
    
"""
t=0
T=5
allh=[50,100,200,400]
z = 1
z1 = []
z2 = []
z3 = []
z4 = []
for i in range(0,3):
    h=allh[i]
    s=np.linspace(t, T, h+1)
    ds=(T-t)/h
    for i in range(h):
        ztmp = z
        ztmp = ztmp + (ds/6.0)*np.exp(s[i])
        ztmp = ztmp+ (2.0*ds/6.0)*np.exp(s[i]+ds/2)
        ztmp = ztmp+ (2.0*ds/6.0)*np.exp()
"""
