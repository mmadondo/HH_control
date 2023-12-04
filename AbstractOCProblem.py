import torch
import torch.autograd as autograd
import matplotlib.pyplot as plt

class AbstractOCProblem:
    def __init__(self):
        return self

    def calH(self,s,z,p,u,M=None):
        """

        calH = f(s,z,u)'*p - L(s,z,u) + trace(sigma^2 M)

        :param s: time
        :param z: state
        :param p: adjoint state
        :param u: control
        :param M: adjoint variable in stochastic OC problems
        :return: calH
        """
        H = - torch.sum(p*self.f(s,z,u),dim=1,keepdim=True) - self.L(s,z,u)
        if M is not None:
            H = H + self.sigma2_Lap(s,z,M)
        return H

    def test_Hamiltonian(self,s,z,p,M=None):
        """
        Test 1: | calH(s,z,p,u_star) - Hamiltonian(s,z,p) |

        Test 2: derivative check for gradpH computed in Hamiltonian

        :param s:
        :param z:
        :param p:
        :param M:
        :return:
        """
        print("\n ===============  test_Hamiltonian =============== ")
        H_true = self.calH(s, z, p, self.u_star(s, z, p))
        assert H_true.dim() == 2
        assert H_true.shape[0] == z.shape[0]
        assert H_true.shape[1] == 1
        H,gradpH = self.Hamiltonian(s,z,p)
        assert H.dim() == 2
        assert H.shape[0] == z.shape[0]
        assert H.shape[1] == 1
        assert gradpH.shape[0] == z.shape[0]
        assert gradpH.shape[1] == p.shape[1]

        err_H = torch.norm(H_true - H)
        print("err_H = %1.2e" % (err_H))

        dp = torch.randn_like(p)
        dd = torch.sum(dp*gradpH,dim=1,keepdim=True)
        h_arr, E0_arr, E1_arr = [], [], []

        for k in range(20):
            h = (0.5)**(k)
            Ht = self.Hamiltonian(s,z,p+h*dp)[0]

            E0 = torch.norm(H-Ht)
            E1 = torch.norm(H + h*dd -Ht)
            print("h=%1.2e\tE0=%1.2e\tE1=%1.2e" %(h,E0,E1))
            h_arr.append(h)
            E0_arr.append(E0)
            E1_arr.append(E1)
        plt.loglog(h_arr,E0_arr, label='f')
        plt.loglog(h_arr,E1_arr, label='gradf')
        plt.legend()
        plt.title("test_Hamiltonian")
        plt.show()


    def test_g(self,z):
        g,gradg = self.g(z)
        # verify sizes
        assert g.dim()==2
        assert g.shape[0]==z.shape[0]
        assert g.shape[1]==1
        assert gradg.shape[0]==z.shape[0]
        assert gradg.shape[1]==z.shape[1]

        dz = torch.randn_like(z)
        dgdz = torch.sum(gradg*dz,dim=1,keepdim=True)
        h_arr, E0_arr, E1_arr = [], [], []
        print("\n ===============  test_g =============== ")

        for k in range(20):
            h = (0.5)**(k)
            gt = self.g(z+h*dz)[0]

            E0 = torch.norm(g-gt)
            E1 = torch.norm(g + h*dgdz -gt)
            print("h=%1.2e\tE0=%1.2e\tE1=%1.2e" %(h,E0,E1))
            h_arr.append(h)
            E0_arr.append(E0)
            E1_arr.append(E1)
        plt.loglog(h_arr,E0_arr, label='f')
        plt.loglog(h_arr,E1_arr, label='gradf')
        plt.legend()
        plt.title("test_g")
        plt.show()

    def test_u_star(self,s,z,p):
        """
        test the feedback form. Note that

        u_star = argmax_u calH(s,z,p,u)

        and therefore |\nabla_u calH(s,z,p,u_star)| \approx 0

        :param s:
        :param z:
        :param p:
        :return:
        """
        # ut = torch.tensor(self.u_star(s,z,p),requires_grad=True)
        ut = self.u_star(s,z,p).clone().detach().requires_grad_(True)
        H = self.calH(s,z,p,ut)
        dH = autograd.grad(H, ut, torch.ones([ut.shape[0], 1], device=ut.device, dtype=ut.dtype), retain_graph=True,
                      create_graph=True)[0]
        err = torch.norm(dH)
        # du = torch.randn_like(ut)
        # dd = torch.sum(du*dH,dim=1,keepdim=True)
        # for k in range(20):
        #     h = (0.5)**(k)
        #     Ht = self.calH(s, z, p, ut + h*du)
        #
        #     E0 = torch.norm(H-Ht)
        #     E1 = torch.norm(H + h*dd -Ht)
        #     print("h=%1.2e\tE0=%1.2e\tE1=%1.2e" %(h,E0,E1))
        return err
