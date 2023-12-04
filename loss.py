import torch
import torch.nn as nn
from Phi import PhiNN


def control_obj(Phi, prob, s, z, gradPhiz, z_star=None, nomean=False):
    nex = z[0].shape[0]
    nt = s.shape[0]
    L = torch.zeros(nex, 1, dtype=z[0].dtype, device=z[0].device)
    for i in range(nt - 1):
        ds = s[i + 1] - s[i]
        if gradPhiz is None:
            Phizi, gradPhizi, _ = Phi(s[i], z[i], do_gradient=True)
        else:
            gradPhizi = gradPhiz[i]

        ui = prob.u_star(s[i], z[i], gradPhizi)
        # print(f"\nui = {ui}\n")
        L = L + ds * prob.L(s[i], z[i], ui, z_star[i])

    G = prob.g(z[-1])[0]

    if nomean:
        return L + G, L, G
    else:
        return torch.mean(L) + torch.mean(G), torch.mean(L), torch.mean(G)


def terminal_penalty(Phi, prob, s, z, dw, Phiz, gradPhiz, z_star=None):
    if Phiz is None or gradPhiz is None:
        Phizi, gradPhizi, _ = Phi(s[-1], z[-1], do_gradient=True)
    else:
        (Phizi, gradPhizi) = (Phiz[-1], gradPhiz[-1])

    G, gradG = prob.g(z[-1])
    cHJBfin = torch.mean(torch.abs(Phizi - G.view(-1, 1)))
    cHJBgrad = torch.mean(torch.sum(torch.abs(gradPhizi - gradG), dim=1))
    return cHJBfin, cHJBgrad


def hjb_penalty(Phi, prob, s, z, dw, Phiz, gradPhiz, z_star=None):
    cHJB = 0.0
    for i in range(s.shape[0] - 1):
        ds = s[i + 1] - s[i]
        if dw is None:
            Phizi, gradPhizi, dtPhizi = Phi(s[i + 1], z[i + 1], do_gradient=True)
            res = - dtPhizi + prob.Hamiltonian(s[i], z[i], gradPhizi, z_star[i])[0]

        elif prob.__class__.__name__ == 'TrajectoryProblem2':
            Phizi, gradPhizi, dtPhizi, hessPhizi = Phi(s[i + 1], z[i + 1], do_gradient=True)
            res = - dtPhizi - 0.5 * prob.tr_sigma2_M(s[i + 1], z[i + 1], hessPhizi) + \
                  prob.Hamiltonian(s[i], z[i], -gradPhizi)[0]
        else:
            Phizi, gradPhizi, dtPhizi, LapPhi = Phi(s[i + 1], z[i + 1], do_gradient=True, do_Laplacian=True)
            res = - dtPhizi - 0.5 * prob.sigma(s[i + 1], z[i + 1]) ** 2 * LapPhi + \
                  prob.Hamiltonian(s[i], z[i], gradPhizi)[0]
        # Phizi, gradPhizi, dtPhizi, hessPhizi = Phi(s[i + 1], z[i + 1], do_gradient=True, do_Hessian=True)
        # if prob.__class__.__name__ == 'TrajectoryProblem':
        #     LapPhi = 0.5* prob.sigma(s[i + 1], z[i + 1])**2*torch.sum(
        #         hessPhizi* torch.eye(z[i + 1].shape[1]).view(1, z[i + 1].shape[1], z[i + 1].shape[1]),
        #         dim=(1, 2)).unsqueeze(1)
        # else:
        #     sigsigthess = prob.sigma(s[i + 1], z[i + 1])*torch.transpose(prob.sigma(s[i + 1], z[i + 1]),1,2)*hessPhizi
        #     LapPhi = 0.5 * torch.sum(
        #         sigsigthess * torch.eye(z[i + 1].shape[1]).view(1, z[i + 1].shape[1], z[i + 1].shape[1]),
        #         dim=(1, 2)).unsqueeze(1)
        # res = - dtPhizi - LapPhi + prob.Hamiltonian(s[i],z[i],-gradPhizi,clamp,None)[0]
        cHJB = cHJB + torch.mean(torch.abs(res), dim=0) * ds

    return cHJB



if __name__ == '__main__':
    from TrajectoryProblem import TrajectoryProblem
    from fsde import EulerMaryama

    d = 2
    nex = 2
    width = [8]

    net = nn.Sequential(
        nn.Linear(d + 1, width[0]),
        nn.Tanh(),
        nn.Linear(width[-1], 1))

    Phi = PhiNN(net)

    prob = TrajectoryProblem()
    nex = 300
    nt = 100
    beta = (1.0, 1.0, 1.0)
    prob.sigma = 0.0  # deterministic
    x = prob.x_init(nex)

    integrator = EulerMaryama(0.0, 1.0, nt)

    loss = FBSDEloss(integrator, beta)

    Jc, L, G, cBSDE, cBSDEfin, cBSDEgrad = loss(Phi, prob, x)
    G.backward()

    W0 = net[0].weight.data.clone()
    W = net[0].weight
    dW = torch.randn_like(W0)
    dFdW = torch.sum(dW * W.grad)
    for k in range(20):
        h = 0.5 ** k
        Phi.net[0].weight.data = W0 + h * dW
        Jt, Lt, Gt, cBSDEt, cBSDEfint, cBSDEgradt = loss(Phi, prob, x)
        # Ft = torch.sum(dPhidyt)
        E0 = torch.norm(G - Gt)
        E1 = torch.norm(G + h * dFdW - Gt)

        print("h=%1.2e\tE0=%1.2e\tE1=%1.2e" % (h, E0, E1))
