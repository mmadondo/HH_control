# evalOC.py
# run a trained model

import argparse
import torch
import os
import time
import datetime
import pandas as pd
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import tikzplotlib
import utils as utils
from Phi import *
from loss import control_obj, hjb_penalty, terminal_penalty

parser = argparse.ArgumentParser('Optimal Control')
parser.add_argument('--prob', choices=['Trajectory', 'DBS'],   type=str, default='DBS')

parser.add_argument("--nt_train"    , type=int, default=1500, help="number of time steps for training")
parser.add_argument("--nt_plot"    , type=int, default=1500, help="number of time steps for plotting")
parser.add_argument("--nt_val", type=int, default=1500, help="number of time steps for validation")
parser.add_argument("--n_train"    , type=int, default=64, help="number of training examples")
parser.add_argument("--n_val"    , type=int, default=128, help="number of validation examples")
parser.add_argument("--n_plot"    , type=int, default=1, help="number of plot examples")
parser.add_argument('--resume'  , type=str, default='./experiments/pretrained/DBSProblem_PhiNN_fwdEuler_betas_0-1_0-1_0-1_1-0_0-0_m64_nTh3_lr0-012023_05_25_09_57_30_checkpt.pth', help="for loading a pretrained model")

parser.add_argument('--lr'    , type=float, default=0.01)
parser.add_argument('--save'    , type=str, default='experiments/oc/eval', help="define the save directory")
parser.add_argument('--prec'    , type=str, default='double', choices=['single','double'], help="single or double precision")
parser.add_argument('--check_subopt', type=bool, default=True, help="check suboptimality")
parser.add_argument('--do_shock', type=bool, default=True, help="check robustness to shocks")
args = parser.parse_args()



# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)

def objF(prob, s, u, z, z_star):
    nex = u.shape[0]
    dt = 30.0/int(u.shape[1])
    l = torch.zeros(nex, 1, dtype=z.dtype, device=z.device)

    for i in range(s.shape[0]-1):
        # Forward Euler update 
        # z = z + dt * prob.f(i, z, u[i])
        # compute running cost
        l += dt * prob.L(s[i], z[:, i, :], u[:, i:i+1], z_star[i])
        
    g = prob.g(z[:, -1, :])[0]

    return l+g, l, g

if __name__ == '__main__':
    start_time = time.time()
    # load model
    logger.info(' ')
    logger.info("loading model: {:}".format(args.resume))
    logger.info(' ')

    # reload model
    checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
    args.m = checkpt['args'].m
    args.nt = checkpt['args'].nt_val
    args.nTh = checkpt['args'].nTh
    dbs_params = np.fromstring(checkpt['args'].dbs_params, sep=",")
    beta = [float(item) for item in checkpt['args'].beta.split(',')]

    cvt = lambda x: x.type(argPrec).to(device, non_blocking=True)

    # set precision and device
    if args.prec =='double':
        argPrec = torch.float64
    else:
        argPrec = torch.float32
    torch.set_default_dtype(argPrec)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.prob == 'DBS':
        from DBSProblem import DBSProblem
        # load target solution
        z_true = torch.from_numpy(np.load("./target_solution/target_sol.npz")["target_sol"])
        prob = DBSProblem(z_star=z_true, dbs_params=dbs_params)
        # define z_star for different samplers
        zstar = torch.cat((z_true[::int(z_true.shape[0] / args.nt)], z_true[-1:]), 0) # train

    if checkpt['args'].net == "ResNet_OTflow":
        from Phi_OTflow import Phi_OTflow
        Phi = Phi_OTflow(args.nTh, args.m, prob.d)
    elif checkpt['args'].net == "ResNN":
        from Phi import PhiNN
        from networks import ResNN
        net = nn.Sequential(
            ResNN(prob.d, args.m, args.nTh),
            nn.Linear(args.m, 1))
        Phi = PhiNN(net)

    Phi.net.load_state_dict(checkpt["state_dict"])
    Phi.net = Phi.net.to(argPrec).to(device)

    if checkpt['args'].sampler_train == "fwdEuler":
        from fsde import fwdEuler
        sampler = fwdEuler(Phi, prob, prob.t, prob.T, args.nt, zstar)
    elif checkpt['args'].sampler_train == "rk4":
        from fsde import rk4
        sampler = rk4(Phi, prob, prob.t, prob.T, args.nt, zstar)

    figPath = args.save + '/figures/evalDBS/'
    if not os.path.exists(os.path.dirname(figPath)):
        os.makedirs(os.path.dirname(figPath))

    strTitle = 'eval_' + args.resume[args.resume.index('fwd'):args.resume.index('check')]

    logger.info("---------------------- Network ----------------------------")
    logger.info(Phi.net)
    logger.info("----------------------- Problem ---------------------------")
    logger.info(prob)
    logger.info(strTitle)
    logger.info("--------------------------------------------------\n")

    bJustXinit = False   # eval model just on xInit

    net.eval()

    if bJustXinit:
        # just the xInit point, printed G includes the alpha_0
        if args.prob == 'DBS':
            xp = prob.xInit(args.nt)
            s, z, _, _, gradPhiz = sampler(xp)
            J, L, G = control_obj(Phi, prob, s, z, gradPhiz, sampler.z_star)
            prob.render(s,z,gradPhiz,J,figPath+args.resume[args.resume.index('_m'):args.resume.index('_ch')])
        else:
            logger.info("plotting not implemented for the provided data")
        logger.info('plot saved to ' + figPath)


    if args.check_subopt:
        logger.info('\n ===== Suboptimality Check =====\n')
        # compute J_star for fixed z = [0,0,0,0]
        z0 = torch.tensor([[0, 0, 0, 0]], dtype=torch.float64)
        s, z, _, _, gradPhiz = sampler(z0)
        J0, _, _ = control_obj(Phi, prob, s, z, gradPhiz, sampler.z_star)

        # load linspace pathological z0s from matlab
        patho_sols = sio.loadmat('./experiments/local_solution/sub_opt_exp/local_solution.mat')
        z0_matlab = np.zeros((100,4))
        z0_matlab[:,0] = patho_sols['Z0'].flatten() # -40:100:40
        xp = torch.from_numpy(z0_matlab)

        # load 'optimal' local solution
        matlab_u = torch.from_numpy(patho_sols['X'][:, :, 4])
        matlab_z = torch.from_numpy(patho_sols['X'][:, :, :-1])
        logger.info(f"nt: {args.nt}")
        columns = ["No.", "z0", "J_HJB", "L_HJB", "G_HJB", "J_matlab", "L_matlab", "G_matlab", "subopt"]
        hist_subopt = pd.DataFrame(columns=columns)
        # compute J for each z0
        s, z, _, _, gradPhiz = sampler(xp)
        J, L, G = control_obj(Phi, prob, s, z, gradPhiz, sampler.z_star, nomean=True)
        # get controls/ion channels
        z = torch.stack(z)
        u_all, iNa_all, iK_all, iL_all = 4*[torch.empty(xp.shape[0],0)]
        for i in range(len(z)-1):
            u_all = torch.cat([u_all, prob.u_star(s[i], z[i], gradPhiz[i])],1)
            iNa_all = torch.cat([iNa_all, (prob.gNa * z[i,:,1]**3 * z[i,:,3] * (z[i,:,0] - prob.ENa)).view(-1,1)], 1)
            iK_all = torch.cat([iK_all, (prob.gK * z[i,:,2]**4 * (z[i,:,0] - prob.EK)).view(-1,1)], 1)
            iL_all = torch.cat([iL_all, (prob.gL * (z[i,:,0] - prob.EL)).view(-1,1)], 1)

        iNa_all = torch.cat([iNa_all, (prob.gNa * z[-1, :, 1] ** 3 * z[-1, :, 3] * (z[-1, :, 0] - prob.ENa)).view(-1, 1)],
                            1)
        iK_all = torch.cat([iK_all, (prob.gK * z[-1, :, 2] ** 4 * (z[-1, :, 0] - prob.EK)).view(-1, 1)], 1)
        iL_all = torch.cat([iL_all, (prob.gL * (z[-1, :, 0] - prob.EL)).view(-1, 1)], 1)

        # compute J from optimal local trajectory and control
        J_matlab, L_matlab, G_matlab = objF(prob, s, matlab_u, matlab_z, sampler.z_star)

        # suboptimality
        subopt = torch.abs(J-J_matlab)/J_matlab

        for i in range(xp.shape[0]):
            hist_subopt.loc[len(hist_subopt.index)] = [i, xp[i][0].item(), J[i].item(), L[i].item(), G[i].item(),
                                                       J_matlab[i].item(), L_matlab[i].item(), G_matlab[i].item(),
                                                       subopt[i].item()]

            if i==0:
                log_message = (hist_subopt.astype({'No.': 'int'})).to_string(index=False)
                logger.info(log_message)
            else:
                ch = hist_subopt.iloc[-1:]
                log_message = (ch.astype({'No.': 'int'})).to_string(index=False, header=False)
                logger.info(log_message)

        elapsed = time.time() - start_time
        logger.info('Evaluation time: %.2f secs' % (elapsed))
        hist_subopt = hist_subopt.astype({'No.': 'int'})
        hist_subopt.to_csv(f"./experiments/local_solution/sub_opt_exp/local_subopt_{strTitle[strTitle.index('fwdE'):]}.csv", index = False)
        # plot subopt
        fig = plt.figure()
        plt.plot(hist_subopt['z0'].values, hist_subopt['subopt'].values)
        plt.xlabel(r'$\xi$')
        plt.ylabel('suboptimality')
        plt.savefig(f"{figPath}{strTitle}_subopt.pdf", bbox_inches='tight')
        # plt.show()
        plt.close()
        logger.info(f"plots saved to {figPath}")

        # plot J's
        fig = plt.figure()
        plt.plot(xp[:,0].detach().numpy(), J_matlab.detach().numpy(), label="local")
        plt.plot(xp[:,0].detach().numpy(), J.detach().numpy(), label="HJB")
        plt.xlabel(r"$z_0$")
        plt.ylabel("Control Objective")
        plt.legend()
        plt.savefig(f"{figPath}{strTitle}_ctrlobj.pdf", bbox_inches='tight')
        # plt.show()
        plt.close()
        logger.info(f"plots saved to {figPath}")

        # plot trajectories/control/ions for a random point from xp
        # s, z, _, _, _ = sampler(xp[55:56])
        fig = plt.figure()
        plt.plot(s.detach().numpy(), z[:, 55:56, 0].detach().numpy(), c='#1f77b4')
        plt.legend()
        plt.xlabel(r'Time ($ms$)')
        plt.ylabel(r'Voltage ($mV$)')
        plt.savefig(f"{figPath}{strTitle}_voltage_rand.pdf", bbox_inches='tight')
        # plt.show()
        plt.close()

        fig, axes = plt.subplots()
        axes.plot(s.detach().numpy(), z[:, 55:56, 1].detach().numpy(), c='k', ls='-', label="m")
        axes.plot(s.detach().numpy(), z[:, 55:56, 2].detach().numpy(), c='r', ls='-', label="n")
        axes.plot(s.detach().numpy(), z[:, 55:56, 3].detach().numpy(), c='#1f77b4', ls='-', label="h")
        lines = axes.get_lines()
        plt.legend()
        plt.ylabel("gating variables")
        plt.xlabel(r'Time ($ms$)')
        plt.savefig(f"{figPath}{strTitle}_gatingvars_rand.pdf", bbox_inches='tight')
        # plt.show()
        plt.close()

        # plot control
        fig = plt.figure()
        plt.plot(s[1:].detach().numpy(), u_all[55, :].detach().numpy(), c='#1f77b4')
        plt.legend()
        plt.xlabel(r'Time ($ms$)')
        plt.ylabel(r'Control ($\mu A/cm^2$)')
        plt.savefig(f"{figPath}{strTitle}_ctrl_rand.pdf", bbox_inches='tight')
        # plt.show()
        plt.close()

        # plot currents
        fig = plt.figure()
        plt.plot(s.detach().numpy(), iNa_all[55, :].detach().numpy(), c='k', ls='-', label="Na")
        plt.plot(s.detach().numpy(), iK_all[55, :].detach().numpy(), c='r', ls='-', label="K")
        plt.plot(s.detach().numpy(), iL_all[55, :].detach().numpy(), c='#1f77b4', ls='-', label="Leak")
        plt.legend()
        plt.ylabel("Ion Channel Current")
        plt.xlabel(r'Time ($ms$)')
        plt.savefig(f"{figPath}{strTitle}_curr_rand.pdf", bbox_inches='tight')
        # plt.show()
        plt.close()

        logger.info(f"plots saved to {figPath}")

        if args.do_shock:
            shockspec = [2.0, torch.Tensor([20.0,-0.1,-0.1,0.1]).view(1,-1)]
            s, z, _, Phiz, gradPhiz = sampler(prob.xInit)
            z = torch.stack(z)
            dt = (prob.T-prob.t)/args.nt
            nShock = int(shockspec[0]/dt)
            z1 = z.clone()
            z1[nShock] = z1[nShock] + shockspec[1]
            gradpH = sampler.prob.Hamiltonian(s[nShock], z1[nShock], gradPhiz[nShock], sampler.z_star[nShock])[1]
            for i in range(nShock, args.nt):
                ds = s[i+1]-s[i]
                z1[i+1] = z1[i] - gradpH * ds
                Phiz[i+1], gradPhiz[i+1], _ = sampler.Phi(s[i + 1], z1[i + 1], do_gradient=True)
                gradpH = sampler.prob.Hamiltonian(s[i + 1], z1[i + 1], gradPhiz[i + 1], sampler.z_star[i + 1])[1]

            # plot membrane potential and gating variables after shock
            for case in ["shock", "both"]:
                fig = plt.figure()
                plt.plot(s.detach().numpy(), z1[:, 0, 0].detach().numpy(), c='#1f77b4', ls='dashdot')
                if case == "both":
                    plt.plot(s.detach().numpy(), z[:, 0, 0].detach().numpy(), linestyle='-', c='#1f77b4')
                    plt.legend(['shock', 'no-shock'], loc="best")
                plt.xlabel(r'Time ($ms$)')
                plt.ylabel(r'Voltage ($mV$)')
                plt.savefig(f"{figPath}{strTitle}_voltage_{case}.pdf", bbox_inches='tight')
                # plt.show()
                plt.close()

                fig, axes = plt.subplots()
                axes.plot(s.detach().numpy(), z1[:, 0, 1].detach().numpy(), c='k', ls='dashdot')
                # plt.plot(s.detach().numpy(), sampler.z_star[:, 1].detach().numpy(), label='z_star', c='k')
                axes.plot(s.detach().numpy(), z1[:, 0, 2].detach().numpy(), c='r', ls='dashdot')
                # plt.plot(s.detach().numpy(), sampler.z_star[:, 2].detach().numpy(), c='green')
                axes.plot(s.detach().numpy(), z1[:, 0, 3].detach().numpy(), c='#1f77b4', ls='dashdot')
                # plt.plot(s.detach().numpy(), sampler.z_star[:, 3].detach().numpy(), c='green')
                lines = axes.get_lines()
                legend1 = plt.legend([lines[i] for i in [0, 1, 2]], ["m", "n", "h"], loc=1)

                if case == "both":
                    axes.plot(s.detach().numpy(), z[:, 0, 1].detach().numpy(), linestyle='-', c='k')
                    axes.plot(s.detach().numpy(), z[:, 0, 2].detach().numpy(), linestyle='-', c='r')
                    axes.plot(s.detach().numpy(), z[:, 0, 3].detach().numpy(), linestyle='-', c='#1f77b4')
                    # dummy lines with NO entries, just to create the black style legend
                    dummy_lines = []
                    dummy_lines.append(axes.plot([], [], c="b", ls="dashdot")[0])
                    dummy_lines.append(axes.plot([], [], c="b", ls="-")[0])
                    legend2 = plt.legend([dummy_lines[i] for i in [0, 1]], ["shock", "no-shock"], loc=4)
                    axes.add_artist(legend2)
                    lines = axes.get_lines()
                    legend1 = plt.legend([lines[i] for i in [3, 4, 5]], ["m", "n", "h"], loc="best")
                axes.add_artist(legend1)
                plt.ylabel("gating variables")
                plt.xlabel(r'Time ($ms$)')
                plt.savefig(f"{figPath}{strTitle}_gatingvars_{case}.pdf", bbox_inches='tight')
                # plt.show()
                plt.close()
                logger.info(f"plots saved to {figPath}")
