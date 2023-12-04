import torch
import torch.nn as nn
import time
from loss import control_obj, hjb_penalty, terminal_penalty
import os
from utils import count_parameters, makedirs, get_logger, floatFormat
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# try:
#     matplotlib.use('TkAgg')
#     import matplotlib.pyplot as plt
# except:
#     matplotlib.use('Agg')  # for linux server with no tkinter
#     import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser('Optimal Control')
parser.add_argument('--prob', choices=['Trajectory', 'DBS'],   type=str, default='DBS')
parser.add_argument('--net', choices=['ResNN', 'ResNet_OTflow'],   type=str, default='ResNN')
parser.add_argument('--sampler_train', choices=['fwdEuler','rk4'],   type=str, default='fwdEuler')
parser.add_argument('--sampler_plot', choices=['fwdEuler','rk4'], type=str, default='fwdEuler')
parser.add_argument('--sampler_val', choices=['fwdEuler','rk4'], type=str, default='fwdEuler')
parser.add_argument("--nt_train"    , type=int, default=1500, help="number of time steps for training")
parser.add_argument("--nt_plot"    , type=int, default=1500, help="number of time steps for plotting")
parser.add_argument("--nt_val", type=int, default=1500, help="number of time steps for validation")
parser.add_argument("--n_train"    , type=int, default=64, help="number of training examples")
parser.add_argument("--n_val"    , type=int, default=128, help="number of validation examples")
parser.add_argument("--n_plot"    , type=int, default=1, help="number of plot examples")
parser.add_argument('--trainsteps_freq' , type=int  , default=10000, help="how often to increase step size")
parser.add_argument('--trainsteps_decay', type=float, default=2, help="how much to increase step size")
parser.add_argument('--beta'  , type=str, default='0.1, 0.1, 0.1, 1.0, 0.0') # Terminal, grad terminal, HJB, J, Phi(0)-J; Note: Terminal already has a weight of 100
parser.add_argument('--m'     , type=int, default=32, help="NN width")
parser.add_argument('--nTh'     , type=int, default=2, help="NN depth")
parser.add_argument('--save'    , type=str, default='experiments/oc/run', help="define the save directory")
parser.add_argument('--prec'    , type=str, default='double', choices=['single','double'], help="single or double precision")
parser.add_argument('--dbs_params'    , type=str, default='1.0, 380.0, 36.0, 0.3, 115.0, -12.0, 10.613',
                    choices=['1.0, 380.0, 36.0, 0.3, 115.0, -12.0, 10.613','1.0, 120.0, 36.0, 0.3, 115.0, -12.0, 10.613'], help="select HH params")
parser.add_argument('--resume'  , type=str, default=None, help="for loading a pretrained model")
parser.add_argument('--n_iters', type=int, default=1000)
parser.add_argument('--lr'    , type=float, default=0.01)
parser.add_argument('--optim' , type=str, choices=['adam', 'lbfgs'], default='adam')
parser.add_argument('--optim_swap_freq' , type=int, default=1000, help="when to swap optim to lbfgs")
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--lr_freq' , type=int  , default=4000, help="how often to decrease lr")
parser.add_argument('--lr_decay', type=float, default=0.1, help="how much to decrease lr")
parser.add_argument('--val_freq', type=int, default=50, help="how often to run model on validation set")
parser.add_argument('--viz_freq', type=int, default=200, help="how often to plot visuals") # must be >= val_freq
parser.add_argument('--print_freq', type=int, default=25, help="how often to print results to log")
parser.add_argument('--sample_freq',type=int, default=10000, help="how often to resample training data")

args = parser.parse_args()

beta = [float(item) for item in args.beta.split(',')]
dbs_params = np.fromstring(args.dbs_params, sep=",")
sStartTime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# logger
logger = {}
makedirs(args.save)
logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__), saving = True)
logger.info("start time: " + sStartTime)
logger.info(args)

if __name__ == '__main__':

    if args.resume is not None:
        logger.info(' ')
        logger.info("loading model: {:}".format(args.resume))
        logger.info(' ')

        checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        args.m = checkpt['args'].m
        args.nTh = checkpt['args'].nTh
        dbs_params = np.fromstring(checkpt['args'].dbs_params, sep=",")
        beta = [float(item) for item in checkpt['args'].beta.split(',')]

    # set precision and device
    if args.prec == 'double':
        argPrec = torch.float64
    else:
        argPrec = torch.float32
    torch.set_default_dtype(argPrec)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.prob == 'DBS':
        from DBSProblem import DBSProblem
        # laod target solution
        z_true = torch.from_numpy(np.load("./target_solution/target_sol.npz")["target_sol"])
        prob = DBSProblem(z_star=z_true, dbs_params=dbs_params)
        # define z_star for different samplers
        zstar = torch.cat((z_true[::int(z_true.shape[0] / args.nt_train)], z_true[-1:]), 0) # train
        zstar_val = torch.cat((z_true[::int(z_true.shape[0] / args.nt_val)], z_true[-1:]), 0)  # validation
        zstar_plot = torch.cat((z_true[::int(z_true.shape[0] / args.nt_plot)], z_true[-1:]), 0)  # plot
    elif args.prob == 'Trajectory':
        from TrajectoryProblem import TrajectoryProblem
        prob = TrajectoryProblem()
    else:
        raise ValueError("Invalid combination of problem and network.")

    if args.net == "ResNet_OTflow":
        from Phi_OTflow import Phi_OTflow
        Phi = Phi_OTflow(args.nTh, args.m, prob.d)
    elif args.net == "ResNN":
        from Phi import PhiNN
        from networks import ResNN
        net = nn.Sequential(
            ResNN(prob.d, args.m, args.nTh),
            nn.Linear(args.m, 1))
        Phi = PhiNN(net)

    if args.resume is not None:
        Phi.net.load_state_dict(checkpt["state_dict"])
        Phi.net = Phi.net.to(argPrec).to(device)

    if args.sampler_train == "fwdEuler":
        from fsde import fwdEuler
        sampler = fwdEuler(Phi, prob, prob.t, prob.T, args.nt_train, zstar)
    elif args.sampler_train == "rk4":
        from fsde import rk4
        sampler = rk4(Phi, prob, prob.t, prob.T, args.nt_train, zstar)

    if args.sampler_val == "fwdEuler":
        from fsde import fwdEuler
        sampler_val = fwdEuler(Phi,prob,prob.t,prob.T,args.nt_val, zstar_val)
    elif args.sampler_val == "rk4":
        from fsde import rk4
        sampler_val = rk4(Phi,prob,prob.t,prob.T,args.nt_val, zstar_val)

    if args.sampler_plot == "fwdEuler":
        from fsde import fwdEuler
        sampler_plot = fwdEuler(Phi, prob, prob.t, prob.T, args.nt_plot, zstar_plot)
    elif args.sampler_plot == "rk4":
        from fsde import rk4
        sampler_plot = rk4(Phi, prob, prob.t, prob.T, args.nt_plot, zstar_plot)

    if args.optim == 'adam':
        lr = args.lr
        optim = torch.optim.Adam(Phi.parameters(), lr=lr, weight_decay=args.weight_decay)


    strTitle = prob.__class__.__name__ + '_' + Phi._get_name() + '_' + sampler._get_name() + '_betas_{:}_{:}_{:}_{:}_{:}_m{:}_nTh{:}_lr{:}'.format(
                     floatFormat(beta[0]), 
                     floatFormat(beta[1]),
                     floatFormat(beta[2]),
                     floatFormat(beta[3]),
                     floatFormat(beta[4]), args.m, args.nTh, floatFormat(args.lr)) + sStartTime  # add a flag before start time for tracking
    logger.info("---------------------- Network ----------------------------")
    logger.info(Phi.net)
    logger.info("----------------------- Problem ---------------------------")
    logger.info(prob)
    logger.info("------------------------ Sampler (train) --------------------------")
    logger.info(sampler)
    logger.info("------------------------ Sampler (validation) --------------------------")
    logger.info(sampler)
    logger.info("--------------------------------------------------")
    logger.info("beta={:}".format(args.beta))
    logger.info("Number of trainable parameters: {}".format(count_parameters(Phi.net)))
    logger.info("--------------------------------------------------")
    logger.info(str(optim)) # optimizer info
    logger.info("dtype={:} device={:}".format(argPrec, device))
    logger.info("n_train={:} n_val={:} n_plot={:}".format(args.n_train, args.n_val, args.n_plot))
    logger.info("maxIters={:} val_freq={:} viz_freq={:}".format(args.n_iters, args.val_freq, args.viz_freq))
    logger.info("saveLocation = {:}".format(args.save))
    logger.info(strTitle)
    logger.info("--------------------------------------------------\n")

    columns = ["step","loss","Phi0", "J", "L","G","cHJB","cPhi","cBSDEfin","cBSDEgrad","lr"]
    train_hist = pd.DataFrame(columns=columns)
    val_hist = pd.DataFrame(columns=columns)

    xp = prob.x_init(args.n_plot)
    # print(f"inital positions: \n {xp} \n")
    s,z,dw,Phiz,gradPhiz = sampler_plot(xp)
    J, L, G = control_obj(Phi, prob, s, z, gradPhiz, sampler_plot.z_star)

    fig = plt.figure()
    ax = plt.gca()
    figPath = args.save + '/figures/DBS/'
    if not os.path.exists(os.path.dirname(figPath)):
        os.makedirs(os.path.dirname(figPath))
    # prob.render(s,z,Phi,Phi(s[0],xp),gradPhiz,J,clamp,os.path.join(figPath, '%s_iter_%s.pdf' % (strTitle, 'pre-training')))

    best_loss = float('inf')
    bestParams = None
    Phi.net.train()

    x = prob.x_init(args.n_train)
    xv = prob.x_init(args.n_val)
    xp = prob.x_init(args.n_plot)

    makedirs(args.save)
    start_time = time.time()

    # for keeping track of validation cost change
    oldJv = torch.zeros(1)
    for itr in range(args.n_iters - 1):
        if itr > 0 and itr % args.sample_freq == 0:
            x = prob.x_init(args.n_train)

        if args.optim == 'adam':
            optim.zero_grad()
            s, z, dw, Phiz, gradPhiz = sampler(x)
            J, L, G = control_obj(Phi, prob, s, z, gradPhiz, sampler.z_star)
            term_Phi, term_gradPhi = terminal_penalty(Phi, prob, s, z, dw, Phiz, gradPhiz, sampler.z_star)
            cHJB = hjb_penalty(Phi, prob, s, z, dw, Phiz, gradPhiz, sampler.z_star)
            Phi0 = Phi(s[0],x)
            cPhi = torch.mean(torch.abs(Phi0-J))
            loss = beta[0]*term_Phi + beta[1]*term_gradPhi + beta[2] * cHJB + beta[3]*J + beta[4] * cPhi
            loss.backward()
            optim.step()

        if args.optim == 'lbfgs':
            def closure():
                optim.zero_grad()
                s, z, dw, Phiz, gradPhiz = sampler(x)
                J, L, G = control_obj(Phi, prob, s, z, gradPhiz, sampler.z_star)
                term_Phi, term_gradPhi = terminal_penalty(Phi, prob, s, z, dw, Phiz, gradPhiz, sampler.z_star)
                cHJB = hjb_penalty(Phi, prob, s, z, dw, Phiz, gradPhiz, sampler.z_star)
                Phi0 = Phi(s[0],x)
                cPhi = torch.mean(torch.abs(Phi0-J))
                loss = beta[0]*term_Phi + beta[1]*term_gradPhi + beta[2] * cHJB + beta[3]*J + beta[4] * cPhi
                loss.backward()
                return loss
            optim.step(closure)

            # Todo: talk to Malvern about this
            # s, z, dw, Phiz, gradPhiz = sampler(x)
            # if args.track_z == 'False':
            #     optim.zero_grad()
            #     (Phiz, gradPhiz) = (None,None)
            """
            Repetion?  
            J, L, G = control_obj(Phi, prob, s, z, gradPhiz, clamp)
            term_Phi, term_gradPhi = terminal_penalty(Phi, prob, s, z, dw, Phiz, gradPhiz)
            cBSDE = bsde_penalty(Phi, prob, s, z, dw, Phiz, gradPhiz, clamp)
            cHJB = hjb_penalty(Phi, prob, s, z, dw, Phiz, gradPhiz, clamp)
            Phi0 = Phi(s[0],x)
            cPhi = torch.mean(torch.abs(Phi0-J))
            loss = beta[0]*cBSDE + beta[0]*term_Phi + beta[1]*term_gradPhi + beta[2] * cHJB + beta[3]*J + beta[4] * cPhi
            """

        train_hist.loc[len(train_hist.index)] = [itr,loss.item(),torch.mean(Phi0).item(), J.item(), L.item(),G.item(),cHJB.item(),cPhi.item(),term_Phi.item(),term_gradPhi.item(),lr]

        # printing
        if itr % args.print_freq == 0:
            ch = train_hist.iloc[-1:]
            if itr >0:
                ch.columns=11*['']
                ch.index.name=None
                log_message = (ch.to_string().split("\n"))[1]
            else:
                log_message = ch
            logger.info(log_message)

        if itr % args.val_freq == 0 or itr == args.n_iters:
            sv, zv, dwv, Phizv, gradPhizv = sampler_val(xv)
            Jv, Lv, Gv = control_obj(Phi, prob, sv, zv, gradPhizv, sampler_val.z_star)
            term_Phiv, term_gradPhiv = terminal_penalty(Phi, prob, sv, zv, dwv, Phizv, gradPhizv, sampler_val.z_star)
            cHJBv = hjb_penalty(Phi, prob, sv, zv, dwv, Phizv, gradPhizv, sampler_val.z_star)
            Phi0v = Phi(sv[0], xv)
            cPhiv = torch.mean(torch.abs(Phi0v - Jv))
            val_loss = beta[0] * term_Phiv + beta[1] * term_gradPhiv + beta[2] * cHJBv + beta[3] * Jv + beta[4] * cPhiv
            val_hist.loc[len(val_hist.index)] = [itr,val_loss.item(), torch.mean(Phi0v).item(), Jv.item(), Lv.item(),
                                                                 Gv.item(), cHJBv.item(), cPhiv.item(),
                                                                 term_Phiv.item(), term_gradPhiv.item(),lr]
            ch = val_hist.iloc[-1:]
            if itr >0:
                ch.columns=11*['']
                ch.index.name=None
                log_message = (ch.to_string().split("\n"))[1]
            else:
                log_message = ch
            logger.info(log_message)

            if val_loss.item() < best_loss:
                best_loss = val_loss.item()
                makedirs(args.save)
                bestParams = Phi.net.state_dict()
                torch.save({
                    'args': args,
                    'state_dict': bestParams,
                    }, os.path.join(args.save, strTitle + '_checkpt.pth'))
        # check if validation J went down after val_freq iterations
        if torch.norm(Jv - oldJv)<1e-16:
            print(f"No change in val control obj. Terminating at {itr} iterations!")
            sp, zp, dwp, Phizp, gradPhizp = sampler_plot(xp)
            Jp, Lp, Gp = control_obj(Phi, prob, sp, zp, gradPhizp, sampler_plot.z_star)
            prob.render(sp, zp, gradPhizp, Jp, os.path.join(figPath, '%s_iter_%d_' % (strTitle, itr)))
            break
        else:
            oldJv = Jv
        # shrink step size
        if (itr + 1) % args.lr_freq == 0:
            lr *= args.lr_decay
            for p in optim.param_groups:
                p['lr'] *= lr
        
        if (itr + 1) % args.trainsteps_freq == 0:
            args.nt_train = args.nt_train * args.trainsteps_decay
            sampler.nt = args.nt_train

        if itr % args.viz_freq == 0:
            sp, zp, dwp, Phizp, gradPhizp = sampler_plot(xp)
            Jp, Lp, Gp = control_obj(Phi, prob, sp, zp, gradPhizp, sampler_plot.z_star)
            prob.render(sp, zp, gradPhizp, Jp, os.path.join(figPath, '%s_iter_%d_' % (strTitle, itr)))

    elapsed = time.time() - start_time

    print('Training time: %.2f secs' % (elapsed))
    s, z, dw, Phiz, gradPhiz = sampler_plot(xp)
    J, L, G = control_obj(Phi, prob, s, z, gradPhiz, sampler_plot.z_star)
    prob.render(s, z, gradPhiz, J, os.path.join(figPath, '%s_iter_%d_' % (strTitle, itr)))
    train_hist.to_csv(os.path.join('./experiments/data/', '%s_train_hist.csv' % (strTitle )))
    val_hist.to_csv(os.path.join('./experiments/data/', '%s_val_hist.csv' % (strTitle )))
