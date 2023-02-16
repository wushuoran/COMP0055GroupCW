from __future__ import division, print_function

import argparse
from ast import main
from bisect import bisect
from datetime import datetime
import numpy as np
import numpy.linalg as la
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model


def setup_argparse():
    parser = argparse.ArgumentParser(description='handle poisoning inputs')

    # dataset file
    #parser.add_argument("-d", "--dataset",default='../datasets/house-processed.csv',\
    #                        help='dataset filename (includes path)')

    # counts
    parser.add_argument("-r", "--trainct",default=300, type=int,\
                            help='number of points to train models with')
    parser.add_argument("-t", "--testct",default=500, type=int,\
                            help = 'number of points to test models on')
    parser.add_argument("-v","--validct",default=250, type=int,\
                            help='size of validation set')
    parser.add_argument("-p", "--poisct",default=75, type=int,\
                            help='number of poisoning points')
    parser.add_argument("-s", "--partct",default=4, type=int,\
                            help='number of increments to poison with')

    # logging results
    parser.add_argument("-ld", "--logdir",default="../results",\
                            help='directory to store output')
    parser.add_argument("-li","--logind",default=0,\
                            help='output files will be err{model}{outputind}.txt, train{model}{outputind}.txt, test{model}{outputind}.txt')

    # model
    parser.add_argument("-m", "--model",default='linreg',\
                            choices=['linreg','lasso','enet','ridge'],\
                            help="choose linreg for linear regression, lasso for lasso, enet for elastic net, or ridge for ridge")

    # params for gd poisoning
    parser.add_argument("-l", "--lambd",default=1, type=float,help='lambda value to use in poisoning;icml 2015')
    parser.add_argument("-n", "--epsilon",default=1e-3, type=float,help='termination condition epsilon;icml 2015')
    parser.add_argument("-a", "--eta",default=0.01, type=float,help='line search multiplier eta; icml 2015')
    parser.add_argument("-b", "--beta",default=0.05, type=float,help='line search decay beta; icml 2015')
    parser.add_argument("-i", "--sigma",default=0.9, type=float,help='line search termination lowercase sigma;icml 2015')

    # objective
    parser.add_argument("-obj","--objective",default=0,type=int,
                            help="objective to use (0 for train, 1 for validation, 2 for norm difference)")
    parser.add_argument('-opty','--optimizey',action='store_true',
                            help='optimize the y values of poisoning as well')

    #init strategy
    parser.add_argument('-init','--initialization',default='inflip',
                            choices=['levflip', 'cookflip', 'alfatilt', 'randflip',\
                                     'randflipnobd', 'inflip', 'ffirst', 'adaptive',\
                                     'adaptilt','rmml'],\
                            help="init strategy")
    parser.add_argument('-numinit', type=int, default=1, help='number of times to attempt initialization')

    # round
    parser.add_argument("-rnd",'--rounding',action='store_true',help='to round or not to round')

    # seed for randmization
    parser.add_argument('-seed',type=int,help='random seed')

    # enable multi processing
    parser.add_argument("-mp","--multiproc",action='store_true',\
                            help='enable to allow for multiprocessing support')

    # visualize simple case
    parser.add_argument("-vis",'--visualize',action='store_true',help="visualize dataset")

    return parser

class GDPoisoner(object):
    def __init__(self, x, y, testx, testy, validx, validy, \
                 eta, beta, sigma, eps, \
                 mproc, \
                 trainfile, resfile, \
                 objective, opty, colmap):
        """
        GDPoisoner handles gradient descent and poisoning routines
        Computations for specific models found in respective classes

        x, y: training set
        testx, testy: test set
        validx, validy: validation set used for evaluation and stopping

        eta: gradient descent step size (note gradients are normalized)
        beta: decay rate for line search
        sigma: line search stop condition
        eps: poisoning stop condition

        mproc: whether to use multiprocessing to parallelize poisoning updates

        trainfile: file storing poisoning points in each iteration
        resfile: file storing each iteration's results in csv format

        objective: which objective to use
        opty: whether to optimize y
        colmap: map of columns for onehot encoded features
        """

        self.trnx = x
        self.trny = y
        self.tstx = testx
        self.tsty = testy
        self.vldx = validx
        self.vldy = validy

        self.samplenum = x.shape[0]
        self.feanum = x.shape[1]

        self.objective = objective
        self.opty = opty

        if (objective == 0):  # training MSE + regularization
            self.attack_comp = self.comp_attack_trn
            self.obj_comp = self.comp_obj_trn

        elif (objective == 1):  # validation MSE
            self.attack_comp = self.comp_attack_vld
            self.obj_comp = self.comp_obj_vld

        elif (objective == 2):  # l2 distance between clean and poisoned
            self.attack_comp = self.comp_attack_l2
            self.obj_comp = self.comp_obj_new

        else:
            raise NotImplementedError

        self.mp = mproc  # use multiprocessing?

        self.eta = eta
        self.beta = beta
        self.sigma = sigma
        self.eps = eps

        self.trainfile = trainfile
        self.resfile = resfile
        self.initclf, self.initlam = None, None

        self.colmap = colmap

    def poison_data(self, poisx, poisy, tstart, visualize, newlogdir):
        """
        poison_data takes an initial set of poisoning points and optimizes it
        using gradient descent with parameters set in __init__

        poisxinit, poisyinit: initial poisoning points
        tstart: start time - used for writing out performance
        visualize: whether we want to visualize the gradient descent steps
        newlogdir: directory to log into, to save the visualization
        """

        poisct = poisx.shape[0]
        print("Poison Count: {}".format(poisct))

        new_poisx = np.zeros(poisx.shape)
        new_poisy = [None for a in poisy]

        if visualize:
            # initialize poisoning histories
            poisx_hist = np.zeros(
                (10, poisx.shape[0], poisx.shape[1]))
            poisy_hist = np.zeros((10, poisx.shape[0]))

            # store first round
            poisx_hist[0] = poisxinit[:]
            poisy_hist[0] = np.array(poisy)

        best_poisx = np.zeros(poisx.shape)
        best_poisy = [None for a in poisy]

        best_obj = 0
        last_obj = 0
        count = 0

        if self.mp:
            import multiprocessing as mp
            workerpool = mp.Pool(max(1, mp.cpu_count() // 2 - 1))
        else:
            workerpool = None

        sig = self.compute_sigma()  # can already compute sigma and mu
        mu = self.compute_mu()  # as x_c does not change them
        eq7lhs = np.bmat([[sig, np.transpose(mu)],
                          [mu, np.matrix([1])]])

        # initial model - used in visualization
        clf_init, lam_init = self.learn_model(self.trnx, self.trny, None)
        clf, lam = clf_init, lam_init

        # figure out starting error
        it_res = self.iter_progress(poisx, poisy, poisx, poisy)

        print("Iteration {}:".format(count))
        print("Objective Value: {} Change: {}".format(it_res[0], it_res[0]))
        print("Validation MSE: {}".format(it_res[2][0]))
        print("Test MSE: {}".format(it_res[2][1]))

        last_obj = it_res[0]
        if it_res[0] > best_obj:
            best_poisx, best_poisy, best_obj = poisx, poisy, it_res[0]

        # stuff to put into self.resfile
        towrite = [poisct, count, it_res[0], it_res[1], \
                   it_res[2][0], it_res[2][1], \
                   (datetime.datetime.now() - tstart).total_seconds()]

        self.resfile.write(','.join([str(val) for val in towrite]) + "\n")
        self.trainfile.write("\n")
        self.trainfile.write(str(poisct) + "," + str(count) + '\n')

        if visualize:
            self.trainfile.write('{},{}\n'.format(poisy[0], new_poisx[0]))
        else:
            for j in range(poisct):
                self.trainfile.write(','.join(
                    [str(val) for val
                     in [poisy[j]] + poisx[j].tolist()[0] \
                     ]) + '\n')

        # main work loop
        while True:
            count += 1
            new_poisx = np.matrix(np.zeros(poisx.shape))
            new_poisy = [None for a in poisy]
            x_cur = np.concatenate((self.trnx, poisx), axis=0)
            y_cur = self.trny + poisy

            clf, lam = self.learn_model(x_cur, y_cur, None)
            pois_params = [(poisx[i], poisy[i], eq7lhs, mu, clf, lam) \
                           for i in range(poisct)]
            outofboundsct = 0

            if workerpool:  # multiprocessing
                for i, cur_pois_res in enumerate(
                        workerpool.map(self.poison_data_subroutine,
                                       pois_params)):
                    new_poisx[i] = cur_pois_res[0]
                    new_poisy[i] = cur_pois_res[1]
                    outofboundsct += cur_pois_res[2]

            else:
                for i in range(poisct):
                    cur_pois_res = self.poison_data_subroutine(pois_params[i])

                    new_poisx[i] = cur_pois_res[0]
                    new_poisy[i] = cur_pois_res[1]
                    outofboundsct += cur_pois_res[2]

            if visualize:
                poisx_hist[count] = new_poisx[:]
                poisy_hist[count] = np.array(new_poisy).ravel()

            it_res = self.iter_progress(poisx, poisy, new_poisx, new_poisy)

            print("Iteration {}:".format(count))
            print("Objective Value: {} Change: {}".format(
                it_res[0], it_res[0] - it_res[1]))

            print("Validation MSE: {}".format(it_res[2][0]))
            print("Test MSE: {}".format(it_res[2][1]))
            print("Y pushed out of bounds: {}/{}".format(
                outofboundsct, poisct))

            # if we don't make progress, decrease learning rate
            if (it_res[0] < it_res[1]):
                print("no progress")
                self.eta *= 0.75
                new_poisx, new_poisy = poisx, poisy
            else:
                poisx = new_poisx
                poisy = new_poisy

            if (it_res[0] > best_obj):
                best_poisx, best_poisy, best_obj = poisx, poisy, it_res[1]

            last_obj = it_res[1]

            towrite = [poisct, count, it_res[0], it_res[1] - it_res[0], \
                       it_res[2][0], it_res[2][1], \
                       (datetime.datetime.now() - tstart).total_seconds()]

            self.resfile.write(','.join([str(val) for val in towrite]) + "\n")
            self.trainfile.write("\n{},{}\n".format(poisct, count))

            for j in range(poisct):
                self.trainfile.write(','.join([str(val) for val in
                                               [new_poisy[j]] + new_poisx[j].tolist()[0]
                                               ]) + '\n')
            it_diff = abs(it_res[0] - it_res[1])

            # stopping conditions
            if (count >= 15 and (it_diff <= self.eps or count > 50)):
                break

            # visualization done - plotting time
            if (visualize and count >= 9):
                self.plot_path(clf_init, lam_init, eq7lhs, mu, \
                               poisx_hist, poisy_hist, newlogdir)
                break

        if workerpool:
            workerpool.close()

        return best_poisx, best_poisy

    def plot_path(self, clf, lam, eq7lhs, mu, \
                  poisx_hist, poisy_hist, newlogdir):
        """
        plot_path makes a pretty picture of the gradient descent path

        clf: initial model
        lam: regularization coef
        eq7lhs, mu: needed for gradient
        poisx_hist, poisy_hist: path of poisoning
        newlogdir: directory to save pretty picture to
        """

        plt.plot(self.x, self.y, 'k.')
        x_line = np.linspace(0, 1, 10)
        y_line = x_line * clf.coef_ + clf.intercept_
        plt.plot(x_line, y_line, 'k-')

        # plot function value colors
        self.plot_func(self.obj_comp)

        # plot gradient vector field
        self.plot_grad(clf, lam, eq7lhs, mu)

        # plot path of poisoning pt
        for i in range(poisx_hist.shape[1]):
            # plot start, path, and end
            plt.plot(poisx_hist[0, i, :], poisy_hist[0, i], 'g.', markersize=10)
            plt.plot(poisx_hist[:, i, :], poisy_hist[:, i], 'g-', linewidth=3)
            plt.plot(poisx_hist[-1, i, :], poisy_hist[-1, i], 'g*', markersize=10)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        # plt.tight_layout() # whatever looks better
        plt.savefig(os.path.join(newlogdir, 'vis2d.png'))

    def plot_func(self, f, min_x=0, max_x=1, \
                  resolution=0.1):
        """
        plot_func plots a heatmap of the objective function

        f: objective function
        min_x: smallest value of x desired in the heatmap
        max_x: largest value of x desired in the heatmap
        resolution: granularity of heatmap
        """

        xx1, xx2 = np.meshgrid(
            np.arange(min_x, max_x + resolution, resolution),
            np.arange(min_x, max_x + resolution, resolution))

        grid_x = np.array([xx1.ravel(), xx2.ravel()]).T
        z = np.zeros(shape=(grid_x.shape[0],))

        for i in range(grid_x.shape[0]):
            poisx = np.concatenate(
                (self.trnx, grid_x[i][0].reshape((1, 1))), axis=0)
            poisy = self.trny + [grid_x[i][1].item()]
            clf, lam = self.learn_model(poisx, poisy, None)
            z[i] = f(clf, lam, None)

        z = z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, z, 30, cmap='jet', alpha=0.7)
        plt.colorbar()
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

    def plot_grad(self, clf, lam, eq7lhs, mu, min_x=0, max_x=1, resolution=0.1):
        """
        plot_grad plots vector field for gradients of the objective function

        clf, lam, eq7lhs, mu: values needed for gradient computation
        min_x: smallest value of x desired in the vector field
        max_x: largest value of x desired in the vector field
        resolution: spacing between arrows
        """

        xx1, xx2 = np.meshgrid(
            np.arange(min_x, max_x + resolution, resolution),
            np.arange(min_x, max_x + resolution, resolution))

        grid_x = np.array([xx1.ravel(), xx2.ravel()]).T
        z = np.zeros(shape=(grid_x.shape[0], 2))

        for i in range(grid_x.shape[0]):
            clf, lam = self.learn_model(
                np.concatenate((self.trnx,
                                grid_x[i, 0].reshape((1, 1))),
                               axis=0),
                self.trny + [self.x[i, 1]], None)
            z[i, 0], z[i, 1] = self.comp_grad_dummy(eq7lhs, mu, clf, lam, \
                                                    self.x[i, 0].reshape((1, 1)), self.x[i, 1])

        U = z[:, 0]
        V = z[:, 1]
        plt.quiver(xx1, xx2, U, V)

    def comp_grad_dummy(self, eq7lhs, mu, clf, lam, poisx, poisy):
        """
        comp_grad_dummy computes gradients for visualization
        """
        m = self.compute_m(clf, poisx, poisy)

        wxc, bxc, wyc, byc = self.compute_wb_zc(eq7lhs, mu, clf.coef_, m, \
                                                self.samplenum, poisx)

        if (self.objective == 0):
            r = self.compute_r(clf, lam)
            otherargs = (r,)
        else:
            otherargs = None

        attack, attacky = self.attack_comp(clf, wxc, bxc, wyc, byc, otherargs)

        allattack = np.array(np.concatenate((attack, attacky), axis=1))
        allattack = allattack.ravel()
        norm = np.linalg.norm(allattack)
        allattack = allattack / norm if norm > 0 else allattack
        attack, attacky = allattack[:-1], allattack[-1]

        return attack, attacky

    def poison_data_subroutine(self, in_tuple):
        """
        poison_data_subroutine poisons a single poisoning point
        input is passed in as a tuple and immediately unpacked for
        use with the multiprocessing.Pool.map function

        poisxelem, poisyelem: poisoning point at the start
        eq7lhs, mu: values for computation
        clf, lam: current model and regularization coef
        """

        poisxelem, poisyelem, eq7lhs, mu, clf, lam = in_tuple
        m = self.compute_m(clf, poisxelem, poisyelem)

        # compute partials
        wxc, bxc, wyc, byc = self.compute_wb_zc(eq7lhs, mu, clf.coef_, m, \
                                                self.samplenum, poisxelem)

        if (self.objective == 0):
            r = self.compute_r(clf, lam)
            otherargs = (r,)
        else:
            otherargs = None

        attack, attacky = self.attack_comp(clf, wxc, bxc, wyc, byc, otherargs)

        # keep track of how many points are pushed out of bounds
        if (poisyelem >= 1 and attacky >= 0) \
                or (poisyelem <= 0 and attacky <= 0):
            outofbounds = True
        else:
            outofbounds = False

        # include y in gradient normalization
        if self.opty:
            allattack = np.array(np.concatenate((attack, attacky), axis=1))
            allattack = allattack.ravel()
        else:
            allattack = attack.ravel()

        norm = np.linalg.norm(allattack)
        allattack = allattack / norm if norm > 0 else allattack
        if self.opty:
            attack, attacky = allattack[:-1], allattack[-1]
        else:
            attack = allattack

        poisxelem, poisyelem, _ = self.lineSearch(poisxelem, poisyelem, \
                                                  attack, attacky)
        poisxelem = poisxelem.reshape((1, self.feanum))

        return poisxelem, poisyelem, outofbounds

    def computeError(self, clf):
        toterr, v_toterr = 0, 0
        rsqnum, v_rsqnum = 0, 0
        rsqdenom, v_rsqdenom = 0, 0

        w = np.reshape(clf.coef_, (self.feanum,))
        sum_w = np.linalg.norm(w, 1)

        mean = sum(self.tsty) / len(self.tsty)
        vmean = sum(self.vldy) / len(self.vldy)

        pred = clf.predict(self.tstx)
        vpred = clf.predict(self.vldx)

        for i, trueval in enumerate(self.vldy):
            guess = vpred[i]
            err = guess - trueval

            v_toterr += err ** 2  # MSE
            v_rsqnum += (guess - vmean) ** 2  # R^2 num and denom
            v_rsqdenom += (trueval - vmean) ** 2

        for i, trueval in enumerate(self.tsty):
            guess = pred[i]
            err = guess - trueval

            toterr += err ** 2  # MSE
            rsqnum += (guess - mean) ** 2  # R^2 num and denom
            rsqdenom += (trueval - mean) ** 2

        vld_mse = v_toterr / len(self.vldy)
        tst_mse = toterr / len(self.tsty)

        return vld_mse, tst_mse
        # computed a bunch of other stuff too
        # sum_w,rsqnum/rsqdenom,v_rsqnum/v_rsqdenom

    def lineSearch(self, poisxelem, poisyelem, attack, attacky):
        k = 0
        x0 = np.copy(self.trnx)
        y0 = self.trny[:]

        curx = np.append(x0, poisxelem, axis=0)
        cury = y0[:]  # why not?
        cury.append(poisyelem)

        clf, lam = self.learn_model(curx, cury, None)
        clf1, lam1 = clf, lam

        lastpoisxelem = poisxelem
        curpoisxelem = poisxelem

        lastyc = poisyelem
        curyc = poisyelem
        otherargs = None

        w_1 = self.obj_comp(clf, lam, otherargs)
        count = 0
        eta = self.eta

        while True:
            if (count > 0):
                eta = self.beta * eta
            count += 1
            curpoisxelem = curpoisxelem + eta * attack
            curpoisxelem = np.clip(curpoisxelem, 0, 1)
            curx[-1] = curpoisxelem

            if self.opty:
                curyc = curyc + attacky * eta
                curyc = min(1, max(0, curyc))
                cury[-1] = curyc
            clf1, lam1 = self.learn_model(curx, cury, clf1)
            w_2 = self.obj_comp(clf1, lam1, otherargs)

            if (count >= 100 or abs(w_1 - w_2) < 1e-8):  # convergence
                break
            if (w_2 - w_1 < 0):  # bad progress
                curpoisxelem = lastpoisxelem
                curyc = lastyc
                break

            lastpoisxelem = curpoisxelem
            lastyc = curyc
            w_1 = w_2
            k += 1

        for col in self.colmap:
            vals = [(curpoisxelem[0, j], j) for j in self.colmap[col]]
            topval, topcol = max(vals)
            for j in self.colmap[col]:
                if (j != topcol):
                    curpoisxelem[0, j] = 0
            if (topval > 1 / (1 + len(self.colmap[col]))):
                curpoisxelem[0, topcol] = 1
            else:
                curpoisxelem[0, topcol] = 0
        curx = np.delete(curx, curx.shape[0] - 1, axis=0)
        curx = np.append(curx, curpoisxelem, axis=0)
        cury[-1] = curyc
        clf1, lam1 = self.learn_model(curx, cury, None)

        w_2 = self.obj_comp(clf1, lam1, otherargs)

        return np.clip(curpoisxelem, 0, 1), curyc, w_2

    def iter_progress(self, lastpoisx, lastpoisy, curpoisx, curpoisy):
        x0 = np.concatenate((self.trnx, lastpoisx), axis=0)
        y0 = self.trny + lastpoisy
        clf0, lam0 = self.learn_model(x0, y0, None)
        w_0 = self.obj_comp(clf0, lam0, None)

        x1 = np.concatenate((self.trnx, curpoisx), axis=0)
        y1 = self.trny + curpoisy
        clf1, lam1 = self.learn_model(x1, y1, None)
        w_1 = self.obj_comp(clf1, lam1, None)
        err = self.computeError(clf1)

        return w_1, w_0, err

    # can compute l2 objective and grads already
    def comp_obj_new(self, clf, lam, otherargs):
        coef_diff = np.linalg.norm(clf.coef_ - self.initclf.coef)
        inter_diff = clf.intercept_ - self.initclf.intercept_
        return coef_diff ** 2 + inter_diff ** 2

    def comp_attack_l2(self, clf, wxc, bxc, wyc, byc, otherargs):
        initw, initb = self.initclf.coef_, self.initclf.intercept_
        curw, curb = clf.coef_, clf.intercept_

        attackx = np.dot(np.transpose(curw - initw), wxc) + (curb - initb) * bxc
        attacky = np.dot(curw - initw, wyc.T) + (curb - initb) * byc

        return attackx, attacky

    # unimplemented functions - handled by heirs
    def learn_model(self, x, y, clf):
        raise NotImplementedError

    def compute_sigma(self):
        raise NotImplementedError

    def compute_mu(self):
        raise NotImplementedError

    def compute_m(self, clf, w, b, poisxelem, poisyelem):
        raise NotImplementedError

    def compute_wb_zc(self, eq7lhs, mu, w, m, n):
        raise NotImplementedError

    def compute_r(self, clf, lam):
        raise NotImplementedError

    def comp_obj_trn(self, clf, lam, otherargs):
        raise NotImplementedError

    def comp_obj_vld(self, clf, lam, otherargs):
        raise NotImplementedError

    def comp_attack_trn(self, clf, wxc, bxc, wyc, byc, otherargs):
        raise NotImplementedError

    def comp_attack_vld(self, clf, wxc, bxc, wyc, byc, otherargs):
        raise NotImplementedError


def inf_flip(X_tr, Y_tr, count):
    Y_tr = np.array(Y_tr)
    inv_cov = (0.01 * np.eye(X_tr.shape[1]) + np.dot(X_tr.T, X_tr)) ** -1
    H = np.dot(np.dot(X_tr, inv_cov), X_tr.T)
    bests = np.sum(H, axis=1)
    room = .5 + np.abs(Y_tr - 0.5)
    yvals = 1 - np.floor(0.5 + Y_tr)
    stat = np.multiply(bests.ravel(), room.ravel())
    stat = stat.tolist()[0]
    print(stat)
    totalprob = sum(stat)
    allprobs = [0]
    poisinds = []
    for i in range(X_tr.shape[0]):
        allprobs.append(allprobs[-1] + stat[i])
    allprobs = allprobs[1:]
    for i in range(count):
        a = np.random.uniform(low=0, high=totalprob)
        poisinds.append(bisect.bisect_left(allprobs, a))

    return X_tr[poisinds], [yvals[a] for a in poisinds]


def main(args):
    init = args.initialization
    poi_train_x = pd.read_csv('train_X.csv')
    poi_train_y = pd.read_csv('train_y.csv')
    poi_test_x = pd.read_csv('test_X.csv')
    poi_test_y = pd.read_csv('test_y.csv')
    poi_val_x = pd.read_csv('val_X.csv')
    poi_val_y = pd.read_csv('val_y.csv')

    totprop = args.poisct / (args.poisct + args.trainct)
    poisx,poisy = inf_flip(poi_train_x, poi_train_y, int(args.trainct * totprop / (1 - totprop) + 0.5))
    print(poisx + " " + poisy)

#Main
if __name__=='__main__':
    print("starting poison ...\n")
    parser = setup_argparse()
    args = parser.parse_args()

    print("-----------------------------------------------------------")
    print(args)
    print("-----------------------------------------------------------")
    main(args)