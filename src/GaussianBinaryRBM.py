# coding=utf-8

import LINEUIManager
import Viewer
import tqdm
import numpy as np
import warnings
import math
from PIL import Image
import os


class GaussianBinaryRBM(object):
    def __init__(self, mini_batches, epoch, num_visible_units, num_hidden_units, sampling_type, sampling_times,
                 learning_rate,
                 momentum_rate, weight_decay_rate, sparse_regularization, width_sf, height_sf, num_sf,dir_for_saving_result):

        # Generates objects
        self.viewer = Viewer.Viewer()
        self.line_ui_manager = LINEUIManager.LINEUIManager()

        # Parameters
        self.mini_batches = mini_batches
        self.epoch = epoch
        self.num_visible_units = num_visible_units
        self.num_hidden_units = num_hidden_units
        self.sampling_type = sampling_type
        self.sampling_times = sampling_times
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.weight_decay_rate = weight_decay_rate
        self.sparse_regularization_target = sparse_regularization[0]
        self.sparse_regularization_rate = sparse_regularization[1]
        self.width_spread_function = width_sf
        self.height_spread_function = height_sf
        self.dir_for_saving_result = dir_for_saving_result

        side_len_img = int(math.sqrt(num_visible_units))
        self.spread_funcs = np.array(
            [self.spread_function((side_len_img, side_len_img),
                                  (np.random.randint(0, side_len_img - 1), np.random.randint(0, side_len_img - 1)), 10,
                                  10) for _ in list(range(num_sf))])
        """
                for index, sf in enumerate(spread_funcs):
            print(sf)
            Image.fromarray(
                np.uint8(np.reshape(sf * 255, (side_len_img, side_len_img)))).save(os.path.join('SF', '%s.jpg' % index))
        """

    def Learning(self):

        """
       Contrastive Divergence Learning
       :param eta:learning rate
       :param mu: rate of change of momentum
       :param num_gibbs_sampling: number_of gibbs_sampling
       :param epoch: epoch:number of iteration
       :return: Learned C, B, W and sigma
       """

        # Initialization
        # C: biases of hidden units(dim 1 * num hidden)
        # B: biases of visible units(dim 1 * num visible)
        # W: weight (dim num hidden * num visible)
        # sigma: scalar or numpy array (dim 1 * visible units)
        C_new = np.zeros((1, self.num_hidden_units))
        B_new = np.zeros((1, self.num_visible_units))
        W_new = np.random.rand(self.num_hidden_units, self.num_visible_units)
        sigma_new = np.array([1])

        # For momentum
        delta_C = 0
        delta_B = 0
        delta_W = 0

        rho_new = 0

        # initialization of X_k
        X_k = np.array(self.mini_batches[0])

        # CD Learning
        for e in tqdm.tqdm(range(0, self.epoch)):

            #####################
            # Line Notification #
            #####################
            if e % (self.epoch * 0.2) == 0:
                self.line_ui_manager.send_line("epoch: %s / %s" % (e, self.epoch))

#            if e % (self.epoch * 0.01) == 0:
#
#                for i in range(W_new.shape[0]):
#                    path_to_file = os.path.join(self.dir_for_saving_result, 'W_%d_%d.jpg' % (i, e))
#                    Image.fromarray(
#                        np.uint8(np.reshape((W_new[i] / np.max(W_new[i])) * 255, (28, 28)))).save(
#                        path_to_file)

            #####################
            # 1. Gibbs Sampling #
            #####################

            X = np.array(self.mini_batches[int(e % len(self.mini_batches))])

            if self.sampling_type == 'CD':
                X_k = self.BlockGibbsSampling(X, C_new, B_new, W_new, sigma_new)

            elif self.sampling_type == 'PCD':
                X_k = self.BlockGibbsSampling(X_k, C_new, B_new, W_new, sigma_new)

            ######################
            # 2. Gradient Update #
            ######################

            P_H_1_X = self.prob_H_1_X(X, C_new, W_new, sigma_new)
            P_H_1_X_k = self.prob_H_1_X(X_k, C_new, W_new, sigma_new)

            rho_old = rho_new
            C_old = C_new
            B_old = B_new
            # W_old = W_new * self.spread_funcs
            W_old = W_new
            sigma_old = sigma_new

            rho_new, grad_E_sparse_W, grad_E_sparse_C = self.sparse_regularization(X,
                                                                                   C_old, W_old,
                                                                                   sigma_old, rho_old)

            C_new = C_old + self.learning_rate * (self.CD_C(P_H_1_X, P_H_1_X_k)
                                                  - self.sparse_regularization_rate * grad_E_sparse_C) \
                    + self.momentum_rate * delta_C

            B_new = B_old + self.learning_rate * self.CD_B(X, X_k, sigma_old) \
                    + self.momentum_rate * delta_B

            W_new = W_old + self.learning_rate * (self.CD_W(X, X_k, P_H_1_X, P_H_1_X_k,
                                                            sigma_old) - self.weight_decay_rate * W_old - self.sparse_regularization_rate * grad_E_sparse_W) \
                    + self.momentum_rate * delta_W

            sigma_new = sigma_old

            delta_C = C_new - C_old
            delta_B = B_new - B_old
            delta_W = W_new - W_old

        # non negative limitation
        W_new[W_new < 0] = 0

        return C_new, B_new, W_new, sigma_new

    def spread_function(self, img_size, center, width, height):
        """

        :param img_size:
        :param center:
        :param width:
        :param height:
        :return:
        """

        h = (height - 1) / 2
        w = (width - 1) / 2
        (r, c) = img_size
        (i, j) = center

        start_x = i - h
        start_y = j - w

        index = []
        for step_y in list(range(0, height)):
            for step_x in list(range(0, width)):

                x = start_x + step_x
                y = start_y + step_y

                if x >= 0 and y >= 0:
                    index.append([x, y])

        sf = np.zeros((r, c))

        for i in index:

            x = int(i[0])
            y = int(i[1])

            try:
                sf[x][y] = 1
            except IndexError:
                pass

        return sf.reshape(r * c)

    def sparse_regularization(self, X, C, W, sigma, rho_old):
        """

        :param sparse_regularization:
        :param X:
        :param C:
        :param W:
        :param sigma:
        :param P_H_1_X:
        :param H_X:
        :param sparse_regularization_target:
        :param rho_old:
        :return:
        """

        N = X.shape[0]

        # dim: 1 * num_hidden_units
        rho_new = 0.9 * rho_old + 0.1 * np.sum(self.prob_H_1_X(X, C, W, sigma), axis=0) / N

        delta_E_sparse_C = (-self.sparse_regularization_target / rho_new + (1 - self.sparse_regularization_target) / (
            1 - rho_new)) / N
        delta_E_sparse_C = delta_E_sparse_C[np.newaxis, :]

        S = np.empty((X.shape[0], delta_E_sparse_C.shape[1], X.shape[1]))
        for index, x in enumerate(X):
            S[index, :, :] = np.dot(delta_E_sparse_C.T, x[np.newaxis, :])

        delta_E_sparse_W = np.sum(S, axis=0) / N

        return rho_new, delta_E_sparse_W, delta_E_sparse_C

    def BlockGibbsSampling(self, X, C, B, W, sigma):
        """
        Block Gibbs Sampling
        :param num_gibbs_sampling: number of sampling
        :param X: values of visible (dim: num data * num visible units)
        :param C: biases of hidden units(dim 1 * num hidden)
        :param B: biases of visible units(dim 1 * num visible)
        :param W: weight (dim num hidden * num visible)
        :param sigma: scalar or numpy array (dim 1 * visible units)
        :return: sampled and averaged visible values X
        """

        temp = np.zeros((X.shape[0], X.shape[1]))
        X_k = X
        for _ in list(range(0, self.sampling_times)):
            H_k_1_X = self.prob_H_1_X(X_k, C, W, sigma)
            H_k = self.sampling_H_X(H_k_1_X)
            X_k = self.sampling_X_H(H_k, B, W, sigma)
            temp += X_k

        return temp / self.sampling_times

    def prob_H_1_X(self, X, C, W, sigma):
        """
        A row is a vector where i-th is the probability of h_i becoming 1 when given X
        :param X: values of visible (dim: num data * num visible units)
        :param C: biases of hidden units(dim 1 * num hidden)
        :param W: weight (dim num hidden * num visible)
        :param sigma: scalar or numpy array (dim 1 * visible units)
        :return: numpy array (dim: num data * num hidden)
        """

        warnings.filterwarnings('error')
        try:

            return 1 / (1 + np.exp(-C - (np.dot(X, np.transpose(W))) / (sigma * sigma)))

        except RuntimeWarning as warn:

            # Over float is interpreted as RuntimeWarning.
            # An array filled with 0 will be returned instead of the array with over floated number.
            self.viewer.display_message("Overfloat happened in calculating prob_h_1_X")
            return np.zeros((X.shape[0], W.shape[0]))

    def sampling_H_X(self, P_H_1):
        """
        Gets samples of H following Bernoulli distribution when given X
        :param P_H_1: probability of H becoming 1 when given X
        :return: array (dim: num_data*num_hidden_units)
        """

        return np.fmax(np.sign(P_H_1 - np.random.rand(P_H_1.shape[0], P_H_1.shape[1])),
                       np.zeros((P_H_1.shape[0], P_H_1.shape[1])))

    def sampling_X_H(self, H, B, W, sigma):
        """
        Gets samples of X following Gaussian distribution when given H
        :param H: values of hidden (dim: num data * num hidden)
        :param B: biases of visible (dim: num data * num visible)
        :param W: weight (dim num hidden * num visible)
        :param sigma: scalar or numpy array (dim 1 * visible units)
        :return: numpy array (dim: num data * num visible)
        """

        return sigma * np.random.randn(H.shape[0], W.shape[1]) + B + np.dot(H, W)

    def CD_C(self, P_H_1_X, P_H_1_X_k):
        """
        Gradient approximation of C
        :param P_H_1_X: probability of H becoming 1 when given X
        :param P_H_1_X_k: probability of H becoming 1 when given X_k
        :return: numpy vector (dim: 1 * num_hidden_units)
        """

        return np.sum(P_H_1_X - P_H_1_X_k, axis=0) / P_H_1_X.shape[0]

    def CD_B(self, X, X_k, sigma):
        """
        Gradient approximation of B
        :param B: biases of visible (dim: num data * num visible)
        :param X_k: values of sampled visible (dim: num data * num visible units)
        :return: numpy vector (dim: 1 * num_visible_units)
        """

        return (np.sum(X - X_k, axis=0)) / (X.shape[0] * sigma * sigma)

    def CD_W(self, X, X_k, P_H_1_X, P_H_1_X_k, sigma):
        """
        Gradient approximation of W
        :param X: values of  visible (dim: num data * num visible units)
        :param X_k: values of sampled visible (dim: num data * num visible units)
        :param P_H_1_X: probability of H becoming 1 when given X
        :param P_H_1_X_k: probability of H becoming 1 when given X_k
        :return: numpy array(dim: num_hidden_units * num_visible_units)
        """

        # Numpy array was faster in some experiments.
        E = np.empty((X.shape[0], P_H_1_X.shape[1], X.shape[1]))

        for index, (P_x, x, P_x_k, x_k) in enumerate(zip((P_H_1_X),
                                                         (X),
                                                         (P_H_1_X_k),
                                                         (X_k))):
            E[index, :, :] = np.dot(P_x[:, np.newaxis], x[np.newaxis, :]) - np.dot(P_x_k[:, np.newaxis],
                                                                                   x_k[np.newaxis, :])

        return np.sum(E, axis=0) / (X.shape[0] * sigma * sigma)

    def CD_sigma(self, X, X_k, P_H_1_X, P_H_1_X_k, B, W, sigma):
        """
        Gradient approximation of sigma
        :param X: values of  visible (dim: num data * num visible units)
        :param X_k: values of sampled visible (dim: num data * num visible units)
        :param P_H_1_X: probability of H becoming 1 when given X
        :param P_H_1_X_k: probability of H becoming 1 when given X_k
        :param B: array (dim: num_data, num_visible_units)
        :param W: weight (dim num hidden * num visible)
        :param sigma: scalar or numpy array (dim 1 * visible units)
        :return: numpy array (dim: 1)
        """
        E_1_1 = np.sum(np.diag(np.dot((X - B), np.transpose((X - B)))), axis=0)
        E_1_2 = np.sum(np.diag(np.dot(X, np.transpose(W)) * P_H_1_X))

        E_2_1 = np.sum(np.diag(np.dot((X_k - B), np.transpose((X_k - B)))), axis=0)

        E_2_2 = np.sum(np.diag(np.dot(X_k, np.transpose(W)) * P_H_1_X_k))

        return (E_1_1 - 2 * E_1_2 - E_2_1 + 2 * E_2_2) / (X.shape[0] * sigma * sigma * sigma)
