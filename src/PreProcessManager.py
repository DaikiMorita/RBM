# coding=utf-8

import collections
import configparser
import numpy as np
import Viewer
import random


class PreProcessManager(object):
    """
    in charge of pre processes.
    """

    def __init__(self, config_file_name):

        self.viewer = Viewer.Viewer()

    def normalization(self, data_array):
        """
        Normalizes data (changes it into data with mean 0 and variance 1)
        :param data_array: data array
        :return: normalized data_array
        """

        data_original = np.array(data_array)

        average = np.sum(data_original, axis=1) / data_original.shape[1]

        data_minus_average = np.array([d - a for d, a in zip(data_array, average)])

        sigma = np.sqrt(np.sum(np.power(data_minus_average, 2), axis=1) / data_minus_average.shape[1])
        sigma[sigma == 0] = 0.001

        data_array = np.array([d / s for d, s in zip(data_minus_average, sigma)])

        return data_array.tolist()

    def make_mini_batch(self, data_array, mini_batch_size):
        """
        makes mini bathes with data_array
        :param data_array:
        :return:2d-list
        """

        # Now that data_array was shuffled,
        # mini batches will contain data with a different label at the almost same rate, statistically,
        # even when mini batches are made by extracting data from the top of the list 'data_array'

        data_array_length = len(data_array)
        rest = data_array_length % mini_batch_size

        mini_batches = [data_array[i:i + mini_batch_size] for i in
                        range(0, data_array_length - rest, mini_batch_size)]

        return mini_batches

    def decorrelation(self, data_array):

        if isinstance(data_array, np.ndarray):
            data_type = 'numpy'
        elif isinstance(data_array, list):
            data_type = 'list'
            data_array = np.array(data_array)
        else:
            self.viewer.display_message(
                "Decorrelation Error: data_array should be list or numpy array type.\n")
            raise Exception

        if data_array.ndim == 2:

            # variance-covariance matrix
            sigma = np.cov(data_array)

            # eigen-vectors
            _, eig_vectors = np.linalg.eig(sigma)

            # linear transformation
            return np.dot(eig_vectors.T, data_array.T).T


        elif data_array.ndim == 3:

            flattened_array = np.empty((1, data_array.shape[0]))
            decorrelated_array = np.empty((1, data_array.shape[0]))

            row = data_array.shape[1]
            column = data_array.shape[2]

            for index, data in enumerate(data_array):
                data = np.reshape(data, (1, row * column))

                flattened_array[index] = data

            # variance-covariance matrix
            sigma = np.cov(flattened_array)

            # eigen-vectors
            _, eig_vectors = np.linalg.eig(sigma)

            # linear transformation
            decorrelated_data = np.dot(eig_vectors.T, decorrelated_array.T).T

            return np.array([np.reshape(data, (row, column)) for data in decorrelated_array])

        else:
            self.viewer.display_message(
                "Decorrelation Error: data_array should be 2 or 3 dimension.\n")
            raise Exception

    def display_image(self):
        pass
