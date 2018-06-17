# coding=utf-8

import collections
import configparser
import numpy as np
import warnings
import traceback
import sys
import Viewer
import os
from PIL import Image
import ExFileManager
import LINEUIManager


class PostProcessManager(object):
    """
    in charge of post processes.
    """

    def __init__(self, config_file_name):
        # Reads a config file
        ini_file = configparser.ConfigParser()
        ini_file.read(config_file_name)
        #self.threshold_h_1 = float(ini_file['Parameter']['threshold_h_1'])

        self.line_ui_manager = LINEUIManager.LINEUIManager()
        self.viewer = Viewer.Viewer()
        self.exfile_manager = ExFileManager.ExFileManager(config_file_name)

    def determine_fired_H(self, each_label_data, C, W):

        label_H = []
        for l_d in each_label_data:
            H_sum = np.zeros((1, C.shape[1]))
            for d in l_d[1]:
                H = self.softmax(C, np.array(d), W)

                H_sum += H
            H_sum = H_sum / len(l_d[1])
            H_sum[H_sum >= 0.9] = 1
            H_sum[H_sum < 0.9] = 0

            print(H_sum.tolist())
            label_H.append([l_d[0], H_sum.tolist()])

        self.exfile_manager.write_to_file('label_H', label_H)

        return label_H

    def softmax(self, C, X, W):

        input_sum = (np.dot(W, X.T)).T + C

        input_sum = input_sum - np.max(input_sum)
        exp_input_sum = np.exp(input_sum)
        sum_exp_input_sum = np.sum(exp_input_sum)

        return exp_input_sum / sum_exp_input_sum

    def array_to_image(self, array, *, image_size=(), store_path="", image_name="image", extension='jpg',
                       Line=False):
        """
        Changes array into image.
        :param array: array to be changed into image
        :param image_size: size of images. valid in 1-d list or numpy array.
        :param store_path: path for staring image
        :param image_name: name of image
        :param extension: extension of image such as jpg,png...
        :param Line: if True, the image will be sent to Line
        :return: None
        """

        if isinstance(array, np.ndarray):
            pass

        elif isinstance(array, list):
            array = np.array(array)

        else:
            self.viewer.display_message(
                "array_to_image Error: data_array should be list or numpy array type.\n")

        if array.ndim == 1:
            array = np.reshape(array, image_size)
            name = '%s.%s' % (image_name, extension)
            path = os.path.join(store_path, name)
            Image.fromarray(np.uint8((array / np.max(array)) * 255)).save(path)

            if Line:
                self.line_ui_manager.send_line(name)

        elif array.ndim == 2:
            name = '%s.%s' % (image_name, extension)
            path = os.path.join(store_path, name)
            Image.fromarray(np.uint8((array / np.max(array)) * 255)).save(path)

            if Line:
                self.line_ui_manager.send_line(name)

        elif array.ndim == 3:

            for index, a in enumerate(array):
                name = '%s_%d.%s' % (image_name, index, extension)
                path = os.path.join(store_path, name)
                Image.fromarray(np.uint8((a / np.max(a)) * 255)).save(path)

                if Line:
                    self.line_ui_manager.send_line(name)
        else:
            self.viewer.display_message(
                "array_to_image Error: array should be at most 3-d.\n")

    def save_numpy_array(self, array, filename, *, path=''):
        """

        :param array: numpy array
        :param filename: name for saving
        :param path: path where array will be saved
        :return: None
        """
        name = os.path.join(path, filename)
        try:
            np.save('%s.npy' % name, array)
        except FileNotFoundError:
            self.viewer.display_message(
                "FileNotFoundError : No such file or directory\n"
                "The array was not saved.")
