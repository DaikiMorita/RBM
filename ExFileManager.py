# coding=utf-8


import configparser
import os
import tqdm
import numba
import numpy as np
from PIL import Image
from skimage import io


class ExFileManager(object):
    """
    Manages external files
    """

    def __init__(self, config_file_name):

        # Reads .config.ini
        ini_file = configparser.ConfigParser()
        ini_file.read(config_file_name)

        # self.depth = float(ini_file.get('GeneralParameter', 'depth'))

    def read_image_data(self, path_all_dirs):
        """
        Reads data.
        This method is applied especially when you try to read "Image"s.
        :return: num_all_data, formated_data, each_label_data
        """

        num_all_data = 0
        all_data = []
        all_labels = []
        each_label_data = []

        for dir in os.listdir(path_all_dirs):
            num_data, data = self.get_data(os.path.join(path_all_dirs, dir))
            labels = [dir] * num_data

            all_labels.append(labels)
            all_data.append(data)
            each_label_data.append([dir, data])

        # all_data_array above was organized like [[data with label A],[data with label B]....]
        # a format like [data with label A, data with label B,...] is easy to use.
        # For example, for shaffling or making mini-batch.
        formatted_labels = []
        formatted_data = []
        for labels, data in zip(all_labels, all_data):

            for label in labels:
                formatted_labels.append(label)

            for datum in data:
                formatted_data.append(datum)

        return formatted_labels, formatted_data, each_label_data

    def get_data(self, path_to_data):
        """
        Gets all data in the all dirs which exist in the specified dir.
        :param path_to_data: path to the dir where all dirs with data exist
        :return: 1st: a float scalar meaning the total amount of data
                 2nd: numpy array where all data are stored. Each row corresponds to an data
        """

        # Counts up the number of non-0-size data.
        num_data = self.count_up_data_num(path_to_data)

        data_array = []

        for data in tqdm.tqdm(os.listdir(path_to_data)):

            # skip if the file's sile is 0
            try:
                if os.stat(os.path.join(path_to_data, data)).st_size != 0:
                    data_array.append(self.image_data_pre_process(os.path.join(path_to_data, data)))
            except OSError:
                num_data -= 1

        return num_data, data_array

    def count_empty_file(self, dir):
        """
        Counts up the number of empty files.
        :param dir: the folder where files of which you want to know the amount exist.
        :return: amount
        """

        return len([index for index, file in enumerate(os.listdir(dir)) if
                    os.stat(os.path.join(dir, file)).st_size == 0])

    def count_up_data_num(self, dir):
        """
        Counts up the number of non-0-size files in a folder
        :param dir: folder name where you want to know the amount of files
        :return: the number of files in the folder
        """
        return len(os.listdir(dir)) - self.count_empty_file(dir)

    def image_data_pre_process(self, path):

        # flatten
        # img = Image.open(path)
        img = Image.open(path)
        width, height = img.size
        return [img.getpixel((i, j)) / 255 for j in range(height) for i in range(width)]

    def numpy_array_save(self, filename, array):
        """
        Saves numpy array into a directory.
        :param filename: filename
        :param array: array to be saved
        :return: None
        """

        np.save('%s.npy' % filename, array)

    def get_image_width_height(self, path):
        """

        :param path:
        :return:
        """

        img = Image.open(path)
        width, height = img.size

        return width, height

    def write_to_file(self, filename, data):
        """

        :param filename:
        :param data:
        :return:
        """

        with open(filename, mode='a', encoding='utf-8') as fh:
            fh.write('%s\n' % data)
