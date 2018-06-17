# coding=utf-8

import Viewer
import ExFileManager
import LINEUIManager
import PreProcessManager
import PostProcessManager

import configparser
import os
import time
from PIL import Image
from datetime import datetime as dt
import shutil
import numpy as np
import GaussianBinaryRBM
from sklearn.model_selection import train_test_split


class Controller(object):
    """
    Controls main process.
    """

    def __init__(self, config_file_name):

        # Reads a config file
        self.config_file_name = config_file_name
        ini_file = configparser.ConfigParser()
        ini_file.read(self.config_file_name)

        # Generates objects
        self.viewer = Viewer.Viewer()
        self.line_ui_manager = LINEUIManager.LINEUIManager()
        self.pre_precess_manager = PreProcessManager.PreProcessManager(self.config_file_name)
        self.post_process_manager = PostProcessManager.PostProcessManager(self.config_file_name)
        self.exfile_manager = ExFileManager.ExFileManager(self.config_file_name)

        # Directory names where data will be stored
        self.path_data_dirs = ''.join(ini_file['Setting']['path_data_dirs'])

        # For learning
        self.mini_batch_size = int(ini_file.get('GeneralParameter', 'mini_batch_size'))

        # dir name for saving result
        self.dir_for_saving_result = os.path.join('Result', dt.now().strftime('%Y-%m-%d-%H-%M-%S'))

        path_image_to_get_size = ini_file.get('Setting', 'path_image_to_get_size')
        self.width, self.height = self.exfile_manager.get_image_width_height(path_image_to_get_size)

    def start_main_process(self):
        """
        Main Process:
            1. Preparation
            2. Reads data
            3. Pre-process of data
            4. Contrastive Divergence Learning
            5. Saves results
            6. Post-process
            7. Test
            8. Finished!!
        :return: None
        """

        ##################
        # 1. Preparation #
        ##################

        self.preparation()

        #################
        # 2. Reads data #
        #################

        all_labels, all_data, each_label_data = self.read_data(self.path_data_dirs)

        ##########################
        # 3. Pre-process of data #
        ##########################

        data_train_batch, label_train_batch, data_train, data_test = self.pre_process_data(all_labels, all_data,
                                                                                           self.mini_batch_size)

        ######################################
        # 4. Contrastive Divergence Learning #
        ######################################

        C, B, W, sigma = self.CD_learning(data_train_batch)

        ####################
        # 5. Saves results #
        ####################

        self.save_result(C, B, W, sigma)

        ###################
        # 6. Post-Process #
        ###################

        self.post_process_manager.determine_fired_H(each_label_data, C, W)

        ###########
        # 7. Test #
        ###########
        # self.test(C, W, sigma, label_list, H_list)

        ##################
        # 8. Finished !! #
        ##################
        self.line_ui_manager.send_line("RBM Finished")
        self.viewer.display_message("\nRBM Finished \n")

    def preparation(self):
        """
        Preparation.
        [1] Makes a dir for saving results.
        [2] Copies config file
        :return: None

        """
        # [1] Makes a dir for saving results.
        # if 'Result' dir already exists,
        # a 'temporary' dir will be made.

        try:
            os.mkdir(self.dir_for_saving_result)
        except FileExistsError:
            self.viewer.display_message("Made a temporary directory.")
            self.dir_for_saving_result = 'temporary'
            os.mkdir('temporary')

        # [2] Copies config file into the same dir as the one where results will be stored
        shutil.copy2(self.config_file_name, self.dir_for_saving_result)

    def read_data(self, path_train_dirs):
        """
        Reads data.
        :return: num_all_data, all_data_array, each_label_data_array
        """

        self.viewer.display_message("\nData Read Starts...\n")

        # num_all_data: float scalar, total amount of data
        # all_data_array: 2-d list (dim: num_all_data * each data dimension)
        # each_label_data_array: 3-d list (dim: num_label*num_all_data * each data dimension)
        formatted_labels, formatted_data, each_label_data = self.exfile_manager.read_image_data(path_train_dirs)

        self.viewer.display_message("\nData Read Finished...\n")

        return formatted_labels, formatted_data, each_label_data

    def pre_process_data(self, all_labels, all_data, mini_batch_size):
        """
        Pre-process od data
        [1] Normalizes data (changes it into data with mean 0 and variance 1)
        [2] Makes mini batches
        [3] Makes a dictionary with keys for learning
        :param all_data_array: 2-d list (dim: num all data * data dimension)
        :return: all_data, dict_data_parameter
        """

        # [1] Normalizes data
        all_data = self.pre_precess_manager.normalization(all_data)

        data_train, data_test, label_train, label_test = train_test_split(all_data, all_labels, test_size=0.1,
                                                                          shuffle=True)

        # [2] makes mini-batches where data with different labels will be contained at the almost same rate.
        data_train_batch = self.pre_precess_manager.make_mini_batch(data_train, mini_batch_size)
        label_train_batch = self.pre_precess_manager.make_mini_batch(label_train, mini_batch_size)

        return data_train_batch, label_train_batch, data_train, data_test

    def CD_learning(self, mini_batch):
        """
        Contastive divergence learning
        :param dict_data_parameter: dictionary with keys for learning
        :return: C, B, W, sigma
        """

        ini_file = configparser.ConfigParser()
        ini_file.read(self.config_file_name)

        ######################
        # General Parameters #
        ######################

        epoch = int(ini_file.get('GeneralParameter', 'Epoch'))

        num_visible_units = self.width * self.height
        num_hidden_units = int(ini_file.get('GeneralParameter', 'Num_Hidden_Unit'))
        learning_rate = float(ini_file.get('GeneralParameter', 'Learning_Rate'))

        ############
        # sampling #
        ############
        sampling_times = int(ini_file.get('SpecialParameter', 'Smapling_Times'))

        sampling_type = ''.join(ini_file['SpecialParameter']['Smapling_Type'])
        if not (sampling_type == 'CD' or sampling_type == 'PCD'):
            self.viewer.display_message("set CD or PCD to a param of sampling_type")
            raise Exception

        ############
        # momentum #
        ############
        if ''.join(ini_file['SpecialParameter']['Momentum']) == 'Yes':
            momentum_rate = float(ini_file.get('SpecialParameter', 'Momentum_Rate'))
        else:
            momentum_rate = 0

        ################
        # weight_decay #
        ################
        if ''.join(ini_file['SpecialParameter']['Weight_Decay']) == 'Yes':
            weight_decay_rate = float(ini_file.get('SpecialParameter', 'Weight_Decay_Rate'))
        else:
            weight_decay_rate = 0

        #########################
        # sparse_regularization #
        #########################
        if ''.join(ini_file['SpecialParameter']['Sparse_Regularization']) == 'Yes':

            sparse_regularization = (float(ini_file.get('SpecialParameter', 'Sparse_Regularization_Target')),
                                     float(ini_file.get('SpecialParameter', 'Sparse_Regularization_Rate')))

        else:
            sparse_regularization = (0, 0)

        width_sf = int(ini_file.get('SpecialParameter', 'Width_Spread_Function'))
        height_sf = int(ini_file.get('SpecialParameter', 'Height_Spread_Function'))
        num_sf = num_hidden_units
        # CD Learning
        # Will get learned numpy arrays
        start = time.time()

        GBRBM = GaussianBinaryRBM.GaussianBinaryRBM(mini_batch, epoch, num_visible_units,
                                                    num_hidden_units,
                                                    sampling_type, sampling_times,
                                                    learning_rate,
                                                    momentum_rate, weight_decay_rate,
                                                    sparse_regularization, width_sf, height_sf, num_sf,self.dir_for_saving_result)

        self.viewer.display_message("\nContrastive Divergence Learning Starts...\n")

        C, B, W, sigma = GBRBM.Learning()

        # Measures time
        elapsed_time = time.time() - start
        h = elapsed_time // 3600
        m = (elapsed_time - h * 3600) // 60
        s = elapsed_time - h * 3600 - m * 60
        self.viewer.display_message(
            "\nContrastive Divergence Learning Finished...\n" + "About %d h %d m %d s \n" % (h, m, s))
        self.line_ui_manager.send_line("\nCD Learning \nAbout %d h %d m %d s" % (h, m, s))

        return C, B, W, sigma

    def save_result(self, C, B, W, sigma):
        """
        [1] Saves learning results
        [2] makes images of arrays
        :param C: biases of hidden units
        :param B: biases of visible units
        :param sigma: variance of gaussian
        :param W: weight
        :return: None
        """

        self.viewer.display_message("Results saving Starts\n")

        # [1] Saves learning results
        self.exfile_manager.numpy_array_save(os.path.join(self.dir_for_saving_result, 'C'), C)
        self.exfile_manager.numpy_array_save(os.path.join(self.dir_for_saving_result, 'B'), B)
        self.exfile_manager.numpy_array_save(os.path.join(self.dir_for_saving_result, 'sigma'), sigma)
        self.exfile_manager.numpy_array_save(os.path.join(self.dir_for_saving_result, 'W'), W)

        # [2] makes images of arrays
        for index, W in enumerate(W):
            path_to_file = os.path.join(self.dir_for_saving_result, 'W_%d.jpg' % index)

            Image.fromarray(
                np.uint8(np.reshape((W / np.max(W)) * 255, (self.width, self.height)))).save(
                path_to_file)

            self.line_ui_manager.send_line('W_%d.jpg' % index, path_to_file)

        self.viewer.display_message("Results saving Starts Finished \n")

    def test(self, C, W, sigma, label_list, H_list):
        """
        Test.
        Reads test data.
        Get H-s corresponding to the data.
        Campares the H-s to the learned H-s
        And estimates labels to the H-s.
        :param C: biases of hidden units
        :param W: weight
        :param sigma: variance of gaussian
        :param label_list: columns correspond to each H in the same index, respectively
        :param H_list: columns correspond to lables in the same index, respectively
        :return: None
        """

        for a_dir_test in os.listdir(self.path_test_dir):

            num_data, data_array = self.exfile_manager.get_data(os.path.join(self.path_test_dir, a_dir_test))
            estimated_labels = self.post_process_manager.estimate_data_category(C, W, sigma, label_list, H_list,
                                                                                data_array)
            for (index, data), estimated_label in zip(enumerate(data_array), estimated_labels):
                path_to_file = os.path.join(self.dir_for_saving_result, '%s_%d.jpg' % (a_dir_test, index))

                data = (data / np.max(data)) * 255

                Image.fromarray(np.uint8(
                    (np.reshape(data, (self.width, self.height))))).save(
                    path_to_file)
                self.line_ui_manager.send_line(
                    'RBM: %s\nTRUE: %s\n' % (estimated_label, a_dir_test,),
                    path_to_file)
