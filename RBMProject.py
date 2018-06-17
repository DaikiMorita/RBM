# coding=utf-8

import Controller
import sys


class RBMProject(object):
    """
    Restricted Boltzmann Machine as a feature extraction machine.

    """

    def __init__(self, config_file_name):
        Controller.Controller(config_file_name).start_main_process()


if __name__ == '__main__':
    # sys.argv[1]: .config.ini file

    RBMProject = RBMProject(sys.argv[1])
