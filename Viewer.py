# coding=utf-8

class Viewer(object):
    """
    User interface
    """

    def __init__(self):
        pass

    def display_message(self, message):
        """
        Standard output of message
        :param message: numeric type or string type
        :return: None
        """
        print(message)

    def get_user_input(self):
        """
        Returns standard input form user
        :return:string
        """
        return input()
