# coding=utf-8
import os
import requests


class LINEUIManager(object):
    """
    Line interaction
    """

    def __init__(self):
        pass

    def send_line(self, message, path_file=""):
        """
        sends a line to me
        :return:
        """

        line_notify_token = 'XsRTUJeMtkozOo3wV3BV2BKi0fGe2dLaMQt7GY3mzSR'
        line_notify_api = 'https://notify-api.line.me/api/notify'

        payload = {'message': '\n' + os.uname()[1] + '\n\n' + message}
        headers = {'Authorization': 'Bearer ' + line_notify_token}
        files = {}

        if path_file != "":
            try:
                with open(path_file, "rb") as file:
                    files = {"imageFile": file}
                    requests.post(line_notify_api, headers=headers, data=payload, files=files)
            except Exception:
                print("Line could not be sent.")

        else:
            try:
                requests.post(line_notify_api, headers=headers, data=payload)
            except Exception:
                print("Line could not be sent.")
