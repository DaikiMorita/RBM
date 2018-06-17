# coding: utf-8

import wx
import configparser
import sys
import pandas as pd
import os
import shutil


class ConfigGenerater(wx.Frame):
    """ We simply derive a new class of Frame. """

    def __init__(self, parent, title, config_file_name):

        #######################
        # Application Setting #
        #######################

        # Reads a config file
        ini = configparser.ConfigParser()
        ini.read(config_file_name)

        self.l_section = self.read_section(ini)
        self.l_data_frame = []
        for s in self.l_section:
            k_v = self.read_key_value(ini, s)
            self.l_data_frame.append(self.make_data_frame(s, k_v))

        self.margin = 10
        self.width_section_box = 400
        self.width_section = 250
        self.margin_section_upper = 10
        self.margin_left_section = (self.width_section_box - self.width_section) / 2
        self.height_section = 30
        self.width_key = 250
        self.height_key = 30
        self.value_position = 250
        self.l_textctrl = []
        self.max_n_key = 0
        self.width_run_button = 150
        self.height_run_button = 100
        self.width_exit_button = 150
        self.height_exit_button = 100

        # Creates a dialog
        width, height = self.design_dialog(self.l_data_frame)

        wx.Frame.__init__(self, parent, title=title, size=(width + 30, height + 30))
        self.Center()

        self.panel = wx.Panel(self, pos=(0, 0), size=(width + 30, height + 30))
        self.panel.SetBackgroundColour("LIGHT BLUE")
        self.layout = wx.BoxSizer(wx.VERTICAL)
        self.main_process()

    def main_process(self):
        """
        Shows text, puts run and exit button on the dialog
        :return: None
        """
        self.show_text(self.l_data_frame)
        self.run_button()
        self.exit_button()
        self.Show(True)

    def show_text(self, l_data_frame):
        """
        Shows text on the dialog
        :param l_data_frame:
        :return:
        """

        for i, s_k_v in enumerate(l_data_frame):

            # Position of section
            x_s = self.margin + self.margin_left_section + self.width_section_box * i
            y_s = self.margin + self.margin_section_upper
            self.set_static_text('[' + s_k_v.columns[0] + ']', x_s, y_s)

            for t, (k, v) in enumerate(zip(s_k_v[s_k_v.columns[0]].index, s_k_v[s_k_v.columns[0]].values)):
                # Position of key and value
                x_k = x_s
                y_k = y_s + (t + 1) * self.height_key
                self.set_static_text(k, x_k, y_k)
                self.set_text_ctrl(v, x_k + self.value_position, y_k)

    def make_data_frame(self, section, key_value):

        all_key = [k_v[0] for k_v in key_value]
        all_values = [k_v[1] for k_v in key_value]

        return pd.DataFrame(all_values, index=all_key, columns=[section])

    def read_section(self, ini):

        # Reads all sections
        return [section for section in ini.sections()]

    def read_key_value(self, ini, section):

        all_key_value = []
        for key in ini.options(section):
            all_key_value.append([key, ini.get(section, key)])

        return all_key_value

    def design_dialog(self, l_data_frame):

        n_section = len(l_data_frame)

        max_n_key = 0
        for s_k_v in l_data_frame:
            n_k_v = len(s_k_v.index)
            if n_k_v > max_n_key:
                max_n_key = n_k_v

        self.max_n_key = max_n_key

        width = self.width_section_box * n_section + self.margin * 2
        height = self.height_section + self.height_key * max_n_key + self.margin * 2
        return width, height

    def set_static_text(self, text, x, y):
        """
        :return:
        """
        text = wx.StaticText(self, wx.ID_ANY, text, pos=(x, y))
        # text.SetBackgroundColour("white")
        # self.layout.Add(text)

    def set_text_ctrl(self, text, x, y):
        """

        :param text:
        :param x:
        :param y:
        :return:
        """
        textctrl = wx.TextCtrl(self, wx.ID_ANY, text, pos=(x, y))
        self.l_textctrl.append(textctrl)

    def run_button(self):
        """
        メニュー画面のスタートボタン
        :return:
        """

        x = + self.width_section_box * len(self.l_section) - self.margin - self.width_run_button
        y = self.margin + self.height_section + self.height_key * self.max_n_key - self.margin - self.height_run_button

        run_b = wx.Button(self.panel, label="Run", pos=(x, y))  # ポジション
        run_b.SetSize((self.width_run_button, self.height_run_button))  # ボタンサイズ

        self.Bind(wx.EVT_BUTTON, self.event_run_button, run_b)  # フレームにBind
        self.layout.Add(run_b)

    def event_run_button(self, event):
        """
        スタートボタンを押したときのイベント
        :return:
        """

        all_values = self.get_new_values()
        self.remake_config(all_values)
        self.make_shell_script('RBM_RUN.sh', os.path.join('Config', 'config.ini'))
        self.Close(True)
        sys.exit(0)

    def remake_config(self, all_values):

        l_data_frame = self.l_data_frame

        num_data_old = 0
        for i, df in enumerate(l_data_frame):
            num_data = len(df.index)
            df[df.columns[0]] = all_values[0 + num_data_old:num_data + num_data_old]
            num_data_old += num_data

            with open('Config/config.ini', mode='w', encoding='utf-8') as fh:

                for i, df in enumerate(l_data_frame):

                    section_name = '[' + df.columns[0] + ']'
                    fh.write(section_name + '\n')
                    for k, v in zip(df[df.columns[0]].index, df[df.columns[0]].values):
                        k_v = k + '=' + v
                        fh.write(k_v + '\n')

            with open('Config/ConfigOriginal/ConfigOriginal.ini', mode='w', encoding='utf-8') as fh:

                for i, df in enumerate(l_data_frame):

                    section_name = '[' + df.columns[0] + ']'
                    fh.write(section_name + '\n')
                    for k, v in zip(df[df.columns[0]].index, df[df.columns[0]].values):
                        k_v = k + '=' + v
                        fh.write(k_v + '\n')




    def make_shell_script(self, filename1, filename):

        with open(filename1, mode='w', encoding='utf-8') as fh:
            fh.write('#!/bin/sh\n')
            fh.write('python RBMProject.py %s' % filename)

    def get_new_values(self):

        return [tc.GetValue() for tc in self.l_textctrl]

    def exit_button(self):
        """
        メニュー画面のスタートボタン
        :return:
        """

        x = + self.width_section_box * len(
            self.l_section) - self.margin - self.width_run_button - self.width_exit_button - 10
        y = self.margin + self.height_section + self.height_key * self.max_n_key - self.margin - self.height_exit_button

        exit_b = wx.Button(self.panel, label="Exit", pos=(x, y))  # ポジション
        exit_b.SetSize((self.width_exit_button, self.height_exit_button))  # ボタンサイズ

        font = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)  # フォントサイズ
        exit_b.SetFont(font)

        self.Bind(wx.EVT_BUTTON, self.event_exit_button, exit_b)  # フレームにBind
        self.layout.Add(exit_b)

    def event_exit_button(self, event):
        """
        スタートボタンを押したときのイベント
        :return:
        """

        self.Close(True)  # Close the frame.
        sys.exit(-1)


class MyApp(wx.App):
    def OnInit(self):
        ConfigGenerater(None, "ConfigGenerater", ini)
        return True


def main():
    application = MyApp()
    application.MainLoop()


if __name__ == '__main__':
    ini = sys.argv[1]
    main()
