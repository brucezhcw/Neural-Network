from datatools import *
import win_unicode_console
win_unicode_console.enable()

if __name__ == '__main__':
    input_folder = 'D:\\CodeHub\\HandNet-master'
    output_folder = '.\\'
    data_name = 'TrainData'
    build_dataset(input_folder, output_folder, data_name)
