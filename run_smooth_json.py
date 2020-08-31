from postprocess import *

if __name__ == '__main__':
    window_len = 5
    window = 'hanning'

    path_folder_json_raw = 'data/smoother'
    path_save_folder_json = 'result/smoother'
    smooth_param(window_len, window, path_folder_json_raw, path_save_folder_json)
