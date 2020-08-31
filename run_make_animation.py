from utils import *

if __name__ == '__main__':
    path_save_video = 'result/animation/result.mp4'
    path_save_folder_image = 'result/animation/image'
    path_ref_obj = 'data/animation/do_avg.obj'
    path_folder_bs = 'data/animation/BS'
    path_folder_json = 'data/animation/json/'
    make_animation(path_save_video, path_save_folder_image, 20, 25, path_ref_obj, path_folder_bs, path_folder_json)
