from collections import defaultdict

from utils import *

def smooth(x, window_len=11, window='hanning'):
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
        
    y=np.convolve(w/w.sum(),s,mode='valid')
    
    return y

def smooth_param(window_len=5, window='hanning', path_folder_json_raw='data/smoother', path_save_folder_json='result/smoother'):
    dict_seq_by_bs = defaultdict(list)
    dict_param_seq = read_param_sequence(path_folder_json_raw)
    
    # bs 기준으로 dict 변경
    for i in range(len(dict_param_seq)):
        param = dict_param_seq[i]

        for j in range(param.shape[1]):
            dict_seq_by_bs[j].append(param[0,j])
            
    # smooth
    dict_seq_by_bs_smooth = {}

    for i in range(len(dict_seq_by_bs)):
        smooth_value = smooth(dict_seq_by_bs[i], window_len=window_len, window=window)
        dict_seq_by_bs_smooth[i] = smooth_value
        
    # json 생성
    for i in range(len(dict_param_seq)):
        li_param = []
        for j in range(dict_param_seq[0].shape[1]):
            li_param.append(dict_seq_by_bs_smooth[j][i])

        path_json_smooth = os.path.join(path_save_folder_json, str(i) + '.json')
        write_3dmm_json(path_json_smooth, np.array([]), np.array(li_param), np.array([]))
        
    print('[INFO] smooth param 생성 완료!!')
