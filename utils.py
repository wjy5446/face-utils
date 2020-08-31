import os
import re
import json

import numpy as np
import imageio

from tqdm import tqdm

from mesh.render import *
from mesh.light import *
from mesh.transform import *


def read_bs_info(path_ref_obj='data/animation/do_avg.obj', path_folder_bs = 'data/animation/BS'):
    ref_obj = read_obj(path_ref_obj) 

    li_bs = [name for name in os.listdir(path_folder_bs) if name.endswith('obj')]
    li_bs = sorted(li_bs, key=lambda x: int(re.findall(r'\d+', x)[0]))

    bs = []
    for i in range(len(li_bs)):
        bs_obj = read_obj(os.path.join(path_folder_bs, li_bs[i]))
        diff = (bs_obj['v'] - ref_obj['v']).reshape(-1)
        bs.append(diff)

    dict_bs_info = {}
    
    dict_bs_info['avg'] = ref_obj['v'].reshape(1, -1)
    dict_bs_info['bs'] = np.stack(bs, 0)
    dict_bs_info['tri'] = ref_obj['f']
    
    return dict_bs_info

def read_param_sequence(path_folder_json = 'data/json/raw'):
    dict_param_sequence = {}
    
    li_json_file = [name for name in os.listdir(path_folder_json) if name.endswith('json')]
    li_json_file = sorted(li_json_file, key=lambda x: int(re.findall(r'\d+', x)[0]))
    
    for i in range(len(li_json_file)):
        path_param = os.path.join(path_folder_json, li_json_file[i])
        dict_param_sequence[i] = read_param_json(path_param)['exp'].reshape(1, -1)

        
    return dict_param_sequence

def make_animation(path_save_video='result/animation/result.mp4', path_save_folder_image = 'result/animation/image', scale = 25., fps=25.,\
                   path_ref_obj='data/animation/do_avg.obj', path_folder_bs = 'data/animation/BS', path_folder_json = 'data/animation/json'):
    light_info = {
        'position': np.array([[0, 0, -1]]),
        'intensity': np.array([[1, 1, 1]])
    }
    
    bs_info = read_bs_info(path_ref_obj, path_folder_bs)
    param_seq_info = read_param_sequence(path_folder_json)
    
    print('[INFO] 이미지 렌더링 시작!!')

    offset = np.array([ 9.30950209e-01,  1.06935143e-01,  2.65788615e-01,  2.73007751e-01,
        7.77550993e-01,  7.26338997e-02,  3.38005349e-01, -9.35633630e-01,
       -1.16190767e+00,  5.50828624e-01,  7.83390254e-01,  1.69388729e+00,
       -9.84009512e-01, -1.42385536e+00,  1.17187378e+00,  3.75027046e-01,
        1.18690567e+00,  5.36381468e-01,  2.50893084e-01,  2.62994753e-01,
       -5.00521779e-01, -3.42887640e-02, -8.05118531e-02,  1.26314674e+00,
        2.51280755e-01, -8.20122510e-02,  2.99760610e-01,  2.64429465e-01,
       -6.07872367e-01,  6.65567964e-02,  2.08824255e-01,  2.70742849e-01,
       -3.39620858e-01,  1.38687402e-01,  3.12591206e-01, -2.86739245e-01,
        1.22895569e-01,  1.52587425e-01, -3.20241079e-01, -4.59863991e-01,
        1.26180649e-01, -1.44582745e-01,  8.56065229e-02,  3.97813506e-01,
       -3.71251181e-02, -3.57503481e-02,  1.02849654e-02,  1.48948032e-01,
       -2.60900147e-02,  1.64385222e-01, -1.14372328e-01, -1.40718352e-02,
        7.43482616e-02,  7.40041547e-02,  9.40444414e-03, -4.25160136e-02,
       -2.70825564e-02, -4.58473936e-02,  1.22195017e-03,  4.60817013e-03,
       -1.08957738e-02,  3.60023109e-02,  9.12482403e-02,  1.12532303e-02,
       -6.64410874e-03, -8.30940041e-03,  1.40307257e-02, -5.20424526e-02,
        8.80412990e-03,  1.03377355e-02,  6.67846086e-03, -1.56032345e-02,
       -1.99090270e-02, -4.98940237e-04, -7.38497032e-03,  9.50660137e-03,
        3.25013762e-02, -2.53932234e-02, -1.09215146e-02])

    for i in tqdm(range(len(param_seq_info))):
        geo = bs_info['avg'] + (param_seq_info[i][:,:79] + offset) @ bs_info['bs'] / 4
        geo = geo.reshape(-1, 3) * scale

        txt = np.full_like(geo, 250. / 255.)

        light = add_light(geo, (bs_info['tri'] - 1), txt, light_info['position'], light_info['intensity'])
        pixel_geo = to_image(geo, 450, 450)
        img_render = render_colors(pixel_geo, (bs_info['tri'] - 1), light, 450, 450)
        img_render = np.minimum((np.maximum(img_render, 0)), 1) 
        img_render = (img_render * 255).astype(np.uint8)
        imageio.imwrite(path_save_folder_image + '/' + str(i) + '.jpg', img_render)
        
    print('[INFO] 이미지 렌더링 완료!!')
    
    make_video(path_save_folder_image, path_save_video, fps=fps)

    print('[INFO] 비디오 생성 완료!!')

def normalize_vertices(arr):
    mag = np.sqrt(arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2)
    arr[:,0] /= mag
    arr[:,1] /= mag
    arr[:,2] /= mag

    return arr

def get_vertices_norm(vertices, faces):
    # vertices (n_vert, 3)    
    # faces (n_face, 3)

    norm = np.zeros(vertices.shape, dtype=vertices.dtype) 
    
    tris = vertices[faces - 1] 

    n = np.cross(tris[::,1] - tris[::,0], tris[::,2] - tris[::,0]) # faces
    n = normalize_vertices(n)

    norm[faces[:,0] - 1] += n
    norm[faces[:,1] - 1] += n
    norm[faces[:,2] - 1] += n

    return normalize_vertices(norm)

def make_video(path_img_folder, video_name, fps=25):
    imgs = []

    li_img = os.listdir(path_img_folder)
    li_img = sorted(li_img, key=lambda x: int(re.findall(r'\d+', x)[0]))

    for img in tqdm(li_img):
        img = imageio.imread(os.path.join(path_img_folder, img))
        imgs.append(img)

    imageio.mimwrite(video_name, imgs, fps=fps, macro_block_size=None)

def read_param_json(path_json):
    dict_info = json.load(open(path_json, 'r')) 
    dict_info = {k: np.array(v).astype(np.float) for k, v in dict_info.items()}
    #dict_info['shp'] = dict_info['shp'].reshape(-1, 1)
    dict_info['exp'] = dict_info['exp'].reshape(-1, 1)

    return dict_info

def read_obj(obj_name):
    dict_info = {}    

    vertices = []
    colors = []
    vertices_norm = []
    faces = []

    with open(obj_name, 'r') as f:
        lines = f.readlines()

        for line in lines:
            elements = [filename for filename in line.split('\n')[0].split(' ') if filename != '']


            if not elements:
                continue

            # ignore # 
            if elements[0][0] == '#':
                continue
            elif elements[0] == 'v':
                vertices.append([float(elements[1]), float(elements[2]), float(elements[3])])

                if len(elements) == 7:
                    colors.append([float(elements[4]), float(elements[5]), float(elements[6])])

            elif elements[0] == 'vn':
                vertices_norm.append([float(elements[1]), float(elements[2]), float(elements[3])])
            elif elements[0] == 'f':
                faces.append([int(elements[1].split('/')[0]), 
                              int(elements[2].split('/')[0]),
                              int(elements[3].split('/')[0])])
        
    if vertices:
        dict_info['v'] = np.array(vertices)
    if colors:
        dcit_info['vc'] = np.array(colors)
    if vertices_norm:
        dict_info['vn'] = np.array(vertices_norm)
    if faces:
        dict_info['f'] = np.array(faces)


    return dict_info

def write_obj(obj_name, vertices, triangles):
    triangles = triangles.copy() # meshlab start with 1

    vertices_norm = get_vertices_norm(vertices, triangles)

    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'

    # write obj
    with open(obj_name, 'w') as f:
        # write vertices & colors
        for i in range(vertices.shape[0]):
            s = 'v {:.4f} {:.4f} {:.4f}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2])
            f.write(s)

        # write vertices normal
        for i in range(vertices.shape[0]):
            s = 'vn {:.4f} {:.4f} {:.4f}\n'.format(vertices_norm[i, 0], vertices_norm[i, 1], vertices_norm[i, 2])
            f.write(s)

        # write f: ver ind/ uv ind
        for i in range(triangles.shape[0]):
            s = 'f {tri0}//{tri0} {tri1}//{tri1} {tri2}//{tri2}\n'.format(tri0=triangles[i, 0], tri1=triangles[i, 1], tri2=triangles[i, 2])
            f.write(s)

def write_3dmm_json(path_json, param_shp, param_exp, param_pos):
    dict_3dmm = {}
    dict_3dmm['shp'] = param_shp.tolist()
    dict_3dmm['exp'] = param_exp.tolist()
    dict_3dmm['pos'] = param_pos.tolist()
    
    with open(path_json, 'w') as f:
        json.dump(dict_3dmm, f)
   
    return
