import face_alignment
import os

import numpy
from skimage import io
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import cv2

id = 'Obama'
id_dir = os.path.join('dataset', id)
Path(id_dir).mkdir(parents=True, exist_ok=True)
ori_imgs_dir = os.path.join('dataset', id, 'ori_imgs')
Path(ori_imgs_dir).mkdir(parents=True, exist_ok=True)
parsing_dir = os.path.join(id_dir, 'parsing')
Path(parsing_dir).mkdir(parents=True, exist_ok=True)
head_imgs_dir = os.path.join('dataset', id, 'head_imgs')
Path(head_imgs_dir).mkdir(parents=True, exist_ok=True)
com_imgs_dir = os.path.join('dataset', id, 'com_imgs')
Path(com_imgs_dir).mkdir(parents=True, exist_ok=True)


def get_valid_img_ids():
    if os.path.exists('valid_img_ids.npy'):
        print("load from npy")
        valid_img_ids = np.load('valid_img_ids.npy')
        tmp_img = cv2.imread(os.path.join(ori_imgs_dir, str(valid_img_ids[0]) + '.jpg'))
        h, w = tmp_img.shape[0], tmp_img.shape[1]
        return valid_img_ids, len(valid_img_ids), h, w

    valid_img_ids = []
    for i in range(100000):
        if os.path.isfile(os.path.join(ori_imgs_dir, str(i) + '.lms')):
            valid_img_ids.append(i)
    valid_img_num = len(valid_img_ids)
    tmp_img = cv2.imread(os.path.join(ori_imgs_dir, str(valid_img_ids[0]) + '.jpg'))
    h, w = tmp_img.shape[0], tmp_img.shape[1]

    return valid_img_ids, valid_img_num, h, w


vii, vin, h, w = get_valid_img_ids()
print(vii, vin, h, w)

if __name__ == '__main__':
    # # Step 0: extract wav & deepspeech feature, better run in terminal to parallel with
    # below commands since this may take a few minutes
    def extract_deep_speech():
        print('--- Step0: extract deepspeech feature ---')
        wav_file = os.path.join('//mnt/cpfs/users/gpuwork/zheng.zhu/talking-head-code/AD-NeRF/test_audio', 'aud.wav')
        extract_wav_cmd = 'ffmpeg -i ' + '/mnt/cpfs/users/gpuwork/zheng.zhu/talking-head-code/AD-NeRF/test_audio/极限挑战这就是命.mp4' + ' -f wav -ar 16000 ' + wav_file
        os.system(extract_wav_cmd)
        extract_ds_cmd = 'python deepspeech_features/extract_ds_features.py --input=' + '//mnt/cpfs/users/gpuwork/zheng.zhu/talking-head-code/AD-NeRF/test_audio'
        os.system(extract_ds_cmd)
        exit()


    valid_img_ids, valid_img_num, h, w = get_valid_img_ids()
    print(f'{valid_img_ids} || {valid_img_num}')
