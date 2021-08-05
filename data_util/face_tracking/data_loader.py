import os
import pickle

import numpy as np
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'


# 很慢的一个方法
def load_dir(path, start, end, cuda_num=0):
    if os.path.exists('lmss.npy') and os.path.exists('imgs_paths.obj'):
        print(">>>> load from stream <<<<")
        lmss = np.load('lmss.npy')
        with open('imgs_paths.obj', 'rb') as f:
            imgs_paths = pickle.load(f)
        return torch.as_tensor(lmss).cuda(cuda_num), imgs_paths

    lmss = []
    imgs_paths = []
    for i in range(start, end):
        if os.path.isfile(os.path.join(path, str(i) + '.lms')):
            lms = np.loadtxt(os.path.join(
                path, str(i) + '.lms'), dtype=np.float32)
            lmss.append(lms)
            imgs_paths.append(os.path.join(path, str(i) + '.jpg'))
        if i % 100 == 0:
            print(f"face_tracker: load data: {i}.jpg")
    lmss = np.stack(lmss)

    np.save('lmss.npy', lmss)
    f = open('imgs_paths.obj', 'wb')
    pickle.dump(imgs_paths, f)

    return lmss, imgs_paths


if __name__ == '__main__':
    lmss, imgs_paths = load_dir("/mnt/cpfs/users/gpuwork/zheng.zhu/talking-head-code/AD-NeRF/dataset/Obama/ori_imgs", 0,
                                0)
    print(lmss)
    print(imgs_paths)
