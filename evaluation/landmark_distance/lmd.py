import platform
import time
import argparse
import cv2
import os

import face_alignment
import numpy as np
import openface
import torch

np.set_printoptions(precision=2)


class landmark_detector():
    def __init__(self, img_dim, gpu, verbose):
        start = time.time()

        self.model_dir = "/Users/gankaiyuan/python/openface/models" if platform.system() == "Darwin" \
            else "/home/pusuan.wk/gky/openface/models"
        self.dlib_model_dir = os.path.join(self.model_dir, 'dlib')
        self.openface_model_dir = os.path.join(self.model_dir, 'openface')

        self.img_dim = img_dim if img_dim is not None else 224

        self.aligner = openface.AlignDlib(os.path.join(self.dlib_model_dir, "shape_predictor_68_face_landmarks.dat"))

        self.face_alignment = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

        if args.verbose:
            print("Loading the dlib and OpenFace models took {} seconds.".format(time.time() - start))

    def getRep(self, bgrImg):
        if bgrImg is None:
            raise Exception("Unable to load image!")

        bgrImg = cv2.resize(bgrImg, (450, 450), interpolation=cv2.INTER_AREA)
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

        rgbImg = torch.tensor(rgbImg).to(self.device)

        if args.verbose:
            print("  + Original size: {}".format(rgbImg.shape))

        start = time.time()
        # 计算关键点坐标
        landmarks = self.face_alignment.get_landmarks(rgbImg)
        if args.verbose:
            for pt in landmarks[0]:
                cv2.circle(bgrImg, tuple(pt), 3, [0, 0, 255], thickness=-1)
            cv2.imwrite("./evaluation/land.jpg",bgrImg)
        # 把关键点坐标合成矩阵
        rep = np.stack(landmarks, 0)
        if args.verbose:
            print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
            print("Representation:")
            print(rep.shape)
            print("-----\n")
        return rep

    def landmark_distance(self, src, dst):
        d = self.getRep(src) - self.getRep(dst)
        d_sqrt = np.mean(d ** 2)
        print("Squared l2 distance between representations: {:0.3f}".format(np.sum(d_sqrt)))
        return np.sum(d_sqrt)


# 测试使用
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
    parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                        default="/Users/gankaiyuan/python/openface/models/dlib/shape_predictor_68_face_landmarks.dat")

    parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                        default="/Users/gankaiyuan/python/openface/models/openface/nn4.small2.v1.t7")
    parser.add_argument('--imgDim', type=int, help="Default image dimension.")
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--gpu', default=0, help="choose gpu to use")

    args = parser.parse_args()

    img0 = cv2.imread(args.imgs[0])
    img1 = cv2.imread(args.imgs[1])

    landmark_detector = landmark_detector(img_dim=img0.shape[0], gpu=0, verbose=True)
    landmark_detector.landmark_distance(img0, img1)
