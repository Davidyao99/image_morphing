from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay, delaunay_plot_2d
import cv2
import torch
import torchvision
from torchvision import transforms
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import matplotlib.image as mpimg

from facenet_pytorch import MTCNN

import sys
sys.path.append("spiga")
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework

from train.detector import ResNetDetector

class Engine():

    def __init__(self, image_dir, video_output_path='assets/output/output_video', device='cuda:0'):
        self.image_dir = image_dir
        self.video_output_path = video_output_path

        self.device = device
        self.face_detector = MTCNN(self.device)
        self.kpt_detector = SPIGAFramework(ModelConfig('wflw'))
        self.custom_detector = ResNetDetector('saved_models/resnet18_pretrain_model2.pt', device)
        self.size = 512

    def get_kpts_custom(self, imgs):
        
        imgs = np.array([np.asarray(img) for img in imgs])
        imgs = imgs / 255.0
        kpts = self.custom_detector.get_kpts(imgs)

        kpts_corners = np.array([[0,0],[self.size,0],[0,self.size],[self.size,self.size]])
        kpts_corners = np.repeat(kpts_corners[None,:,:], repeats=kpts.shape[0], axis=0)
        kpts = np.concatenate((kpts, kpts_corners), axis=1) # add 4 corners to all images

        return kpts # numpy array of size  b x 72 x 2

    def get_kpts(self, imgs):
        
        bboxs, _ = self.face_detector.detect(imgs)

        all_landmarks = []

        for i,bbox in enumerate(bboxs):
            features = self.kpt_detector.inference(np.array(imgs[i]), [bbox[0]])
            landmarks = np.array(features['landmarks'][0])
            landmarks = np.vstack((landmarks, np.array([[0,0],[self.size,0],[0,self.size],[self.size,self.size]])))
            all_landmarks.append(landmarks)

        return all_landmarks # list of n numpy arrays of 102 x 2

    def generate_triangle_mask(self, tri, P):
    
        A,B,C = tri[0,:], tri[1,:], tri[2,:]
        
        denominator = ((B[1] - C[1]) * (A[0] - C[0]) +
                    (C[0] - B[0]) * (A[1] - C[1]))
        a = ((B[1] - C[1]) * (P[:,0] - C[0]) +
            (C[0] - B[0]) * (P[:,1] - C[1])) / denominator
        b = ((C[1] - A[1]) * (P[:,0] - C[0]) +
            (A[0] - C[0]) * (P[:,1] - C[1])) / denominator
        c = 1 - a - b

        return np.logical_and(np.logical_and(a>=0,b>=0), c>=0)

    def solve_tri_affine(self, triangles, kp1, kp2):
    
        """
        Solver for affine transformation from kp1 -> kp2 for each triangle in triangles
        """
        
        affine_Ts = np.zeros((triangles.shape[0], 2, 3))
        
        for i in range(triangles.shape[0]):
            
            tri_idx = triangles[i]
            
            kp1_tri = kp1[tri_idx]
            kp2_tri = kp2[tri_idx]
            
            A = []
            b = []
            
            for j in range(3):
            
                x1,y1,x2,y2 = kp1_tri[j,0], kp1_tri[j, 1], kp2_tri[j,0], kp2_tri[j,1]
                A.append([x1,y1,1,0,0,0])
                A.append([0,0,0,x1,y1,1])
                b.append(x2)
                b.append(y2)
                
            A=np.array(A)
            b=np.array(b)
            
            affine_T = np.linalg.solve(A, b)
            
            affine_Ts[i] = affine_T.reshape((2,3))
        
        return affine_Ts

    def morph(self, im1, im2, kp1, kp2, step_size=0.02):

        """
        Morphs im1 into im2 using keypoints kp1 and kp2
        """

        avg_points = 0.5 * kp1 + 0.5 * kp2
        tri = Delaunay(avg_points)
        triangles = tri.simplices # in the form of indexes of keypoints
        affine_Ts = self.solve_tri_affine(triangles, kp1, kp2)

        res = []

        for t in np.arange(0.0, 1.0, step_size):
            
            frame_res = np.zeros(im1.shape)
            
            kp_inter = (1-t) * kp1 + t * kp2
            affine_Ts_1 = self.solve_tri_affine(triangles, kp1, kp_inter)
            affine_Ts_2 = self.solve_tri_affine(triangles, kp2, kp_inter)
            
            res_im1 = np.zeros((im1.shape), dtype=np.uint8)
            res_im2 = np.zeros((im2.shape), dtype=np.uint8)
            
            for i in range(affine_Ts_1.shape[0]):
                affine_T_1 = affine_Ts_1[i]
                affine_T_2 = affine_Ts_2[i]
                tri_1 = kp1[triangles[i]]
                tri_2 = kp2[triangles[i]]
                tri_inter = kp_inter[triangles[i]]
                
                x = np.arange(self.size)
                y = np.arange(self.size)
                xv, yv = np.meshgrid(x, y)
                P = np.stack((xv,yv), axis=-1).reshape(-1,2) # gets a (hxw,2) numpy array of all coordinates of image
                
                mask = self.generate_triangle_mask(tri_inter, P).reshape((self.size, self.size)) # gets binary triangle mask for image1
                res_im1[mask] = cv2.warpAffine(im1, affine_T_1, (self.size, self.size), cv2.WARP_INVERSE_MAP)[mask]
                res_im2[mask] = cv2.warpAffine(im2, affine_T_2, (self.size, self.size), cv2.WARP_INVERSE_MAP)[mask]
            res_im = (res_im1 * (1-t) + res_im2 * t).astype(np.uint8)
            res.append(res_im)

            print(f"Progress: {t*100:.2f}%", end="\r", flush=True)
            
        return res

    def create_video(self, res):

        output_video_path = f'{self.video_output_path}.avi'

        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can choose another codec based on your needs
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 20.0,  (self.size, self.size))

        # Iterate through each frame and write it to the video
        for frame in res:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
        video_writer.release()

    def run(self, custom=True):

        #get all images in directory
        images_path = os.listdir(self.image_dir)

        images_all = []

        for image_path in images_path:
            im2 = Image.open(os.path.join(self.image_dir, image_path)).resize((self.size, self.size))
            images_all.append(im2)

        if custom:
            kpts = self.get_kpts_custom(images_all)
        else:
            kpts = self.get_kpts(images_all)

        all_res = []

        for i in range(len(images_all)-1):
            print(f"Working on morphing of image{i} and image{i+1}")
            res = self.morph(np.array(images_all[i]), np.array(images_all[i+1]), kpts[i], kpts[i+1])
            all_res += res

        self.create_video(all_res)


if __name__ == '__main__':
    engine = Engine('assets/input_out', 'assets/output/output_video_out')
    engine.run(False)
