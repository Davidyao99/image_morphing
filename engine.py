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

class Engine():

    def __init__(self, image_dir):
        self.image_dir = image_dir

        self.face_detector = MTCNN(device='cuda:0')
        self.kpt_detector = SPIGAFramework(ModelConfig('wflw'))

    def get_kpts(self, imgs):
        
        bboxs, _ = self.face_detector.detect(imgs)

        all_landmarks = []

        w,h = imgs[0].size

        for i,bbox in enumerate(bboxs):
            features = self.kpt_detector.inference(np.array(imgs[i]), [bbox[0]])
            landmarks = np.array(features['landmarks'][0])
            landmarks = np.vstack((landmarks, np.array([[0,0],[w,0],[0,h],[w,h]])))
            all_landmarks.append(landmarks)

        return all_landmarks # list of n numpy arrays of 98 x 2

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
            
            h,w,_ = im1.shape
            
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
                
                x = np.arange(w)
                y = np.arange(h)
                xv, yv = np.meshgrid(x, y)
                P = np.stack((xv,yv), axis=-1).reshape(-1,2) # gets a (hxw,2) numpy array of all coordinates of image
                
                mask = self.generate_triangle_mask(tri_inter, P).reshape((h,w)) # gets binary triangle mask for image1
                res_im1[mask] = cv2.warpAffine(im1, affine_T_1, (w, h), cv2.WARP_INVERSE_MAP)[mask]
                res_im2[mask] = cv2.warpAffine(im2, affine_T_2, (w, h), cv2.WARP_INVERSE_MAP)[mask]
            res_im = (res_im1 * (1-t) + res_im2 * t).astype(np.uint8)
            res.append(res_im)

            print(f"Progress: {t*100:.2f}%", end="\r", flush=True)
            
        return res

    def create_video(self, res, w, h, video_name='output_video_test'):

        output_video_path = f'{video_name}.avi'

        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can choose another codec based on your needs
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 20.0,  (w, h))

        # Iterate through each frame and write it to the video
        for frame in res:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
        video_writer.release()

    def run(self):

        #get all images in directory
        images_path = os.listdir(self.image_dir)

        im1 = Image.open(os.path.join(self.image_dir, images_path[0]))
        w, h = (448, 488)

        images_all = [im1.resize((w,h))]

        for i in range(len(images_path)-1):
            im2 = Image.open(os.path.join(self.image_dir, images_path[i+1])).resize((w, h))
            images_all.append(im2)

        kpts = self.get_kpts(images_all) # batch process at once

        all_res = []

        for i in range(len(images_all)-1):
            print(f"Working on morphing of image{i} and image{i+1}")
            res = self.morph(np.array(images_all[i]), np.array(images_all[i+1]), kpts[i], kpts[i+1])
            all_res += res

        self.create_video(all_res, w, h)


if __name__ == '__main__':
    engine = Engine('assets')
    engine.run()
