"""
Heavily referenced from:
https://github.com/nalbert9/Facial-Keypoint-Detection
https://www.kaggle.com/code/james146/facial-keypoints-detection-pytorch
"""


from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms, utils
import pandas as pd
import os
import matplotlib.image as mpimg
import cv2
import numpy as np

class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])
        
        image = mpimg.imread(image_name)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        key_pts = self.key_pts_frame.iloc[idx, 1:].to_numpy()
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        key_pts_copy = np.copy(key_pts)

        key_pts_copy = (key_pts_copy - 100)/50.0


        return {'image': image, 'keypoints': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class RandomRotation(object):
    
    def __init__(self, angle: int, p=0.5):
        super().__init__()
        if not (isinstance(p, float) and (0.0 <= p <= 1.0)):
            raise ValueError("probability should be float between 0 and 1")
        self.p = p
        self.angle = angle

    def __call__(self, sample):
        if torch.rand(1) < self.p:
            angle = np.random.randint(-self.angle, self.angle)
            image, key_pts = sample['image'], sample['keypoints']

            height, width = image.shape[:2]

            # Calculate the rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)

            key_pts_homo = np.hstack((key_pts, np.ones((key_pts.shape[0], 1))))
            key_pts_homo_rot = np.dot(rotation_matrix, key_pts_homo.T).T
            key_pts_rot = key_pts_homo_rot[:, :2]

            return {'image': rotated_image, 'keypoints': key_pts_rot}

        return sample
    
class RandomTranslation(object):
    def __init__(self, translate: (float, float), p=0.5):
        """

        :type translate: (float, float) x, y translate in percent - from 0 to 1
        """
        super().__init__()
        if not (isinstance(p, float) and (0.0 <= p <= 1.0)):
            raise ValueError("probability should be float between 0 and 1")
        if not (len(translate) == 2 and (0.0 <= translate[0] <= 1.0) and (0.0 <= translate[1] <= 1.0)):
            raise ValueError("there should be 2 numbers in translate, both between 0 and 1")
        self.p = p
        self.translate = translate

    def __call__(self, sample):
        if torch.rand(1) < self.p:
            image, key_pts = sample['image'], sample['keypoints']

            height, width = image.shape[0], image.shape[1]

            x_translate_rate = np.random.uniform(low=-self.translate[0], high=self.translate[0])
            y_translate_rate = np.random.uniform(low=-self.translate[1], high=self.translate[1])
            x_translate_pixel = width * x_translate_rate
            y_translate_pixel = height * y_translate_rate

            # Image translation
            M = np.float32([[1, 0, x_translate_pixel], [0, 1, y_translate_pixel]])
            shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

            # Keypoints translation
            key_pts = key_pts + np.array([x_translate_pixel, y_translate_pixel])

            return {'image': shifted, 'keypoints': key_pts}

        return sample
    
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}