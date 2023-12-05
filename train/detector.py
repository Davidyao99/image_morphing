import torch
import torchvision
from torchvision import transforms, utils

class ResNetDetector():

    def __init__(self, model_path, device='cuda'):
        self.device = device

        model = torchvision.models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 68*2)
        model = model.type(torch.FloatTensor)
        self.net = model.to(device)
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()

    def get_kpts(self, imgs):

        """
        imgs: ndarray of images b x h x w x c
        """

        imgs = torch.from_numpy(imgs.transpose((0, 3, 1, 2)))
        imgs = imgs.type(torch.FloatTensor).to(self.device)

        with torch.no_grad():
            output_pts = self.net(imgs)

        output_pts = output_pts.view(output_pts.size()[0], 68, -1)*76.80763153436327+285.1921855457837 # add back normalization

        return output_pts.cpu().numpy()