import torch
import torch.nn as nn
import torchvision.models as models

"""
VisualExtractor Module

This module defines a PyTorch neural network model for extracting visual features 
from input images using a pre-trained model from torchvision.

Classes:
    VisualExtractor: A PyTorch nn.Module that extracts patch-level and average 
                     features from input images.

Methods:
    __init__(self, args):
        Initializes the VisualExtractor module.
        
        Args:
            args: An object containing the following attributes:
                - visual_extractor (str): The name of the pre-trained model to use 
                  (e.g., 'resnet50').
                - visual_extractor_pretrained (bool): Whether to use a pre-trained 
                  version of the model.

    forward(self, images):
        Performs a forward pass through the VisualExtractor module.
        
        Args:
            images (torch.Tensor): A batch of input images with shape 
                                   (batch_size, channels, height, width).
        
        Returns:
            tuple:
                - patch_feats (torch.Tensor): Patch-level features with shape 
                  (batch_size, num_patches, feature_size).
                - avg_feats (torch.Tensor): Average features with shape 
                  (batch_size, feature_size).
"""

class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats
