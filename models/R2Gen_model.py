import numpy as np
import torch
import torch.nn as nn

from .CMN import CMN
from .R2Gen import R2Gen
from utils.visual_extractor import VisualExtractor

"""
This module is the interface for the R2Gen model, which is used for generating radiology reports from medical images.

Classes:
    R2GenModel(nn.Module):
        A model for generating radiology reports from medical images. It uses a visual extractor
        and an encoder-decoder architecture. The forward method is dynamically set based on the dataset.

        Methods:
            __init__(args, tokenizer):
                Initializes the R2GenModel with the given arguments and tokenizer.
            __str__():
                Returns a string representation of the model, including the number of trainable parameters.
            forward_iu_xray(images, targets=None, mode='train'):
                Forward pass for the IU X-Ray dataset. Processes paired images and generates outputs.
            forward_mimic_cxr(images, targets=None, mode='train'):
                Forward pass for the MIMIC-CXR dataset. Processes single images and generates outputs.
"""

class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = R2Gen(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample')
            return output, output_probs
        else:
            raise ValueError

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample')
            return output, output_probs
        else:
            raise ValueError