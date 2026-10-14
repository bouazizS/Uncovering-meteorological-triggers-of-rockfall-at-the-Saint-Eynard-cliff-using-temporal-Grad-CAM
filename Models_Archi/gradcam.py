
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random 
import cv2 
#from inception_time import InceptionTimeModel
import matplotlib.colors as colors
import matplotlib.cm as cmx


class MultiBranchGradCAM:
    def __init__(self, model, target_branch):
        """ param target_branch: 'model.branch_1', 'model.branch_2'"""

        self.model = model
        self.target_branch = target_branch
        self.gradients = None
        self.activations = None
        # Automatically find last Inception module in the current branch 
        self.target_layer = self._find_last_inception_module(target_branch)

        # Register a backward hook to capture gradients
        self.target_layer.register_full_backward_hook(self.save_gradient)
        # Register a forward hook to capture the activations
        self.target_layer.register_forward_hook(self.save_activation)


    def _find_last_inception_module(self, branch):
        """
        Finds the last Inception module inside a branch  (as last conv layer)
        """
        inception_modules = []
        for name, module in branch.named_modules():
            if "inception" in module.__class__.__name__.lower():
                inception_modules.append(module)

        if len(inception_modules) == 0:
            raise ValueError("No Inception module found in target branch.")
        
        return inception_modules[-1]

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()  # Sauver le gradient
        # print("Gradients shape:", self.gradients.shape)    #(B, c,T)
    def save_activation(self, module, input, output):
        self.activations = output.detach()  # Sauver l'activation
        # print("Activations shape:", self.activations.shape) #(B, c,T)

    def __call__(self, *inputs, target_class=None):
        # Forward pass
        # output ,w1,w2 = self.model(x1, x2)      # !!!!!!! w1, w2 for channel-wise archi
        output  = self.model(*inputs)  

        # If target_class is not specified, use the predicted class
        if target_class is None:
            target_class = output.argmax(dim=1)

        # Backward pass to compute gradients with respect to the target class
        self.model.zero_grad()

        # For each item in the batch, compute the loss with respect to its predicted target_class
        class_loss = output[torch.arange(output.size(0)), target_class].sum() 
        class_loss.backward()  #retain_graph=True

        # Compute the Grad-CAM heatmap
        weights = self.gradients.mean(dim=2, keepdim=True)
        gradcam_map = F.relu((weights * self.activations).sum(dim=1))   #(B,T)

        # print("Grad-CAM map shape:", gradcam_map.shape)  [1,14]
        # print("Grad-CAM map.unsqueeze(0) shape:", gradcam_map.unsqueeze(0).shape)  [1,1,14]


        # Upsample heatmap to match input tensor size
        gradcam_map = F.interpolate(gradcam_map.unsqueeze(0),
                                    size=inputs[0].shape[2:],  #here i have x1 and x2 have same length
                                    mode='linear', align_corners=False)
        heatmap = gradcam_map.squeeze().cpu().detach().numpy()

        return heatmap
