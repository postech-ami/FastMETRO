# ----------------------------------------------------------------------------------------------
# FastMETRO Official Code
# Copyright (c) POSTECH Algorithmic Machine Intelligence Lab. (P-AMI Lab.) All Rights Reserved 
# Licensed under the MIT license.
# ----------------------------------------------------------------------------------------------
# Modified from GraphCMR (https://github.com/nkolot/GraphCMR)
# Copyright (c) University of Pennsylvania. All Rights Reserved [see https://github.com/nkolot/GraphCMR/blob/master/LICENSE for details]
# ----------------------------------------------------------------------------------------------

"""
(Optional) SMPL parameter regressor.
"""
from __future__ import absolute_import, division, print_function
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

class SMPL_Parameter_Regressor(nn.Module):
    """SMPL Parameter Regressor"""
    def __init__(self):
        super().__init__()
        self.regressor = nn.Sequential(FCBlock(1723*3, 1024),
                                        FCResBlock(1024, 1024),
                                        FCResBlock(1024, 1024),
                                        nn.Linear(1024, 24*3*3+10))
    
    def forward(self, pred_vertices):
        """Forward pass.
        Input:
            vertices (from non-parametric): size = (batch_size, num_vertices, 3)
        Returns:
            SMPL pose parameters as rotation matrices: size = (batch_size, 24, 3, 3)
            SMPL shape parameters: size = (batch_size, 10)
        """
        device = pred_vertices.device
        batch_size = pred_vertices.size(0)

        rotmat_beta = self.regressor(pred_vertices.reshape(batch_size, 1723*3))
        rotmat = rotmat_beta[:, :24*3*3].view(-1, 24, 3, 3).contiguous()
        pred_betas = rotmat_beta[:, 24*3*3:].contiguous()
        
        rotmat = rotmat.view(-1, 3, 3).contiguous()
        rotmat =  rotmat.cpu()
        U, S, V = batch_svd(rotmat)
        rotmat = torch.matmul(U, V.transpose(1,2))
        det = torch.zeros(rotmat.shape[0], 1, 1).to(rotmat.device)
        with torch.no_grad():
            for i in range(rotmat.shape[0]):
                det[i] = torch.det(rotmat[i])
        pred_rotmat = rotmat * det
        pred_rotmat = pred_rotmat.view(batch_size, 24, 3, 3)
        pred_rotmat = pred_rotmat.to(device)

        return pred_rotmat, pred_betas


class FCBlock(nn.Module):
    """Wrapper around nn.Linear that includes batch normalization and activation functions."""
    def __init__(self, in_size, out_size, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(FCBlock, self).__init__()
        module_list = [nn.Linear(in_size, out_size)]
        if batchnorm:
            module_list.append(nn.BatchNorm1d(out_size))
        if activation is not None:
            module_list.append(activation)
        if dropout:
            module_list.append(dropout)
        self.fc_block = nn.Sequential(*module_list)

    def forward(self, x):
        return self.fc_block(x)


class FCResBlock(nn.Module):
    """Residual block using fully-connected layers."""
    def __init__(self, in_size, out_size, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(FCResBlock, self).__init__()
        self.fc_block = nn.Sequential(nn.Linear(in_size, out_size),
                                      nn.BatchNorm1d(out_size),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(out_size, out_size),
                                      nn.BatchNorm1d(out_size))

    def forward(self, x):
        return F.relu(x + self.fc_block(x))

def batch_svd(A):
    """Wrapper around torch.svd that works when the input is a batch of matrices."""
    U_list = []
    S_list = []
    V_list = []
    for i in range(A.shape[0]):
        U, S, V = torch.svd(A[i])
        U_list.append(U)
        S_list.append(S)
        V_list.append(V)
    U = torch.stack(U_list, dim=0)
    S = torch.stack(S_list, dim=0)
    V = torch.stack(V_list, dim=0)
    return U, S, V

def build_smpl_parameter_regressor():
    return SMPL_Parameter_Regressor()