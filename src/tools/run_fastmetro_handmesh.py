# ----------------------------------------------------------------------------------------------
# FastMETRO Official Code
# Copyright (c) POSTECH Algorithmic Machine Intelligence Lab. (P-AMI Lab.) All Rights Reserved 
# Licensed under the MIT license.
# ----------------------------------------------------------------------------------------------
# Modified from METRO (https://github.com/microsoft/MeshTransformer)
# Copyright (c) Microsoft Corporation. All Rights Reserved [see https://github.com/microsoft/MeshTransformer/blob/main/LICENSE for details]
# ----------------------------------------------------------------------------------------------
"""
Training and evaluation codes for 
3D hand mesh reconstruction from an image
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
import os.path as op
import code
import json
import time
import datetime
import torch
import torchvision.models as models
from torchvision.utils import make_grid
import numpy as np
import cv2
from torch.nn import functional as F
from src.modeling.model import FastMETRO_Hand_Network as FastMETRO_Network
from src.modeling._mano import MANO, Mesh
from src.modeling.hrnet.hrnet_cls_net_featmaps import get_cls_net
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
import src.modeling.data.config as cfg
from src.datasets.build import make_hand_data_loader
from src.utils.logger import setup_logger
from src.utils.comm import is_main_process, get_rank, get_world_size
from src.utils.miscellaneous import mkdir, set_seed
from src.utils.metric_logger import AverageMeter
from src.utils.geometric_layers import orthographic_projection
from src.utils.renderer_opendr import OpenDR_Renderer, visualize_reconstruction_opendr, visualize_reconstruction_multi_view_opendr
try:
    from src.utils.renderer_pyrender import PyRender_Renderer, visualize_reconstruction_pyrender, visualize_reconstruction_multi_view_pyrender
except:
    print("Failed to import renderer_pyrender. Please see docs/Installation.md")
     

def save_checkpoint(model, args, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, iteration))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for _ in range(num_trial):
        try:
            torch.save(model_to_save.state_dict(), op.join(checkpoint_dir, 'state_dict.bin'))
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir

def keypoint_2d_loss(criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d, has_pose_2d):
    """
    Compute 2D reprojection loss if 2D keypoint annotations are available.
    The confidence (conf) is binary and indicates whether the keypoints exist or not.
    """
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
    return loss

def keypoint_3d_loss(criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, device):
    """
    Compute 3D keypoint loss if 3D keypoint annotations are available.
    """
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    conf = conf[has_pose_3d == 1]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
    if len(gt_keypoints_3d) > 0:
        gt_root = gt_keypoints_3d[:, 0,:]
        gt_keypoints_3d = gt_keypoints_3d - gt_root[:, None, :]
        pred_root = pred_keypoints_3d[:, 0,:]
        pred_keypoints_3d = pred_keypoints_3d - pred_root[:, None, :]
        return (conf * criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.).to(device) 

def vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_mesh, device):
    """
    Compute per-vertex loss if vertex annotations are available.
    """
    pred_vertices_with_shape = pred_vertices[has_mesh == 1]
    gt_vertices_with_shape = gt_vertices[has_mesh == 1]
    if len(gt_vertices_with_shape) > 0:
        return criterion_vertices(pred_vertices_with_shape, gt_vertices_with_shape)
    else:
        return torch.FloatTensor(1).fill_(0.).to(device) 


class EdgeLengthGTLoss(torch.nn.Module):
    """
    Modified from [I2L-MeshNet](https://github.com/mks0601/I2L-MeshNet_RELEASE/blob/master/common/nets/loss.py)
    """
    def __init__(self, face):
        super().__init__()
        self.face = face # num_faces X 3

    def forward(self, pred_vertices, gt_vertices, has_mesh, device):
        face = self.face
        coord_out = pred_vertices[has_mesh == 1]
        coord_gt = gt_vertices[has_mesh == 1]
        if len(coord_gt) > 0:
            d1_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,1],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1
            d2_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1
            d3_out = torch.sqrt(torch.sum((coord_out[:,face[:,1],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1

            d1_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,0],:] - coord_gt[:,face[:,1],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1
            d2_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,0],:] - coord_gt[:,face[:,2],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1
            d3_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,1],:] - coord_gt[:,face[:,2],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1

            diff1 = torch.abs(d1_out - d1_gt)
            diff2 = torch.abs(d2_out - d2_gt) 
            diff3 = torch.abs(d3_out - d3_gt) 
            edge_diff = torch.cat((diff1, diff2, diff3),1)
            loss = edge_diff.mean()
        else:
            loss = torch.FloatTensor(1).fill_(0.).to(device) 

        return loss


class NormalVectorLoss(torch.nn.Module):
    """
    Modified from [I2L-MeshNet](https://github.com/mks0601/I2L-MeshNet_RELEASE/blob/master/common/nets/loss.py)
    """
    def __init__(self, face):
        super().__init__()
        self.face = face # num_faces X 3

    def forward(self, pred_vertices, gt_vertices, has_mesh, device):
        face = self.face
        coord_out = pred_vertices[has_mesh == 1]
        coord_gt = gt_vertices[has_mesh == 1]
        if len(coord_gt) > 0:
            v1_out = coord_out[:,face[:,1],:] - coord_out[:,face[:,0],:] # batch_size X num_faces X 3
            v1_out = F.normalize(v1_out, p=2, dim=2) # L2 normalize to make unit vector
            v2_out = coord_out[:,face[:,2],:] - coord_out[:,face[:,0],:] # batch_size X num_faces X 3
            v2_out = F.normalize(v2_out, p=2, dim=2) # L2 normalize to make unit vector
            v3_out = coord_out[:,face[:,2],:] - coord_out[:,face[:,1],:] # batch_size X num_faces X 3
            v3_out = F.normalize(v3_out, p=2, dim=2) # L2 nroamlize to make unit vector

            v1_gt = coord_gt[:,face[:,1],:] - coord_gt[:,face[:,0],:] # batch_size X num_faces X 3
            v1_gt = F.normalize(v1_gt, p=2, dim=2) # L2 normalize to make unit vector
            v2_gt = coord_gt[:,face[:,2],:] - coord_gt[:,face[:,0],:] # batch_size X num_faces X 3
            v2_gt = F.normalize(v2_gt, p=2, dim=2) # L2 normalize to make unit vector
            normal_gt = torch.cross(v1_gt, v2_gt, dim=2) # batch_size X num_faces X 3
            normal_gt = F.normalize(normal_gt, p=2, dim=2) # L2 normalize to make unit vector

            cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True)) # batch_size X num_faces X 1
            cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True)) # batch_size X num_faces X 1
            cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True)) # batch_size X num_faces X 1
            loss = torch.cat((cos1, cos2, cos3),1).mean()
        else:
            loss = torch.FloatTensor(1).fill_(0.).to(device) 

        return loss

def run_train(args, train_dataloader, FastMETRO_model, mano_model, mesh_sampler, renderer):    
    max_iter = len(train_dataloader)
    iters_per_epoch = max_iter // args.num_train_epochs
    args.logging_steps = iters_per_epoch // 2
    iteration = args.resume_epoch * iters_per_epoch

    FastMETRO_model_without_ddp = FastMETRO_model
    if args.distributed:
        FastMETRO_model = torch.nn.parallel.DistributedDataParallel(
            FastMETRO_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
        FastMETRO_model_without_ddp = FastMETRO_model.module
        if is_main_process():
            logger.info(
                    ' '.join(
                    ['Local-Rank: {o}', 'Max-Iteration: {a}', 'Iterations-per-Epoch: {b}','Number-of-Training-Epochs: {c}',]
                    ).format(o=args.local_rank, a=max_iter, b=iters_per_epoch, c=args.num_train_epochs)
                )

    param_dicts = [
        {"params": [p for p in FastMETRO_model_without_ddp.parameters() if p.requires_grad]}
    ]

    # optimizer & learning rate scheduler
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
                        
    # define loss functions for joints & vertices
    criterion_2d_keypoints = torch.nn.L1Loss(reduction='none').cuda(args.device)
    criterion_3d_keypoints = torch.nn.L1Loss(reduction='none').cuda(args.device)
    criterion_3d_vertices = torch.nn.L1Loss().cuda(args.device)
    
    # define loss functions for edge length & normal vector
    edge_gt_loss = EdgeLengthGTLoss(mano_model.face)
    normal_loss = NormalVectorLoss(mano_model.face)

    start_training_time = time.time()
    FastMETRO_model.train()
    log_losses = AverageMeter()
    log_loss_3d_joints = AverageMeter()
    log_loss_3d_vertices = AverageMeter()
    log_loss_edge_normal = AverageMeter()
    log_loss_2d_joints = AverageMeter()

    for _, (img_keys, images, annotations) in enumerate(train_dataloader):
        FastMETRO_model.train()
        iteration = iteration + 1
        epoch = iteration // iters_per_epoch
        batch_size = images.size(0)

        images = images.cuda(args.device) # batch_size X 3 X 224 X 224 

        # gt 2d joints
        gt_2d_joints = annotations['joints_2d'].cuda(args.device)
        gt_pose = annotations['pose'].cuda(args.device)
        gt_betas = annotations['betas'].cuda(args.device)
        has_mesh = annotations['has_smpl'].cuda(args.device)
        has_3d_joints = has_mesh.clone()
        has_2d_joints = has_mesh.clone()

        # generate mesh
        gt_3d_vertices_fine, gt_3d_joints = mano_model.layer(gt_pose, gt_betas)
        gt_3d_vertices_fine = gt_3d_vertices_fine / 1000.0
        gt_3d_joints = gt_3d_joints / 1000.0
        gt_3d_vertices_coarse = mesh_sampler.downsample(gt_3d_vertices_fine)
        
        # normalize gt based on hand's wrist 
        gt_3d_root = gt_3d_joints[:,cfg.J_NAME.index('Wrist'),:]
        gt_3d_vertices_fine = gt_3d_vertices_fine - gt_3d_root[:, None, :]
        gt_3d_vertices_coarse = gt_3d_vertices_coarse - gt_3d_root[:, None, :]
        gt_3d_joints = gt_3d_joints - gt_3d_root[:, None, :]
        gt_3d_joints_with_tag = torch.ones((batch_size,gt_3d_joints.shape[1],4)).cuda(args.device)
        gt_3d_joints_with_tag[:,:,:3] = gt_3d_joints
        
        # forward-pass
        out = FastMETRO_model(images)
        pred_cam, pred_3d_joints_from_token = out['pred_cam'], out['pred_3d_joints']
        pred_3d_vertices_coarse, pred_3d_vertices_fine = out['pred_3d_vertices_coarse'], out['pred_3d_vertices_fine']

        # obtain 3d joints, which are regressed from the full mesh
        pred_3d_joints_from_mano = mano_model.get_3d_joints(pred_3d_vertices_fine) # batch_size X 21 X 3
        pred_3d_joints_from_mano_wrist = pred_3d_joints_from_mano[:,cfg.J_NAME.index('Wrist'),:]
        # normalize predicted vertices 
        pred_3d_vertices_fine = pred_3d_vertices_fine - pred_3d_joints_from_mano_wrist[:, None, :] # batch_size X 778 X 3
        pred_3d_vertices_coarse = pred_3d_vertices_coarse - pred_3d_joints_from_mano_wrist[:, None, :] # batch_size X 195 X 3
        # normalize predicted joints 
        pred_3d_joints_from_mano = pred_3d_joints_from_mano - pred_3d_joints_from_mano_wrist[:, None, :] # batch_size X 21 X 3
        pred_3d_joints_from_token_wrist = pred_3d_joints_from_token[:,0,:]
        pred_3d_joints_from_token = pred_3d_joints_from_token - pred_3d_joints_from_token_wrist[:, None, :] # batch_size X 21 X 3
        # obtain 2d joints, which are projected from 3d joints of mano mesh
        pred_2d_joints_from_mano = orthographic_projection(pred_3d_joints_from_mano.contiguous(), pred_cam.contiguous()) # batch_size X 21 X 2
        pred_2d_joints_from_token = orthographic_projection(pred_3d_joints_from_token.contiguous(), pred_cam.contiguous()) # batch_size X 21 X 2

        # compute 3d joint loss  
        loss_3d_joints = (keypoint_3d_loss(criterion_3d_keypoints, pred_3d_joints_from_token, gt_3d_joints_with_tag, has_3d_joints, args.device) + \
                         keypoint_3d_loss(criterion_3d_keypoints, pred_3d_joints_from_mano, gt_3d_joints_with_tag, has_3d_joints, args.device))
        # compute 3d vertex loss
        loss_3d_vertices = (args.vertices_coarse_loss_weight * vertices_loss(criterion_3d_vertices, pred_3d_vertices_coarse, gt_3d_vertices_coarse, has_mesh, args.device) + \
                           args.vertices_fine_loss_weight * vertices_loss(criterion_3d_vertices, pred_3d_vertices_fine, gt_3d_vertices_fine, has_mesh, args.device))
        # compute edge length loss (GT supervision) & normal vector loss
        loss_edge_normal = (args.edge_gt_loss_weight * edge_gt_loss(pred_3d_vertices_fine, gt_3d_vertices_fine, has_mesh, args.device) + \
                           args.normal_loss_weight * normal_loss(pred_3d_vertices_fine, gt_3d_vertices_fine, has_mesh, args.device))
        # compute 2d joint loss
        loss_2d_joints = (keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_token, gt_2d_joints, has_2d_joints) + \
                         keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_mano, gt_2d_joints, has_2d_joints))
        
        # empirically set hyperparameters to balance different losses
        loss = (args.joints_3d_loss_weight * loss_3d_joints + \
               args.vertices_3d_loss_weight * loss_3d_vertices + \
               args.edge_normal_loss_weight * loss_edge_normal + \
               args.joints_2d_loss_weight * loss_2d_joints)

        # update logs
        log_loss_3d_joints.update(loss_3d_joints.item(), batch_size)
        log_loss_3d_vertices.update(loss_3d_vertices.item(), batch_size)
        log_loss_edge_normal.update(loss_edge_normal.item(), batch_size)
        log_loss_2d_joints.update(loss_2d_joints.item(), batch_size)
        log_losses.update(loss.item(), batch_size)

        # back-propagation
        optimizer.zero_grad()
        loss.backward() 
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(FastMETRO_model.parameters(), args.clip_max_norm)
        optimizer.step()

        # logging
        if (iteration % (iters_per_epoch//2) == 0) and is_main_process():
            print("Complete", iteration, "th iterations!")
        if ((iteration == 10) or (iteration == 100) or ((iteration % args.logging_steps) == 0) or (iteration == max_iter)) and is_main_process():
            logger.info(
                ' '.join(
                ['epoch: {ep}', 'iterations: {iter}',]
                ).format(ep=epoch, iter=iteration,) 
                + ' loss: {:.4f}, 3D-joint-loss: {:.4f}, 3D-vertex-loss: {:.4f}, edge-normal-loss: {:.4f}, 2D-joint-loss: {:.4f}, lr: {:.6f}'.format(
                    log_losses.avg, log_loss_3d_joints.avg, log_loss_3d_vertices.avg, log_loss_edge_normal.avg, log_loss_2d_joints.avg,
                    optimizer.param_groups[0]['lr'])
            )
            # visualize estimation results during training
            if args.visualize_training and (iteration >= args.logging_steps):
                visual_imgs = visualize_mesh(renderer,
                                            annotations['ori_img'].detach(),
                                            pred_3d_vertices_fine.detach(), 
                                            pred_cam.detach())
                visual_imgs = visual_imgs.transpose(0,1)
                visual_imgs = visual_imgs.transpose(1,2)
                visual_imgs = np.asarray(visual_imgs)
                if is_main_process():
                    stamp = str(epoch) + '_' + str(iteration)
                    temp_fname = args.output_dir + 'visual_' + stamp + '.jpg'
                    if args.use_opendr_renderer:
                        visual_imgs[:,:,::-1] = visual_imgs[:,:,::-1]*255
                    cv2.imwrite(temp_fname, np.asarray(visual_imgs[:,:,::-1]))

        # save checkpoint
        if (iteration % iters_per_epoch) == 0:
            lr_scheduler.step()
            if (epoch != 0) and ((epoch % args.saving_epochs) == 0):
                checkpoint_dir = save_checkpoint(FastMETRO_model, args, epoch, iteration)
        
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info('Total training time: {} ({:.4f} s / iter)'.format(total_time_str, total_training_time / max_iter))
    checkpoint_dir = save_checkpoint(FastMETRO_model, args, epoch, iteration)


def run_eval_and_save(args, split, val_dataloader, FastMETRO_model, mano_model, renderer, mesh_sampler):
    if args.distributed:
        FastMETRO_model = torch.nn.parallel.DistributedDataParallel(
            FastMETRO_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    FastMETRO_model.eval()

    run_inference_hand_mesh(args, val_dataloader, 
                            FastMETRO_model, 
                            mano_model, renderer, split)
    checkpoint_dir = save_checkpoint(FastMETRO_model, args, 0, 0)
    
    logger.info("The experiment completed successfully. Finalizing run...")
    
    return


def run_inference_hand_mesh(args, val_loader, FastMETRO_model, mano_model, renderer, split):
    # switch to evaluate mode
    FastMETRO_model.eval()
    fname_output_save = []
    mesh_output_save = []
    joint_output_save = []
    
    with torch.no_grad():
        for i, (img_keys, images, annotations) in enumerate(val_loader):
            batch_size = images.size(0)
            # compute output
            images = images.cuda(args.device) # batch_size X 3 X 224 X 224 

            # forward-pass
            out = FastMETRO_model(images)
            pred_cam, pred_3d_vertices_fine = out['pred_cam'], out['pred_3d_vertices_fine']

            # obtain 3d joints from full mesh
            pred_3d_joints_from_mano = mano_model.get_3d_joints(pred_3d_vertices_fine)
            pred_3d_joints_from_mano_wrist = pred_3d_joints_from_mano[:,cfg.J_NAME.index('Wrist'),:]
            
            # normalize predicted vertices 
            pred_3d_vertices_fine = pred_3d_vertices_fine - pred_3d_joints_from_mano_wrist[:, None, :]
            # normalize predicted joints 
            pred_3d_joints_from_mano = pred_3d_joints_from_mano - pred_3d_joints_from_mano_wrist[:, None, :]
            
            for j in range(batch_size):
                fname_output_save.append(img_keys[j])
                pred_3d_vertices_list = pred_3d_vertices_fine[j].tolist()
                mesh_output_save.append(pred_3d_vertices_list)
                pred_3d_joints_from_mano_list = pred_3d_joints_from_mano[j].tolist()
                joint_output_save.append(pred_3d_joints_from_mano_list)

            if args.run_eval_and_visualize:
                if (i % 20) == 0:
                    # obtain 3d joints, which are regressed from the full mesh
                    pred_3d_joints_from_mano = mano_model.get_3d_joints(pred_3d_vertices_fine)
                    # obtain 2d joints, which are projected from 3d joints of mesh
                    pred_2d_joints_from_mano = orthographic_projection(pred_3d_joints_from_mano.contiguous(), pred_cam.contiguous())
                    
                    # visualization
                    visual_imgs = visualize_mesh(renderer,
                                                annotations['ori_img'].detach(),
                                                pred_3d_vertices_fine.detach(), 
                                                pred_cam.detach())
                    visual_imgs = visual_imgs.transpose(0,1)
                    visual_imgs = visual_imgs.transpose(1,2)
                    visual_imgs = np.asarray(visual_imgs)
                    
                    inference_setting = 'sc%02d_rot%s'%(int(args.sc*10),str(int(args.rot)))
                    temp_fname = args.output_dir + 'visual_' + inference_setting + '_batch' + str(i) + '.jpg'
                    if args.use_opendr_renderer:
                        visual_imgs[:,:,::-1] = visual_imgs[:,:,::-1]*255
                    cv2.imwrite(temp_fname, np.asarray(visual_imgs[:,:,::-1]))

    print('save results to pred.json')
    with open('pred.json', 'w') as f:
        json.dump([joint_output_save, mesh_output_save], f)

    inference_setting = 'sc%02d_rot%s'%(int(args.sc*10),str(int(args.rot)))
    resolved_submit_cmd = 'zip ' + args.output_dir + 'FastMETRO-' + inference_setting +'-pred.zip  ' +  'pred.json'
    print(resolved_submit_cmd)
    os.system(resolved_submit_cmd)
    resolved_submit_cmd = 'rm pred.json'
    print(resolved_submit_cmd)
    os.system(resolved_submit_cmd)
    
    return 

def visualize_mesh(renderer, images, pred_vertices, pred_cam):
    rend_imgs = []
    batch_size = pred_vertices.shape[0]
    for i in range(batch_size):
        img = images[i].cpu().numpy().transpose(1,2,0)
        # Get predicted vertices for the particular example
        vertices = pred_vertices[i].cpu().numpy()
        cam = pred_cam[i].cpu().numpy()
        # Visualize reconstruction
        if args.use_opendr_renderer:
            if args.visualize_multi_view:
                rend_img = visualize_reconstruction_multi_view_opendr(img, vertices, cam, renderer)
            else:
                rend_img = visualize_reconstruction_opendr(img, vertices, cam, renderer)
        else:
            if args.visualize_multi_view:
                rend_img = visualize_reconstruction_multi_view_pyrender(img, vertices, cam, renderer)
            else:
                rend_img = visualize_reconstruction_pyrender(img, vertices, cam, renderer)
        rend_img = rend_img.transpose(2,0,1)
        rend_imgs.append(torch.from_numpy(rend_img))   
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs

def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--data_dir", default='datasets', type=str, required=False,
                        help="Directory with all datasets, each in one subfolder")
    parser.add_argument("--train_yaml", default='freihand/train.yaml', type=str, required=False,
                        help="Yaml file with all data for training.")
    parser.add_argument("--val_yaml", default='freihand/test.yaml', type=str, required=False,
                        help="Yaml file with all data for validation.")
    parser.add_argument("--num_workers", default=4, type=int, 
                        help="Workers in dataloader.")
    parser.add_argument("--img_scale_factor", default=1, type=int, 
                        help="adjust image resolution.") 
    #########################################################
    # Loading/Saving checkpoints
    #########################################################
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--saving_epochs", default=20, type=int)
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--resume_epoch", default=0, type=int)
    #########################################################
    # Training parameters
    #########################################################
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--lr', "--learning_rate", default=1e-4, type=float, 
                        help="The initial lr.")
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm', default=0.3, type=float,
                        help='gradient clipping maximal norm')
    parser.add_argument("--num_train_epochs", default=200, type=int, 
                        help="Total number of training epochs to perform.")
    # Loss coefficients
    parser.add_argument("--joints_2d_loss_weight", default=100.0, type=float)
    parser.add_argument("--vertices_3d_loss_weight", default=100.0, type=float)
    parser.add_argument("--edge_normal_loss_weight", default=100.0, type=float)
    parser.add_argument("--joints_3d_loss_weight", default=1000.0, type=float)
    parser.add_argument("--vertices_fine_loss_weight", default=0.50, type=float) 
    parser.add_argument("--vertices_coarse_loss_weight", default=0.50, type=float)
    parser.add_argument("--edge_gt_loss_weight", default=1.0, type=float) 
    parser.add_argument("--normal_loss_weight", default=0.1, type=float)
    # Model parameters
    parser.add_argument("--model_name", default='FastMETRO-L', type=str,
                        help='Transformer architecture: FastMETRO-S, FastMETRO-M, FastMETRO-L')
    parser.add_argument("--model_dim_1", default=512, type=int)
    parser.add_argument("--model_dim_2", default=128, type=int)
    parser.add_argument("--feedforward_dim_1", default=2048, type=int)
    parser.add_argument("--feedforward_dim_2", default=512, type=int)
    parser.add_argument("--conv_1x1_dim", default=2048, type=int)
    parser.add_argument("--transformer_dropout", default=0.1, type=float)
    parser.add_argument("--transformer_nhead", default=8, type=int)
    parser.add_argument("--pos_type", default='sine', type=str)    
    # CNN backbone
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                        help='CNN backbone architecture: hrnet-w64, resnet50')
    #########################################################
    # Others
    #########################################################
    parser.add_argument("--run_evaluation", default=False, action='store_true',) 
    parser.add_argument("--run_eval_and_visualize", default=False, action='store_true',)
    parser.add_argument('--logging_steps', type=int, default=1000, 
                        help="Log every X steps.")
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")
    parser.add_argument("--local_rank", type=int, default=0, 
                        help="For distributed training.")
    parser.add_argument('--model_save', default=False, action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--exp", default='FastMETRO', type=str, required=False)
    parser.add_argument("--visualize_training", default=False, action='store_true',) 
    parser.add_argument("--visualize_multi_view", default=False, action='store_true',) 
    parser.add_argument("--use_opendr_renderer", default=False, action='store_true',) 
    parser.add_argument("--multiscale_inference", default=False, action='store_true',) 
    # if enable "multiscale_inference", dataloader will apply transformations to the test image based on
    # the rotation "rot" and scale "sc" parameters below 
    parser.add_argument("--rot", default=0, type=float) 
    parser.add_argument("--sc", default=1.0, type=float) 
    parser.add_argument("--aml_eval", default=False, action='store_true',) 


    args = parser.parse_args()
    return args


def main(args):
    print("FastMETRO for 3D Hand Mesh Reconstruction!")
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)
    if args.distributed:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        print("Init distributed training on local rank {} ({}), world size {}".format(args.local_rank, int(os.environ["LOCAL_RANK"]), args.num_gpus))
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        args.device = torch.device("cuda", local_rank)
        torch.distributed.barrier()
    mkdir(args.output_dir)
    logger = setup_logger("FastMETRO", args.output_dir, get_rank())
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and MANO utils
    mano_model = MANO().to(args.device)
    mano_model.layer = mano_model.layer.to(args.device)
    mesh_sampler = Mesh()

    # Renderer for visualization
    if args.use_opendr_renderer:
        renderer = OpenDR_Renderer(faces=mano_model.face)
    else:
        renderer = PyRender_Renderer(faces=mano_model.face)
    
    logger.info("Training Arguments %s", args)
    
    # Load model
    if args.run_evaluation and (args.resume_checkpoint != None) and (args.resume_checkpoint != 'None') and ('state_dict' not in args.resume_checkpoint):
        # if only run eval, load checkpoint
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _FastMETRO_Network = torch.load(args.resume_checkpoint)
    else:
        # init ImageNet pre-trained backbone model
        if args.arch == 'hrnet-w64':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w64 model')
        elif args.arch == 'resnet50':
            logger.info("=> using pre-trained model '{}'".format(args.arch))
            backbone = models.__dict__[args.arch](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        else:
            assert False, "The CNN backbone name is not valid"

        _FastMETRO_Network = FastMETRO_Network(args, backbone, mesh_sampler)
        # number of parameters
        overall_params = sum(p.numel() for p in _FastMETRO_Network.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        transformer_params = overall_params - backbone_params
        logger.info('Number of CNN Backbone learnable parameters: {}'.format(backbone_params))
        logger.info('Number of Transformer Encoder-Decoder learnable parameters: {}'.format(transformer_params))
        logger.info('Number of Overall learnable parameters: {}'.format(overall_params))
        
        if (args.resume_checkpoint != None) and (args.resume_checkpoint != 'None'):
            # for fine-tuning or resume training or inference, load weights from checkpoint
            logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
            cpu_device = torch.device('cpu')
            state_dict = torch.load(args.resume_checkpoint, map_location=cpu_device)
            _FastMETRO_Network.load_state_dict(state_dict, strict=False)
            del state_dict
    
    _FastMETRO_Network.to(args.device)

    if args.run_evaluation:
        val_dataloader = make_hand_data_loader(args, args.val_yaml, args.distributed, is_train=False, scale_factor=args.img_scale_factor)
        run_eval_and_save(args, 'freihand', val_dataloader, _FastMETRO_Network, mano_model, renderer, mesh_sampler)
    else:
        train_dataloader = make_hand_data_loader(args, args.train_yaml, args.distributed, is_train=True, scale_factor=args.img_scale_factor)
        run_train(args, train_dataloader, _FastMETRO_Network, mano_model, mesh_sampler, renderer)


if __name__ == "__main__":
    args = parse_args()
    main(args)