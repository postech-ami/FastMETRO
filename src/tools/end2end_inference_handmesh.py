# ----------------------------------------------------------------------------------------------
# FastMETRO Official Code
# Copyright (c) POSTECH Algorithmic Machine Intelligence Lab. (P-AMI Lab.) All Rights Reserved 
# Licensed under the MIT license.
# ----------------------------------------------------------------------------------------------
# Modified from METRO (https://github.com/microsoft/MeshTransformer)
# Copyright (c) Microsoft Corporation. All Rights Reserved [see https://github.com/microsoft/MeshTransformer/blob/main/LICENSE for details]
# ----------------------------------------------------------------------------------------------
"""
End-to-End inference codes for 
3D hand mesh reconstruction from an image
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
import os.path as op
import torch
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from src.modeling.model import FastMETRO_Hand_Network as FastMETRO_Network
from src.modeling._mano import MANO, Mesh
from src.modeling.hrnet.hrnet_cls_net_featmaps import get_cls_net
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
import src.modeling.data.config as cfg
from src.utils.logger import setup_logger
from src.utils.miscellaneous import mkdir, set_seed
from src.utils.geometric_layers import orthographic_projection
from src.utils.renderer_opendr import OpenDR_Renderer, visualize_reconstruction_opendr
try:
    from src.utils.renderer_pyrender import PyRender_Renderer, visualize_reconstruction_pyrender
except:
    print("Failed to import renderer_pyrender. Please see docs/Installation.md")


transform = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_visualize = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()])                   

def run_inference(args, image_list, FastMETRO_model, mano_model, renderer):
    # switch to evaluate mode
    FastMETRO_model.eval()
    
    for image_file in image_list:
        if 'pred' not in image_file:
            img = Image.open(image_file)
            img_tensor = transform(img)
            img_visual = transform_visualize(img)

            batch_imgs = torch.unsqueeze(img_tensor, 0).cuda()
            batch_visual_imgs = torch.unsqueeze(img_visual, 0).cuda()

            # forward-pass
            out = FastMETRO_model(batch_imgs)
            pred_cam, pred_3d_vertices_fine = out['pred_cam'], out['pred_3d_vertices_fine']     
            
            # obtain 3d joints from full mesh
            pred_3d_joints_from_mano = mano_model.get_3d_joints(pred_3d_vertices_fine)
            pred_3d_joints_from_mano_wrist = pred_3d_joints_from_mano[:,cfg.J_NAME.index('Wrist'),:]
            # normalize predicted vertices 
            pred_3d_vertices_fine = pred_3d_vertices_fine - pred_3d_joints_from_mano_wrist[:, None, :]
            # normalize predicted joints 
            pred_3d_joints_from_mano = pred_3d_joints_from_mano - pred_3d_joints_from_mano_wrist[:, None, :]

            # visualization
            visual_img = visualize_mesh(renderer,
                                        batch_visual_imgs[0],
                                        pred_3d_vertices_fine[0].detach(), 
                                        pred_cam[0].detach())
            visual_img = visual_img.transpose(1,2,0)
            visual_img = np.asarray(visual_img)
            if args.use_opendr_renderer:
                visual_img[:,:,::-1] = visual_img[:,:,::-1]*255
            temp_fname = image_file[:-4] + '_fastmetro_pred.jpg'
            print('save to ', temp_fname)
            cv2.imwrite(temp_fname, np.asarray(visual_img[:,:,::-1]))
        
    logger.info("The inference completed successfully. Finalizing run...")

    return 


def visualize_mesh(renderer, image, pred_vertices, pred_cam):
    img = image.cpu().numpy().transpose(1,2,0)

    # Get predicted vertices for the particular example
    vertices = pred_vertices.cpu().numpy()
    cam = pred_cam.cpu().numpy()

    # Visualize reconstruction
    if args.use_opendr_renderer:
        rend_img = visualize_reconstruction_opendr(img, vertices, cam, renderer)
    else:
        rend_img = visualize_reconstruction_pyrender(img, vertices, cam, renderer)
    rend_img = rend_img.transpose(2,0,1)
    
    return rend_img


def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--image_file_or_path", default='./test_images/hand', type=str, 
                        help="test data")
    #########################################################
    # Loading/Saving checkpoints
    #########################################################
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    #########################################################
    # Model architectures
    #########################################################
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
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")
    parser.add_argument("--use_opendr_renderer", default=False, action='store_true',) 


    args = parser.parse_args()
    return args


def main(args):
    print("FastMETRO for 3D Hand Mesh Reconstruction!")
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)

    mkdir(args.output_dir)
    logger = setup_logger("FastMETRO Inference", args.output_dir, 0)
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

    # Load pretrained model    
    logger.info("Inference: Loading from checkpoint {}".format(args.resume_checkpoint))

    if (args.resume_checkpoint != None) and (args.resume_checkpoint != 'None') and ('state_dict' not in args.resume_checkpoint):
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
    logger.info("Run inference")

    image_list = []
    if not args.image_file_or_path:
        raise ValueError("image_file_or_path not specified")
    if op.isfile(args.image_file_or_path):
        image_list = [args.image_file_or_path]
    elif op.isdir(args.image_file_or_path):
        # should be a path with images only
        for filename in os.listdir(args.image_file_or_path):
            if filename.endswith(".png") or filename.endswith(".jpg") and 'pred' not in filename:
                image_list.append(args.image_file_or_path+'/'+filename) 
    else:
        raise ValueError("Cannot find images at {}".format(args.image_file_or_path))

    run_inference(args, image_list, _FastMETRO_Network, mano_model, renderer) 

if __name__ == "__main__":
    args = parse_args()
    main(args)