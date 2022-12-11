# ----------------------------------------------------------------------------------------------
# Modified from Pose2Mesh (https://github.com/hongsukchoi/Pose2Mesh_RELEASE)
# Copyright (c) Hongsuk Choi. All Rights Reserved [see https://github.com/hongsukchoi/Pose2Mesh_RELEASE/blob/main/LICENSE for details]
# ----------------------------------------------------------------------------------------------

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
import numpy as np
import torch.nn.functional as F
import math
import cv2
import trimesh
import pyrender
from pyrender.constants import RenderFlags


class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self, scale, translation, znear=pyrender.camera.DEFAULT_Z_NEAR, zfar=None, name=None):
        super(WeakPerspectiveCamera, self).__init__(znear=znear, zfar=zfar, name=name)
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P


class PyRender_Renderer:
    def __init__(self, resolution=(224, 224), faces=None, orig_img=False, wireframe=False):
        self.resolution = resolution
        self.faces = faces
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(viewport_width=self.resolution[0],
                                                   viewport_height=self.resolution[1],
                                                   point_size=1.0)

        # set the scene & create light source
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.05, 0.05, 0.05))
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        light_pose = trimesh.transformations.rotation_matrix(np.radians(-45), [1, 0, 0])
        self.scene.add(light, pose=light_pose)
        light_pose = trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0])
        self.scene.add(light, pose=light_pose)

        # mesh colors
        self.colors_dict = {'blue': np.array([0.35, 0.60, 0.92]),
                            'neutral': np.array([0.7, 0.7, 0.6]),
                            'pink': np.array([0.7, 0.5, 0.5]),
                            'white': np.array([1.0, 0.98, 0.94]),
                            'green': np.array([0.5, 0.55, 0.3]),
                            'sky': np.array([0.3, 0.5, 0.55])}

    def __call__(self, verts, img=np.zeros((224, 224, 3)), cam=np.array([1, 0, 0]),
                 angle=None, axis=None, mesh_filename=None, color_type=None, color=[0.7, 0.7, 0.6]):
        if color_type != None:
            color = self.colors_dict[color_type]
            
        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)
        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)
        if mesh_filename is not None:
            mesh.export(mesh_filename)
        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        sy, tx, ty = cam
        sx = sy
        camera = WeakPerspectiveCamera(scale=[sx, sy], translation=[tx, ty], zfar=1000.0)

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            roughnessFactor=1.0,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        mesh_node = self.scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgb, depth = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (depth > 0)[:, :, np.newaxis]
        output_img = rgb[:, :, :3] * valid_mask + (1 - valid_mask) * img
        image = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        return image


def visualize_reconstruction_pyrender(img, vertices, camera, renderer, color='blue', focal_length=1000):
    img = (img * 255).astype(np.uint8)
    save_mesh_path = None
    rend_color = color

    # Render front view
    rend_img = renderer(vertices,
                        img=img,
                        cam=camera,
                        color_type=rend_color,
                        mesh_filename=save_mesh_path)
    
    combined = np.hstack([img, rend_img])
    
    return combined

def visualize_reconstruction_multi_view_pyrender(img, vertices, camera, renderer, color='blue', focal_length=1000):
    img = (img * 255).astype(np.uint8)
    save_mesh_path = None
    rend_color = color
    
    # Render front view
    rend_img = renderer(vertices,
                        img=img,
                        cam=camera,
                        color_type=rend_color,
                        mesh_filename=save_mesh_path)

    # Render side views    
    aroundy0 = cv2.Rodrigues(np.array([0, np.radians(0.), 0]))[0]
    aroundy1 = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
    aroundy2 = cv2.Rodrigues(np.array([0, np.radians(180.), 0]))[0]
    aroundy3 = cv2.Rodrigues(np.array([0, np.radians(270.), 0]))[0]
    aroundy4 = cv2.Rodrigues(np.array([0, np.radians(45.), 0]))[0]
    center = vertices.mean(axis=0)
    rot_vertices0 = np.dot((vertices - center), aroundy0) + center
    rot_vertices1 = np.dot((vertices - center), aroundy1) + center
    rot_vertices2 = np.dot((vertices - center), aroundy2) + center
    rot_vertices3 = np.dot((vertices - center), aroundy3) + center
    rot_vertices4 = np.dot((vertices - center), aroundy4) + center

    # Render side-view shape
    img_side0 = renderer(rot_vertices0,
                        img=np.ones_like(img)*255,
                        cam=camera,
                        color_type=rend_color,
                        mesh_filename=save_mesh_path)
    img_side1 = renderer(rot_vertices1,
                        img=np.ones_like(img)*255,
                        cam=camera,
                        color_type=rend_color,
                        mesh_filename=save_mesh_path)
    img_side2 = renderer(rot_vertices2,
                        img=np.ones_like(img)*255,
                        cam=camera,
                        color_type=rend_color,
                        mesh_filename=save_mesh_path)
    img_side3 = renderer(rot_vertices3,
                        img=np.ones_like(img)*255,
                        cam=camera,
                        color_type=rend_color,
                        mesh_filename=save_mesh_path)
    img_side4 = renderer(rot_vertices4,
                        img=np.ones_like(img)*255,
                        cam=camera,
                        color_type=rend_color,
                        mesh_filename=save_mesh_path)
    
    combined = np.hstack([img, rend_img, img_side0, img_side1, img_side2, img_side3, img_side4])
    
    return combined

def visualize_reconstruction_smpl_pyrender(img, vertices, camera, renderer, smpl_vertices, color='blue', focal_length=1000):
    img = (img * 255).astype(np.uint8)
    save_mesh_path = None
    rend_color = color

    # Render front view
    rend_img = renderer(vertices,
                        img=img,
                        cam=camera,
                        color_type=rend_color,
                        mesh_filename=save_mesh_path)
    
    rend_img_smpl = renderer(smpl_vertices,
                            img=img,
                            cam=camera,
                            color_type=rend_color,
                            mesh_filename=save_mesh_path)
    
    combined = np.hstack([img, rend_img, rend_img_smpl])
    
    return combined