# ----------------------------------------------------------------------------------------------
# Modified from METRO (https://github.com/microsoft/MeshTransformer)
# Copyright (c) Microsoft Corporation. All Rights Reserved [see https://github.com/microsoft/MeshTransformer/blob/main/LICENSE for details]
# ----------------------------------------------------------------------------------------------
"""
Rendering tools for 3D mesh visualization on 2D image.
Parts of the code are taken from https://github.com/akanazawa/hmr
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import code
import opendr
import math
from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer, TexturedRenderer
from opendr.lighting import LambertianPointLight
import random


# Rotate the points by a specified angle.
def rotateY(points, angle):
    ry = np.array([
        [np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ])
    return np.dot(points, ry)

def draw_skeleton(input_image, joints, draw_edges=True, vis=None, radius=None):
    """
    joints is 3 x 19. but if not will transpose it.
    0: Right ankle
    1: Right knee
    2: Right hip
    3: Left hip
    4: Left knee
    5: Left ankle
    6: Right wrist
    7: Right elbow
    8: Right shoulder
    9: Left shoulder
    10: Left elbow
    11: Left wrist
    12: Neck
    13: Head top
    14: nose
    15: left_eye
    16: right_eye
    17: left_ear
    18: right_ear
    """

    if radius is None:
        radius = max(4, (np.mean(input_image.shape[:2]) * 0.01).astype(int))

    colors = {
        'pink': (197, 27, 125),  # L lower leg
        'light_pink': (233, 163, 201),  # L upper leg
        'light_green': (161, 215, 106),  # L lower arm
        'green': (77, 146, 33),  # L upper arm
        'red': (215, 48, 39),  # head
        'light_red': (252, 146, 114),  # head
        'light_orange': (252, 141, 89),  # chest
        'purple': (118, 42, 131),  # R lower leg
        'light_purple': (175, 141, 195),  # R upper
        'light_blue': (145, 191, 219),  # R lower arm
        'blue': (69, 117, 180),  # R upper arm
        'gray': (130, 130, 130),  #
        'white': (255, 255, 255),  #
    }

    image = input_image.copy()
    input_is_float = False

    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        max_val = image.max()
        if max_val <= 2.:  # should be 1 but sometimes it's slightly above 1
            image = (image * 255).astype(np.uint8)
        else:
            image = (image).astype(np.uint8)

    if joints.shape[0] != 2:
        joints = joints.T
    joints = np.round(joints).astype(int)

    jcolors = [
        'light_pink', 'light_pink', 'light_pink', 'pink', 'pink', 'pink',
        'light_blue', 'light_blue', 'light_blue', 'blue', 'blue', 'blue',
        'purple', 'purple', 'red', 'green', 'green', 'white', 'white',
        'purple', 'purple', 'red', 'green', 'green', 'white', 'white'
    ]

    if joints.shape[1] == 19:
        # parent indices -1 means no parents
        parents = np.array([
            1, 2, 8, 9, 3, 4, 7, 8, 12, 12, 9, 10, 14, -1, 13, -1, -1, 15, 16
        ])
        # Left is light and right is dark
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            8: 'light_blue',
            9: 'blue',
            10: 'blue',
            11: 'blue',
            12: 'purple',
            17: 'light_green',
            18: 'light_green',
            14: 'purple'
        }
    elif joints.shape[1] == 14:
        parents = np.array([
            1,
            2,
            8,
            9,
            3,
            4,
            7,
            8,
            -1,
            -1,
            9,
            10,
            13,
            -1,
        ])
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            10: 'light_blue',
            11: 'blue',
            12: 'purple'
        }
    elif joints.shape[1] == 21:  # hand
        parents = np.array([
            -1,
            0,
            1,
            2,
            3,
            0,
            5,
            6,
            7,
            0,
            9,
            10,
            11,
            0,
            13,
            14,
            15,
            0,
            17,
            18,
            19,
        ])
        ecolors = {
            0: 'light_purple',
            1: 'light_green',
            2: 'light_green',
            3: 'light_green',
            4: 'light_green',
            5: 'pink',
            6: 'pink',
            7: 'pink',
            8: 'pink',
            9: 'light_blue',
            10: 'light_blue',
            11: 'light_blue',
            12: 'light_blue',
            13: 'light_red',
            14: 'light_red',
            15: 'light_red',
            16: 'light_red',
            17: 'purple',
            18: 'purple',
            19: 'purple',
            20: 'purple',
        }
    else:
        print('Unknown skeleton!!')

    for child in range(len(parents)):
        point = joints[:, child]
        # If invisible skip
        if vis is not None and vis[child] == 0:
            continue
        if draw_edges:
            cv2.circle(image, (point[0], point[1]), radius, colors['white'],
                       -1)
            cv2.circle(image, (point[0], point[1]), radius - 1,
                       colors[jcolors[child]], -1)
        else:
            # cv2.circle(image, (point[0], point[1]), 5, colors['white'], 1)
            cv2.circle(image, (point[0], point[1]), radius - 1,
                       colors[jcolors[child]], 1)
            # cv2.circle(image, (point[0], point[1]), 5, colors['gray'], -1)
        pa_id = parents[child]
        if draw_edges and pa_id >= 0:
            if vis is not None and vis[pa_id] == 0:
                continue
            point_pa = joints[:, pa_id]
            cv2.circle(image, (point_pa[0], point_pa[1]), radius - 1,
                       colors[jcolors[pa_id]], -1)
            if child not in ecolors.keys():
                print('bad')
            cv2.line(image, (point[0], point[1]), (point_pa[0], point_pa[1]),
                     colors[ecolors[child]], radius - 2)

    # Convert back in original dtype
    if input_is_float:
        if max_val <= 1.:
            image = image.astype(np.float32) / 255.
        else:
            image = image.astype(np.float32)

    return image

def draw_text(input_image, content):
    """
    content is a dict. draws key: val on image
    Assumes key is str, val is float
    """
    image = input_image.copy()
    input_is_float = False
    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        image = (image * 255).astype(np.uint8)

    black = (255, 255, 0)
    margin = 15
    start_x = 5
    start_y = margin
    for key in sorted(content.keys()):
        text = "%s: %.2g" % (key, content[key])
        cv2.putText(image, text, (start_x, start_y), 0, 0.45, black)
        start_y += margin

    if input_is_float:
        image = image.astype(np.float32) / 255.
    return image

def visualize_reconstruction_opendr(img, vertices, camera, renderer, color='light_blue', focal_length=1000):
    """
    Renderer is an instance of OpenDR Renderer.
    """
    # Fix a flength so i can render this with persp correct scale
    res = img.shape[1]
    camera_t = np.array([camera[1], camera[2], 2*focal_length/(res * camera[0] +1e-9)])
    rend_img = renderer.render(vertices, camera_t=camera_t,
                               img=img, use_bg=True,
                               focal_length=focal_length,
                               body_color=color)

    combined = np.hstack([img, rend_img])

    return combined

def visualize_reconstruction_multi_view_opendr(img, vertices, camera, renderer, color='light_blue', focal_length=1000):
    # Fix a flength so i can render this with persp correct scale
    res = img.shape[1]
    camera_t = np.array([camera[1], camera[2], 2*focal_length/(res * camera[0] +1e-9)])
    rend_img = renderer.render(vertices, camera_t=camera_t,
                               img=img, use_bg=True,
                               focal_length=focal_length,
                               body_color='light_blue')

    # rotate
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
    
    # # Render non-parametric shape
    img_side0 = renderer.render(rot_vertices0, camera_t=camera_t,
                               img=np.ones_like(img), use_bg=True,
                               focal_length=focal_length,
                               body_color='light_blue')
    img_side1 = renderer.render(rot_vertices1, camera_t=camera_t,
                               img=np.ones_like(img), use_bg=True,
                               focal_length=focal_length,
                               body_color='light_blue')
    img_side2 = renderer.render(rot_vertices2, camera_t=camera_t,
                               img=np.ones_like(img), use_bg=True,
                               focal_length=focal_length,
                               body_color='light_blue')
    img_side3 = renderer.render(rot_vertices3, camera_t=camera_t,
                               img=np.ones_like(img), use_bg=True,
                               focal_length=focal_length,
                               body_color='light_blue')
    img_side4 = renderer.render(rot_vertices4, camera_t=camera_t,
                               img=np.ones_like(img), use_bg=True,
                               focal_length=focal_length,
                               body_color='light_blue')

    combined = np.hstack([img, rend_img, img_side0, img_side1, img_side2, img_side3, img_side4])

    return combined

def visualize_reconstruction_smpl_opendr(img, vertices, camera, renderer, smpl_vertices, color='light_blue', focal_length=1000):
    # Fix a flength so i can render this with persp correct scale
    res = img.shape[1]
    camera_t = np.array([camera[1], camera[2], 2*focal_length/(res * camera[0] +1e-9)])
    rend_img = renderer.render(vertices, camera_t=camera_t,
                               img=img, use_bg=True,
                               focal_length=focal_length,
                               body_color=color)
    
    rend_img_smpl = renderer.render(smpl_vertices, camera_t=camera_t,
                                    img=img, use_bg=True,
                                    focal_length=focal_length,
                                    body_color=color)

    combined = np.hstack([img, rend_img, rend_img_smpl])

    return combined


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2]) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2]) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord


class OpenDR_Renderer(object):
    """
    Render mesh using OpenDR for visualization.
    """

    def __init__(self, width=800, height=600, near=0.5, far=1000, faces=None):
        self.colors = {'hand': [.9, .9, .9], 'pink': [.9, .7, .7], 'light_blue': [0.65098039, 0.74117647, 0.85882353] }
        self.width = width
        self.height = height
        self.faces = faces
        self.renderer = ColoredRenderer()

    def render(self, vertices, faces=None, img=None,
               camera_t=np.zeros([3], dtype=np.float32),
               camera_rot=np.zeros([3], dtype=np.float32),
               camera_center=None,
               use_bg=False,
               bg_color=(0.0, 0.0, 0.0),
               body_color=None,
               focal_length=5000,
               disp_text=False,
               gt_keyp=None,
               pred_keyp=None,
               **kwargs):
        if img is not None:
            height, width = img.shape[:2]
        else:
            height, width = self.height, self.width

        if faces is None:
            faces = self.faces

        if camera_center is None:
            camera_center = np.array([width * 0.5,
                                      height * 0.5])

        self.renderer.camera = ProjectPoints(rt=camera_rot,
                                             t=camera_t,
                                             f=focal_length * np.ones(2),
                                             c=camera_center,
                                             k=np.zeros(5))
        dist = np.abs(self.renderer.camera.t.r[2] -
                      np.mean(vertices, axis=0)[2])
        far = dist + 20

        self.renderer.frustum = {'near': 1.0, 'far': far,
                                 'width': width,
                                 'height': height}

        if img is not None:
            if use_bg:
                self.renderer.background_image = img
            else:
                self.renderer.background_image = np.ones_like(
                    img) * np.array(bg_color)

        if body_color is None:
            color = self.colors['light_blue']
        else:
            color = self.colors[body_color]

        if isinstance(self.renderer, TexturedRenderer):
            color = [1.,1.,1.]

        self.renderer.set(v=vertices, f=faces,
                          vc=color, bgcolor=np.ones(3))
        albedo = self.renderer.vc
        # Construct Back Light (on back right corner)
        yrot = np.radians(120)

        self.renderer.vc = LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([-200, -100, -100]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        # Construct Left Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([800, 10, 300]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        #  Construct Right Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([-500, 500, 1000]), yrot),
            vc=albedo,
            light_color=np.array([.7, .7, .7]))

        return self.renderer.r


    def render_vertex_color(self, vertices, faces=None, img=None,
               camera_t=np.zeros([3], dtype=np.float32),
               camera_rot=np.zeros([3], dtype=np.float32),
               camera_center=None,
               use_bg=False,
               bg_color=(0.0, 0.0, 0.0),
               vertex_color=None,
               focal_length=5000,
               disp_text=False,
               gt_keyp=None,
               pred_keyp=None,
               **kwargs):
        if img is not None:
            height, width = img.shape[:2]
        else:
            height, width = self.height, self.width

        if faces is None:
            faces = self.faces

        if camera_center is None:
            camera_center = np.array([width * 0.5,
                                      height * 0.5])

        self.renderer.camera = ProjectPoints(rt=camera_rot,
                                             t=camera_t,
                                             f=focal_length * np.ones(2),
                                             c=camera_center,
                                             k=np.zeros(5))
        dist = np.abs(self.renderer.camera.t.r[2] -
                      np.mean(vertices, axis=0)[2])
        far = dist + 20

        self.renderer.frustum = {'near': 1.0, 'far': far,
                                 'width': width,
                                 'height': height}

        if img is not None:
            if use_bg:
                self.renderer.background_image = img
            else:
                self.renderer.background_image = np.ones_like(
                    img) * np.array(bg_color)

        if vertex_color is None:
            vertex_color = self.colors['light_blue']


        self.renderer.set(v=vertices, f=faces,
                          vc=vertex_color, bgcolor=np.ones(3))
        albedo = self.renderer.vc
        # Construct Back Light (on back right corner)
        yrot = np.radians(120)

        self.renderer.vc = LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([-200, -100, -100]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        # Construct Left Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([800, 10, 300]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        #  Construct Right Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([-500, 500, 1000]), yrot),
            vc=albedo,
            light_color=np.array([.7, .7, .7]))

        return self.renderer.r