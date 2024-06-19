# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Helper functions for constructing camera parameter matrices. Primarily used in visualization and inference scripts.
"""

import math
# from turtle import forward
import cv2
import numpy as np
import torch
import torch.nn as nn

from training.volumetric_rendering import math_utils

class GaussianCameraPoseSampler:
    """
    Samples pitch and yaw from a Gaussian distribution and returns a camera pose.
    Camera is specified as looking at the origin.
    If horizontal and vertical stddev (specified in radians) are zero, gives a
    deterministic camera pose with yaw=horizontal_mean, pitch=vertical_mean.
    The coordinate system is specified with y-up, z-forward, x-left.
    Horizontal mean is the azimuthal angle (rotation around y axis) in radians,
    vertical mean is the polar angle (angle from the y axis) in radians.
    A point along the z-axis has azimuthal_angle=0, polar_angle=pi/2.

    Example:
    For a camera pose looking at the origin with the camera at position [0, 0, 1]:
    cam2world = GaussianCameraPoseSampler.sample(math.pi/2, math.pi/2, radius=1)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
        h = torch.randn((batch_size, 1), device=device) * horizontal_stddev + horizontal_mean
        v = torch.randn((batch_size, 1), device=device) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)

        forward_vectors = math_utils.normalize_vecs(-camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)


class LookAtPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    camera is specified as looking at 'lookat_position', a 3-vector.

    Example:
    For a camera pose looking at the origin with the camera at position [0, 0, 1]:
    cam2world = LookAtPoseSampler.sample(math.pi/2, math.pi/2, torch.tensor([0, 0, 0]), radius=1)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, lookat_position, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
        h = torch.randn((batch_size, 1), device=device) * horizontal_stddev + horizontal_mean
        v = torch.randn((batch_size, 1), device=device) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)

        # forward_vectors = math_utils.normalize_vecs(-camera_origins)
        forward_vectors = math_utils.normalize_vecs(lookat_position - camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)

class UniformCameraPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    pose is sampled from a uniform distribution with range +-[horizontal/vertical]_stddev.

    Example:
    For a batch of random camera poses looking at the origin with yaw sampled from [-pi/2, +pi/2] radians:

    cam2worlds = UniformCameraPoseSampler.sample(math.pi/2, math.pi/2, horizontal_stddev=math.pi/2, radius=1, batch_size=16)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
        h = (torch.rand((batch_size, 1), device=device) * 2 - 1) * horizontal_stddev + horizontal_mean
        v = (torch.rand((batch_size, 1), device=device) * 2 - 1) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)

        forward_vectors = math_utils.normalize_vecs(-camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)    

def create_cam2world_matrix(forward_vector, origin):
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    Works on batches of forward_vectors, origins. Assumes y-axis is up and that there is no camera roll.
    """
    # print('1---------------')
    # print(forward_vector)
    forward_vector = math_utils.normalize_vecs(forward_vector)
    # print('2---------------')
    # print(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=origin.device).expand_as(forward_vector)
    # print('3---------------')
    # print(up_vector)
    right_vector = -math_utils.normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = math_utils.normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))
    # print('4---------------')
    # print(right_vector)
    # print(up_vector)

    rotation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)
    # print('5---------------')
    # print(right_vector)
    # print(up_vector)
    # print(forward_vector)
    # print(rotation_matrix)

    translation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    return cam2world


# def eular_to_cam2world(h, v, r, device='cpu', batch_size=1, radius=1):
    
#     # forward vector
#     theta = h
#     v = v / math.pi
#     phi = torch.arccos(1 - 2*v)

#     camera_origins = torch.zeros((batch_size, 3), device=device)

#     camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
#     camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
#     camera_origins[:, 1:2] = radius*torch.cos(phi)

#     forward_vector = math_utils.normalize_vecs(-camera_origins)

#     # right vector and up vector
#     up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)
#     right_vector = -math_utils.normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
#     up_vector = math_utils.normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))
    

# 从旋转向量转换为欧拉角
def get_euler_angle(rotation_vector):
    # calculate rotation angles
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)
    
    # transformed to quaterniond
    w = math.cos(theta / 2)
    x = math.sin(theta / 2)*rotation_vector[0][0] / theta
    y = math.sin(theta / 2)*rotation_vector[1][0] / theta
    z = math.sin(theta / 2)*rotation_vector[2][0] / theta
    
    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    print('t0:{}, t1:{}'.format(t0, t1))
    pitch = math.atan2(t0, t1)
    
    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)
    
    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)
    
    print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))
    
    # 单位转换：将弧度转换为度
    # Y = int((pitch/math.pi)*180)
    # X = int((yaw/math.pi)*180)
    # Z = int((roll/math.pi)*180)
    
    return pitch, yaw, roll

def cam2world_to_eular_and_origins(cam2world):
    # using left-hand coordinate and Z-X-Y for eular calculation
    assert(cam2world.shape == (4, 4))
    camera_origins = cam2world[:3, 3].reshape(-1, 3)
    camera_origins = camera_origins[0, :]

    rotation_matrix = cam2world[:3, :3]
    if np.abs(rotation_matrix[2, 1] + 1) < 1e-5:
        pitch = np.pi / 2.
        roll = 0
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) # horizontal 
    elif np.abs(rotation_matrix[2, 1] - 1) < 1e-5:
        pitch = -np.pi / 2.
        roll = 0
        yaw = np.arctan2(-rotation_matrix[1, 0], rotation_matrix[0, 0]) # horizontal 
    else:
        pitch = np.arcsin(-rotation_matrix[2, 1]) # vertical
        yaw = np.arctan2(rotation_matrix[2, 0], rotation_matrix[2, 2]) # horizontal 
        roll = np.arctan2(rotation_matrix[0, 1], rotation_matrix[1, 1]) 

    return pitch, yaw, roll, camera_origins

def eular_and_origins_to_cam2world(pitch, yaw, roll, camera_origins):
    # using left-hand coordinate and Z-X-Y for eular calculation
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), np.sin(pitch)],
                   [0, -np.sin(pitch), np.cos(pitch)]], dtype=np.float32)
    Ry = np.array([[np.cos(yaw), 0, -np.sin(yaw)],
                   [0, 1, 0],
                   [np.sin(yaw), 0, np.cos(yaw)]], dtype=np.float32)
    Rz = np.array([[np.cos(roll), np.sin(roll), 0],
                   [-np.sin(roll), np.cos(roll), 0],
                   [0, 0, 1]], dtype=np.float32)          
    rotation_matrix = Rz @ Rx @ Ry
    cam2world = np.eye(4, dtype=np.float32)
    cam2world[:3, :3] = rotation_matrix
    cam2world[:3, 3] = camera_origins
    return cam2world


def eular_and_origins_to_cam2world_torch(angles):
    # using left-hand coordinate and Z-X-Y for eular calculation
    batch_size = angles.shape[0]
    ones = torch.ones([batch_size, 1]).to(angles.device)
    zeros = torch.zeros([batch_size, 1]).to(angles.device)
    pitch, yaw, roll = angles[:, :1], angles[:, 1:2], angles[:, 2:3]
    horizontal, vertical, radius = angles[:, 3:4], angles[:, 4:5], angles[:, 5:6]

    Rx = torch.cat([ones, zeros, zeros,
                zeros, torch.cos(pitch), torch.sin(pitch),
                zeros, -torch.sin(pitch), torch.cos(pitch)], dim=1).reshape([batch_size, 3, 3])
    Ry = torch.cat([torch.cos(yaw), zeros, -torch.sin(yaw),
                zeros, ones, zeros,
                torch.sin(yaw), zeros, torch.cos(yaw)], dim=1).reshape([batch_size, 3, 3])
    Rz = torch.cat([torch.cos(roll), torch.sin(roll), zeros,
                -torch.sin(roll), torch.cos(roll), zeros,
                zeros, zeros, ones], dim=1).reshape([batch_size, 3, 3])

    rotation_matrix = Rz @ Rx @ Ry
    # print(rotation_matrix.shape)

    cam2world = torch.eye(4, device=angles.device).unsqueeze(0).repeat(angles.shape[0], 1, 1)
    cam2world[:, :3, :3] = rotation_matrix

    camera_origins = hvr_to_origins_torch(horizontal, vertical, radius, batch_size=batch_size)
    cam2world[:, :3, 3] = camera_origins


    return cam2world

def hvr_to_origins_torch(horizontal_mean, vertical_mean, radius, batch_size=1):
    v = torch.clamp(vertical_mean, 1e-5, math.pi - 1e-5)
    v = v / math.pi
    phi = torch.arccos(1 - 2*v)
    theta = horizontal_mean

    camera_origins = torch.zeros([batch_size, 3], device=v.device)

    camera_origins[:, 0:1] = radius * torch.sin(phi) * torch.cos(math.pi-theta)
    camera_origins[:, 2:3] = radius * torch.sin(phi) * torch.sin(math.pi-theta)
    camera_origins[:, 1:2] = radius * torch.cos(phi)

    return camera_origins


def origins_to_hvr(camera_origins):
    assert(camera_origins.shape == (3,))
    # camera_origins = torch.from_numpy(camera_origins).to(device)
    x, y, z = camera_origins
    r = np.sqrt(x*x + y*y + z*z)
    phi = np.arccos(y / r)
    theta = np.arctan2(z, -x)
    h = theta
    v = math.pi / 2. * (1 - np.cos(phi))

    return h, v, r

def hvr_to_origins(horizontal_mean, vertical_mean, radius):
    h = horizontal_mean
    v = vertical_mean
    # v = np.clamp(v, 1e-5, math.pi - 1e-5)

    theta = h
    v = v / math.pi
    phi = np.arccos(1 - 2*v)

    camera_origins = np.zeros((3), dtype=np.float32)

    camera_origins[0:1] = radius*np.sin(phi) * np.cos(math.pi-theta)
    camera_origins[2:3] = radius*np.sin(phi) * np.sin(math.pi-theta)
    camera_origins[1:2] = radius*np.cos(phi)

    return camera_origins

def FOV_to_intrinsics(fov_degrees, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """
    
    focal_length = float(1 / (math.tan(fov_degrees * 3.14159 / 360) * 1.414))
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    return intrinsics