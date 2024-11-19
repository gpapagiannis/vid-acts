#!/usr/bin/env python3
import copy
import random
import time

import numpy as np
import rospy
import torchvision.transforms
from sawyer_controller.sawyer_controller import SawyerController
import torch
import matplotlib.pyplot as plt
from sawyer_controller.robotiq_gripper import RobotiqGripper
import signal
import intera_interface
from PIL import Image
from torchvision.transforms import CenterCrop, Resize
from copy import deepcopy
import sawyer_controller.se3_tools as se3
import os
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
    Twist,
)
import mediapy
from sawyer_controller.robotiq_gripper import RobotiqGripper

from std_msgs.msg import UInt8

import cv2
import open3d as o3d
import numpy as np
import time
import scipy.spatial.transform.rotation as R
import numpy as np
# local imports
from util_functions import *
from global_vars import *
# from video_to_gripper_hamer_kpts import *
import matplotlib.animation as animation

os.chdir(os.path.expanduser(
    "/home/georgios/sawyer_workspace/src/in_hand_reorientation/src/hand-action-visual-il/"))  # set default experiments location

rospy.init_node('REPLAY_DEMO')
controller = SawyerController(reset_on_init=False, control_frequency=100, use_real_sense=True)
head = intera_interface.Head()
head.set_pan(-4 * np.pi / 180, active_cancellation=True)
intrinsics_camera = INTRINSICS_REAL_CAMERA
extrinsics_camera = np.load("T_WC_head.npy")
IM_WIDTH = 640
IM_HEIGHT = 480


def get_gripper_transform_in_camera_frame(gripper_scaled_to_hand_pcd, original_hand_pcd, vizualize=False,
                                          return_zero_meaned_gripper=False):
    # Add a world frame
    world_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.100, origin=[0, 0, 0])
    # Show in open3d
    # o3d.visualization.draw_geometries([gripper_scaled_to_hand_pcd, sphere])
    gripper_zero_origin = copy.deepcopy(np.asarray(original_hand_pcd.points))
    # zero mean the z-axis
    gripper_zero_origin[:, 2] = gripper_zero_origin[:, 2] - np.mean(gripper_zero_origin[:, 2])
    # rotate 90 degrees around the x-axis
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    gripper_zero_origin = np.dot(R, gripper_zero_origin.T).T
    gripper_zero_origin_o3d = o3d.geometry.PointCloud()
    gripper_zero_origin_o3d.points = o3d.utility.Vector3dVector(gripper_zero_origin)
    p = np.asarray(gripper_zero_origin_o3d.points)
    q = np.asarray(gripper_scaled_to_hand_pcd.points)
    T = find_scaled_transformation(p, q, use_scale=False)
    gripper_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.100, origin=[0, 0, 0])
    gripper_coord.transform(T)

    # ttest = np.eye(4)
    # ttest[2, 3] = 0.05
    # gripper_zero_origin_o3d.transform(T @ ttest)
    # ttest_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.100, origin=[0, 0, 0])
    # ttest_coord.transform(T @ ttest)

    # gripper_zero_origin_o3d.paint_uniform_color([1, 0, 0])
    if vizualize:
        o3d.visualization.draw_geometries(
            [gripper_scaled_to_hand_pcd, gripper_zero_origin_o3d, gripper_coord, world_coord])
    if return_zero_meaned_gripper:
        return T, gripper_zero_origin_o3d
    return T


def pose_to_ros_msg(pose):
    # T = se3.pose2posevec(pose)
    T = pose
    pose_message = Pose()
    # T = T.tolist()
    pose_message.position.x = T[0]
    pose_message.position.y = T[1]
    pose_message.position.z = T[2]
    pose_message.orientation.x = T[3]
    pose_message.orientation.y = T[4]
    pose_message.orientation.z = T[5]
    pose_message.orientation.w = T[6]
    return pose_message


def align_hand_to_gripper_press(gripper_pcd,
                                actions,
                                vizualize=False,
                                bias_transformation=np.eye(4)):
    gripper_pcd_original_mesh = copy.deepcopy(gripper_pcd)
    dense_pcd_kpts = {"index_front": 517980,
                      "index_middle": 231197,
                      "index_bottom": 335530,
                      "thumb_front": 248802,
                      "thumb_middle": 71859,
                      "thumb_bottom": 523328,
                      "wrist": 246448}

    gripper_fingers = np.array([gripper_pcd.points[dense_pcd_kpts["index_front"]],
                                gripper_pcd.points[dense_pcd_kpts["index_middle"]],
                                gripper_pcd.points[dense_pcd_kpts["index_bottom"]],
                                gripper_pcd.points[dense_pcd_kpts["thumb_front"]],
                                gripper_pcd.points[dense_pcd_kpts["thumb_middle"]],
                                gripper_pcd.points[dense_pcd_kpts["thumb_bottom"]],
                                gripper_pcd.points[dense_pcd_kpts["wrist"]]])

    gripper_fingers[0] = gripper_fingers[0] - (gripper_fingers[0] - gripper_fingers[3]) / 2
    gripper_fingers[4] = gripper_fingers[1] - (gripper_fingers[1] - gripper_fingers[4]) / 2
    gripper_fingers[-1] = gripper_fingers[2] - (gripper_fingers[2] - gripper_fingers[5]) / 2
    key_fingers_points = actions

    kpt_o3d_sphere = []
    count = 0

    for kpt in key_fingers_points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=.003)
        sphere.compute_vertex_normals()
        count += 1
        if count % 3 == 0:
            red, green, blue = 1, 0, 1
        elif count % 3 == 1:
            red, green, blue = 0, 1, 0
        else:
            red, green, blue = 1, 0.5, 0
        sphere.paint_uniform_color([red, green, blue])
        sphere.translate(kpt)
        kpt_o3d_sphere.append(sphere)

    gripper_fingers_4pts = np.array([gripper_fingers[0], gripper_fingers[4], gripper_fingers[-1]])
    gripper_fingers = np.array(gripper_fingers_4pts)
    gripper_fingers_o3d = []
    count = 0
    # Create vizualizer to sequentilaly add spheres to the gripper fingers
    for kpt in gripper_fingers:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=.003)
        sphere.compute_vertex_normals()
        count += 1
        if count % 3 == 0:
            red, green, blue = 1, 0, 1  # color: purple
        elif count % 3 == 1:
            red, green, blue = 0, 1, 0
        else:
            red, green, blue = 1, 0.5, 0
        sphere.paint_uniform_color([red, green, blue])
        sphere.translate(kpt)
        gripper_fingers_o3d.append(sphere)

    # o3d.visualization.draw_geometries([gripper_pcd] + gripper_fingers_o3d )

    T = find_scaled_transformation(gripper_fingers, key_fingers_points, use_scale=False)
    # transform the gripper_fingers_o3d to the hand frame
    for sphere in gripper_fingers_o3d:
        sphere.transform(T)
    gripper_pcd.transform(T)

    if bias_transformation is None:
        bias_transformation = np.eye(4)
    # apply bias transformation in the gripper frame
    gripper_pose, gripper_zero_mean = get_gripper_transform_in_camera_frame(gripper_pcd,
                                                                            gripper_pcd_original_mesh,
                                                                            return_zero_meaned_gripper=True,
                                                                            vizualize=vizualize)
    gripper_pose = gripper_pose @ bias_transformation
    gripper_zero_mean.transform(gripper_pose)
    gripper_pcd = copy.deepcopy(gripper_zero_mean)

    if vizualize:
        # o3d.visualization.draw_geometries([pcd_image, gripper_scaled_to_hand_pcd])

        gripper_frame_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.100, origin=[0, 0, 0])
        gripper_frame_coord.transform(gripper_pose)
        gripper_pcd.paint_uniform_color([0.0, 0.0, 0.0])
        o3d.visualization.draw_geometries([gripper_pcd] + kpt_o3d_sphere)
        # o3d.visualization.draw_geometries([hand_pcd_as_o3d, gripper_scaled_to_hand_pcd] + gripper_fingers_o3d + kpt_o3d_sphere + [line_o3d_rotation_axis])

    return gripper_pcd, gripper_pose


def actions_to_gripper_poses(gripper_scaled_to_hand_pcd,
                             actions,
                             gripper_pcd_dense_mesh,
                             vizualize=False,
                             bias_transformation=np.eye(4)):
    dense_pcd_kpts = {"index_front": 517980, "thumb_front": 248802, "wrist": 246448}
    gripper_fingers = np.array([gripper_scaled_to_hand_pcd.points[dense_pcd_kpts["index_front"]],
                                gripper_scaled_to_hand_pcd.points[dense_pcd_kpts["index_front"]],
                                gripper_scaled_to_hand_pcd.points[dense_pcd_kpts["index_front"]],
                                gripper_scaled_to_hand_pcd.points[dense_pcd_kpts["thumb_front"]],
                                gripper_scaled_to_hand_pcd.points[dense_pcd_kpts["thumb_front"]],
                                gripper_scaled_to_hand_pcd.points[dense_pcd_kpts["thumb_front"]],
                                gripper_scaled_to_hand_pcd.points[dense_pcd_kpts["wrist"]]])

    kpt_o3d_sphere = []
    count = 0
    key_fingers_points_4pts = actions
    key_fingers_points = np.array(key_fingers_points_4pts)
    gripper_open = 1
    finger_opening = np.linalg.norm(key_fingers_points[0] - key_fingers_points[1])
    if finger_opening < 0.05:
        gripper_open = 0
        print("Close gripper action detected")
    unit_vec_difference = (key_fingers_points[1] - key_fingers_points[0]) / np.linalg.norm(
        key_fingers_points[1] - key_fingers_points[0])
    distance_gripper_fingers = np.linalg.norm(gripper_fingers[0] - gripper_fingers[4])
    distance_key_fingers = np.linalg.norm(key_fingers_points[0] - key_fingers_points[1])
    difference_half = np.abs(distance_gripper_fingers - distance_key_fingers) / 2
    pt1 = key_fingers_points[0] - unit_vec_difference * difference_half
    pt2 = key_fingers_points[1] + unit_vec_difference * difference_half
    middle_finger_point = key_fingers_points[0] + unit_vec_difference * distance_key_fingers / 2
    unit_difference_between_middle_finger_point_and_key_fingers_last = (middle_finger_point - key_fingers_points[
        -1]) / np.linalg.norm(middle_finger_point - key_fingers_points[-1])

    distance_pt1_pt2 = np.linalg.norm(pt1 - pt2)
    print(f"Distance between pt1 and pt2: {distance_pt1_pt2}")
    print(f"Distance between gripper fingers: {distance_gripper_fingers}")
    key_fingers_points = np.array([pt1, pt2, key_fingers_points[-1]])
    # key_fingers_points = np.array([pt1, pt2])

    for kpt in key_fingers_points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=.003)
        sphere.compute_vertex_normals()
        count += 1
        if count % 3 == 0:
            red, green, blue = 1, 0, 1
        elif count % 3 == 1:
            red, green, blue = 0, 1, 0
        else:
            red, green, blue = 1, 0.5, 0
        sphere.paint_uniform_color([red, green, blue])
        sphere.translate(kpt)
        kpt_o3d_sphere.append(sphere)

    # add middle finger point
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=.003)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([1, .5, .777])
    sphere.translate(middle_finger_point)
    kpt_o3d_sphere.append(sphere)

    # gripper_fingers_4pts = np.array([gripper_fingers[1], gripper_fingers[2], gripper_fingers[4], gripper_fingers[5]])
    gripper_fingers_4pts = np.array([gripper_fingers[0], gripper_fingers[4], gripper_fingers[-1]])
    # gripper_fingers_4pts = np.array([gripper_fingers[0], gripper_fingers[4]])

    gripper_fingers = np.array(gripper_fingers_4pts)
    gripper_fingers_o3d = []
    count = 0
    # Create vizualizer to sequentilaly add spheres to the gripper fingers
    for kpt in gripper_fingers:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=.003)
        sphere.compute_vertex_normals()
        count += 1
        if count % 3 == 0:
            red, green, blue = 1, 0, 1  # color: purple
        elif count % 3 == 1:
            red, green, blue = 0, 1, 0
        else:
            red, green, blue = 1, 0.5, 0
        sphere.paint_uniform_color([red, green, blue])
        sphere.translate(kpt)
        gripper_fingers_o3d.append(sphere)

    T = find_scaled_transformation(gripper_fingers[:2], key_fingers_points[:2], use_scale=False)
    # transform the gripper_fingers_o3d to the hand frame
    for sphere in gripper_fingers_o3d:
        sphere.transform(T)

    gripper_pcd_before_transform = copy.deepcopy(gripper_scaled_to_hand_pcd)
    gripper_scaled_to_hand_pcd.transform(T)
    # Assume R is the rotation matrix you've computed and t is the translation
    # Transform z2
    R = T[:3, :3]
    t = T[:3, 3]
    z1 = key_fingers_points[-1]
    x1 = key_fingers_points[0]
    y1 = key_fingers_points[1]
    z2 = gripper_fingers[-1]
    x2 = gripper_fingers[0]
    y2 = gripper_fingers[1]

    z2_transformed = R @ z2 + t
    # Compute rotation axis (using x2 and y2 after transformation)
    x2_transformed = R @ x2 + t
    y2_transformed = R @ y2 + t
    rotation_axis = y2_transformed - x2_transformed  # np.cross(x2_transformed - y2_transformed, z2_transformed - y2_transformed)

    # find theta that bring z2 as closest as possible to z2 while keeping the rotation axis the same
    distance = 10e10
    rotation_theta = None
    for theta in np.linspace(0, 2 * np.pi, 1000):
        R_additional = rotation_matrix(rotation_axis, theta)
        z2_final = (z2_transformed - (y2_transformed + x2_transformed) / 2) @ R_additional.T + (
                y2_transformed + x2_transformed) / 2
        distance_temp = np.linalg.norm(z2_final - z1)
        if distance_temp < distance:
            distance = distance_temp
            rotation_theta = theta

    # Apply rotation about the axis
    R_additional = rotation_matrix(rotation_axis, rotation_theta)
    z2_final = (z2_transformed - (y2_transformed + x2_transformed) / 2) @ R_additional.T + (
            y2_transformed + x2_transformed) / 2

    T2 = np.eye(4)
    T2[:3, :3] = R_additional
    gripper_scaled_to_hand_pcd_points = np.asarray(gripper_scaled_to_hand_pcd.points)
    gripper_scaled_to_hand_pcd_points = (gripper_scaled_to_hand_pcd_points - (
            y2_transformed + x2_transformed) / 2) @ R_additional.T + (y2_transformed + x2_transformed) / 2
    gripper_scaled_to_hand_pcd.points = o3d.utility.Vector3dVector(gripper_scaled_to_hand_pcd_points)
    # gripper_scaled_to_hand_pcd.transform(T2)
    # gripper_aligned_to_hand_pcd_as_o3d.paint_uniform_color([.1, 1, 1])

    # z2_final = gripper_scaled_to_hand_pcd.points[dense_pcd_kpts["wrist"]]
    # add z2_final to sphere
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=.003)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([0, 0, 0])
    sphere.translate(z2_final)
    kpt_o3d_sphere.append(sphere)

    if bias_transformation is None:
        bias_transformation = np.eye(4)
        # bias_transformation[2, 3] = 0.02
    # apply bias transformation in the gripper frame
    gripper_pose, gripper_zero_mean = get_gripper_transform_in_camera_frame(gripper_scaled_to_hand_pcd,
                                                                            gripper_pcd_dense_mesh,
                                                                            return_zero_meaned_gripper=True,
                                                                            vizualize=vizualize)
    gripper_pose = gripper_pose @ bias_transformation
    gripper_zero_mean.transform(gripper_pose)
    gripper_scaled_to_hand_pcd = copy.deepcopy(gripper_zero_mean)

    if vizualize:
        # o3d.visualization.draw_geometries([pcd_image, gripper_scaled_to_hand_pcd])
        line_o3d = o3d.geometry.LineSet()
        line_o3d.points = o3d.utility.Vector3dVector(
            [key_fingers_points[0], key_fingers_points[0] + unit_vec_difference * 3])
        line_o3d.lines = o3d.utility.Vector2iVector([[0, 1]])

        line_o3d_2 = o3d.geometry.LineSet()
        line_o3d_2.points = o3d.utility.Vector3dVector(
            [key_fingers_points[0], key_fingers_points[0] - unit_vec_difference * 3])
        line_o3d_2.lines = o3d.utility.Vector2iVector([[0, 1]])

        line_o3d_3 = o3d.geometry.LineSet()
        line_o3d_3.points = o3d.utility.Vector3dVector([middle_finger_point, key_fingers_points[
            1] + unit_difference_between_middle_finger_point_and_key_fingers_last * 3])
        line_o3d_3.lines = o3d.utility.Vector2iVector([[0, 1]])

        line_o3d_4 = o3d.geometry.LineSet()
        line_o3d_4.points = o3d.utility.Vector3dVector([middle_finger_point, key_fingers_points[
            1] - unit_difference_between_middle_finger_point_and_key_fingers_last * 3])
        line_o3d_4.lines = o3d.utility.Vector2iVector([[0, 1]])

        line_o3d_rotation_axis = o3d.geometry.LineSet()
        line_o3d_rotation_axis.points = o3d.utility.Vector3dVector(
            [x2_transformed, x2_transformed + 10 * rotation_axis])
        line_o3d_rotation_axis.lines = o3d.utility.Vector2iVector([[0, 1]])

        line_o3d_rotation_axis_2 = o3d.geometry.LineSet()
        line_o3d_rotation_axis_2.points = o3d.utility.Vector3dVector(
            [y2_transformed, y2_transformed - 10 * rotation_axis])
        line_o3d_rotation_axis_2.lines = o3d.utility.Vector2iVector([[0, 1]])

        gripper_frame_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.100, origin=[0, 0, 0])
        gripper_frame_coord.transform(gripper_pose)
        gripper_scaled_to_hand_pcd.paint_uniform_color([0.0, 0.0, 0.0])
        o3d.visualization.draw_geometries(
            [gripper_scaled_to_hand_pcd, gripper_frame_coord] + gripper_fingers_o3d + kpt_o3d_sphere + [line_o3d,
                                                                                                        line_o3d_2,
                                                                                                        line_o3d_3,
                                                                                                        line_o3d_4,
                                                                                                        line_o3d_rotation_axis,
                                                                                                        line_o3d_rotation_axis_2])
        # o3d.visualization.draw_geometries([hand_pcd_as_o3d, gripper_scaled_to_hand_pcd] + gripper_fingers_o3d + kpt_o3d_sphere + [line_o3d_rotation_axis])

    return gripper_scaled_to_hand_pcd, gripper_pose, gripper_open


def extract_actions(folder="assets/test_inference", vizualize=False, vizualize_3d=False, frame_step=15):
    rgb_im = np.load(f"{folder}/rgb_test.npy")
    depth_im = np.load(f"{folder}/depth_test.npy") / 1000
    gripper_actions_loaded = np.load(f"{folder}/gripper.npy")
    print("gripper actions loaded: ", np.asarray(gripper_actions_loaded).shape)
    print(gripper_actions_loaded)
    # depth_im to point cloud
    point_cloud_camera = depth_to_point_cloud(depth_im,
                                              INTRINSICS_REAL_CAMERA[0, 0],
                                              INTRINSICS_REAL_CAMERA[1, 1],
                                              INTRINSICS_REAL_CAMERA[0, 2],
                                              INTRINSICS_REAL_CAMERA[
                                                  1, 2])  # Camera intrinsics, depth at real scale, although the shape is not accurate
    point_cloud_camera = point_cloud_camera.reshape(-1, 3)
    pcd_image = o3d.geometry.PointCloud()
    pcd_image.points = o3d.utility.Vector3dVector(point_cloud_camera)
    pcd_image.colors = o3d.utility.Vector3dVector(rgb_im.reshape(-1, 3) / 255.0)

    predicted_actions = np.load(f"{folder}/predicted_actions.npy")
    if vizualize:
        plt.imshow(rgb_im)
        plt.show()

    gripper_poses_actions = []
    # NEURAL NETWORK ACTIONS TO GRIPPER POSES
    gripper_pcd = np.load('gripper_point_cloud_dense.npy')
    gripper_pcd = gripper_pcd / 1000  # scale down by 1000
    gripper_pcd_as_o3d = o3d.geometry.PointCloud()
    gripper_pcd_as_o3d.points = o3d.utility.Vector3dVector(gripper_pcd)
    gripper_pcds = []
    gripper_actions = []
    gripper_pose_frame = []
    gripper_opening = []
    press_task = False
    counter = 0
    print("predicted actions shape: ", predicted_actions.shape)
    for act in predicted_actions:

        try:
            if not press_task:
                actions_test = np.array([act[0],
                                         act[1],
                                         (act[2] + act[3]) / 2, ])

                gripper, gripper_pose, gripper_open = actions_to_gripper_poses(copy.deepcopy(gripper_pcd_as_o3d),
                                                                               actions_test,
                                                                               copy.deepcopy(gripper_pcd_as_o3d),
                                                                               vizualize=False)
                if gripper_actions_loaded[counter] == 0:
                    gripper_open = 1

                else:
                    gripper_open = 0

                if counter == 0:
                    gripper_open = 1
            else:
                actions_test = np.array([act[0],
                                         act[1],
                                         act[2], ])

                gripper, gripper_pose = align_hand_to_gripper_press(copy.deepcopy(gripper_pcd_as_o3d),
                                                                    actions_test,
                                                                    vizualize=False)

                gripper_open = 0


            gripper_opening.append(gripper_open); counter+=1
            gripper_actions.append(gripper_pose)
            gripper_pcds.append(gripper)
            # create frame for each gripper pose
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.100, origin=[0, 0, 0])
            coord_frame.transform(gripper_pose)
            gripper_pose_frame.append(coord_frame)

            if vizualize_3d:
                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.100, origin=[0, 0, 0])
                coord_frame.transform(gripper_pose)
                o3d.visualization.draw_geometries([pcd_image, gripper, coord_frame])
        except Exception as e:
            print(e)
            continue
    video_imgs = []
    for i in range(0, len(gripper_actions)):
        gripper_pcd_np = np.asarray(gripper_pcds[i].points)
        gripper_pcd_depth_im = point_cloud_to_depth_image(gripper_pcd_np,
                                                          INTRINSICS_REAL_CAMERA[0, 0],
                                                          INTRINSICS_REAL_CAMERA[1, 1],
                                                          INTRINSICS_REAL_CAMERA[0, 2],
                                                          INTRINSICS_REAL_CAMERA[1, 2],
                                                          width=IM_WIDTH,
                                                          height=IM_HEIGHT)
        gripper_pcd_depth_im[gripper_pcd_depth_im > 0] = 1
        im = .5 * rgb_im / 255 + .5 * gripper_pcd_depth_im[:, :, np.newaxis]
        video_imgs.append(im[:, :, ::-1])

    fig = plt.figure()
    ims = []
    for i in range(len(video_imgs)):
        video_img = video_imgs[i][:, :, ::-1]
        im = plt.imshow(video_img, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=300, blit=True, repeat_delay=1000)
    ani.save(f'{folder}/expected_gripper_trajectory.mp4', writer='ffmpeg', fps=3)

    print("Interpolate trajectories...")
    interpolated_gripper_poses = interpolate_pose_sequence(gripper_actions, frame_step)

    # repeat each element in gripper_opening by frame_step times
    gripper_opening = np.repeat(gripper_opening, frame_step)
    print(gripper_opening)
    print(gripper_opening.shape, interpolated_gripper_poses.shape)

    return interpolated_gripper_poses, gripper_pcds, gripper_opening, gripper_pose_frame


def execute_actions(actions=None, gripper_opening=None, gripper_pcds=None, gripper_pose_frame=None, vizualize=False,
                    folder=None):
    assert actions is not None
    gripper = RobotiqGripper(is_open=True)

    interpolated_gripper_poses = actions
    # #TODO: Interpolate actions
    print("Number of actions: ", interpolated_gripper_poses.shape[0])
    video_ims = []
    gripper_poses = []
    head.set_pan(-4 * np.pi / 180, active_cancellation=True)

    transf_down = np.eye(4)
    transf_down[2, 3] = -0.05
    gripper_pose = interpolated_gripper_poses[0]
    gripper_pose = gripper_pose @ transf_down
    gripper_pose = extrinsics_camera @ gripper_pose

    gripper_posevec = np.zeros(7)
    gripper_posevec[:3] = gripper_pose[:3, 3]
    rot = R.from_matrix(gripper_pose[:3, :3])
    gripper_posevec[3:] = rot.as_quat()
    gripper_pose = se3.posevec2pose(gripper_posevec)
    # controller.go_to_pose_in_base(gripper_pose, threshold_pos=2e3)

    input("start smooth controller")
    "Send message to start visual servoing"
    target_pose_publisher = rospy.Publisher("vs_target_pose", Pose, queue_size=10)
    start_visual_servo_publisher = rospy.Publisher("start_vs", UInt8, queue_size=10)
    start_signal = UInt8()
    start_signal.data = 0
    start_visual_servo_publisher.publish(start_signal)
    init_vs_pose = controller.get_eef_posevec()
    init_vs_pose = pose_to_ros_msg(init_vs_pose)
    start_signal = UInt8()
    start_signal.data = 1
    target_pose_publisher.publish(init_vs_pose)
    start_visual_servo_publisher.publish(start_signal)

    # def get_live_pcd(vizualize=False):
    #     # capture depth
    #     depth_img = controller.camera.get_aligned_depth()
    #     rgb_img = controller.camera.get_rgb()
    #     depth_img = depth_img / 1000
    #     # depth img to point cloud
    #     # depth_img = depth_img.reshape(-1)
    #     depth_pcd = depth_to_point_cloud(depth_img, intrinsics_camera[0, 0], intrinsics_camera[1, 1], intrinsics_camera[0, 2],
    #                                      intrinsics_camera[1, 2])
    #     depth_pcd = depth_pcd.reshape(-1, 3)
    #     depth_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(depth_pcd))
    #     depth_pcd.colors = o3d.utility.Vector3dVector(rgb_img.reshape(-1, 3) / 255)
    #     if vizualize:
    #         o3d.visualization.draw_geometries([depth_pcd])
    #     return depth_pcd
    #
    # live_pcd = get_live_pcd(vizualize=True)
    # transform points in live_pcd with camera_extrinsics by first extracting the points a numpyu asrray
    # and then transforming them with the extrinsics matrix
    # live_pcd_color = np.asarray(live_pcd.colors)
    # live_pcd_np = np.asarray(live_pcd.points)
    # live_pcd_np = extrinsics_camera[:3, :3] @ live_pcd_np.T + extrinsics_camera[:3, 3][:, np.newaxis]
    # live_pcd_np = live_pcd_np.T
    # live_pcd.points = o3d.utility.Vector3dVector(live_pcd_np)
    # live_pcd.colors = o3d.utility.Vector3dVector(live_pcd_color)
    # live_pcd.transform(extrinsics_camera)
    # curren_eef_pose = controller.get_eef_pose()
    # current_eef_pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.100)
    # current_eef_pose_frame.transform(curren_eef_pose)

    # vizualize live pcd
    # o3d.visualization.draw_geometries([live_pcd, current_eef_pose_frame])
    # o3d.io.write_point_cloud(f'{task_folder}/scene_files/live_image_pcd_0.ply', depth_pcd)
    gripper_state = not gripper_opening[0]
    for j in range(0, interpolated_gripper_poses.shape[0]):

        transf_down = np.eye(4)
        transf_down[2, 3] = -0.065
        # transf_down[1, 3] = .05
        gripper_pose = interpolated_gripper_poses[j]
        gripper_pose = gripper_pose @ transf_down
        gripper_pose = extrinsics_camera @ gripper_pose
        gripper_poses.append(gripper_pose)
        gripper_posevec = np.zeros(7)
        gripper_posevec[:3] = gripper_pose[:3, 3]
        rot = R.from_matrix(gripper_pose[:3, :3])
        gripper_posevec[3:] = rot.as_quat()
        pose_message = pose_to_ros_msg(gripper_posevec)
        target_pose_publisher.publish(pose_message)
        start_visual_servo_publisher.publish(start_signal)
        time.sleep(0.25)
        head.set_pan(-4 * np.pi / 180, active_cancellation=True)
        print(gripper_opening[j])
        # Sorry no time now
        if gripper_opening[j] == 0:
            if gripper_state == 1:
                print("CLOSE GRIPPER")
                gripper.close()
                gripper_state = 0
        else:
            if gripper_state == 0:
                print("OPEN GRIPPER")
                gripper.open()
                gripper_state = 1
        # gripper_pose = se3.posevec2pose(gripper_posevec)"""

        # controller.go_to_pose_in_base(gripper_pose, threshold_pos=2e3)
        # head.set_pan(-4 * np.pi / 180, active_cancellation=True)
        video_ims.append(controller.camera.get_rgb())

        if j == 0:
            for _ in range(100):
                target_pose_publisher.publish(pose_message)
                start_visual_servo_publisher.publish(start_signal)
                time.sleep(0.05)
                video_ims.append(controller.camera.get_rgb())

        # live_pcd = get_live_pcd()
        # eef_pose = controller.get_eef_pose()
        # eef_pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.100)
        # eef_pose_frame.transform(eef_pose)
        # gpcd = gripper_pcds[j].transform(extrinsics_camera)#
        # live_pcd.transform(extrinsics_camera)
        # gposeframe = gripper_pose_frame[j]
        # gposeframe.transform(extrinsics_camera)
        # o3d.visualization.draw_geometries([live_pcd, gpcd, gposeframe, eef_pose_frame])
    "Send message to STOP visual servoing"
    start_signal = UInt8()
    start_signal.data = 0
    start_visual_servo_publisher.publish(start_signal)
    time.sleep(.1)

    for _ in range(20):
        time.sleep(0.05)
        video_ims.append(controller.camera.get_rgb())

    np.save(f"{folder}/video_imgs.npy", video_ims)
    np.save(f"{folder}/world_frame_gripper_poses.npy", gripper_poses)

    # generate the video with mediapy

    mediapy.write_video(f"{folder}/live_execution.mp4", video_ims, fps=20)
    #
    # fig = plt.figure()
    # ims = []
    # for i in range(len(video_ims)):
    #     im = plt.imshow(video_ims[i], animated=True)
    #     ims.append([im])
    #
    # ani = animation.ArtistAnimation(fig, ims, interval=300, blit=True, repeat_delay=1000)
    # ani.save(f'{folder}/live_execution.mp4', writer='ffmpeg', fps=20)


def execute_actions_world_frame(actions=None, gripper_opening=None, gripper_pcds=None, gripper_pose_frame=None, vizualize=False,
                    folder=None):
    assert actions is not None
    gripper = RobotiqGripper(is_open=True)

    interpolated_gripper_poses = actions
    # #TODO: Interpolate actions
    print("Number of actions: ", interpolated_gripper_poses.shape[0])
    video_ims = []
    gripper_poses = []
    head.set_pan(-4 * np.pi / 180, active_cancellation=True)

    transf_down = np.eye(4)
    transf_down[2, 3] = -0.05
    gripper_pose = interpolated_gripper_poses[0]
    gripper_pose = gripper_pose @ transf_down
    gripper_pose = extrinsics_camera @ gripper_pose

    gripper_posevec = np.zeros(7)
    gripper_posevec[:3] = gripper_pose[:3, 3]
    rot = R.from_matrix(gripper_pose[:3, :3])
    gripper_posevec[3:] = rot.as_quat()
    gripper_pose = se3.posevec2pose(gripper_posevec)
    # controller.go_to_pose_in_base(gripper_pose, threshold_pos=2e3)

    input("start smooth controller")
    "Send message to start visual servoing"
    target_pose_publisher = rospy.Publisher("vs_target_pose", Pose, queue_size=10)
    start_visual_servo_publisher = rospy.Publisher("start_vs", UInt8, queue_size=10)
    start_signal = UInt8()
    start_signal.data = 0
    start_visual_servo_publisher.publish(start_signal)
    init_vs_pose = controller.get_eef_posevec()
    init_vs_pose = pose_to_ros_msg(init_vs_pose)
    start_signal = UInt8()
    start_signal.data = 1
    target_pose_publisher.publish(init_vs_pose)
    start_visual_servo_publisher.publish(start_signal)

    # def get_live_pcd(vizualize=False):
    #     # capture depth
    #     depth_img = controller.camera.get_aligned_depth()
    #     rgb_img = controller.camera.get_rgb()
    #     depth_img = depth_img / 1000
    #     # depth img to point cloud
    #     # depth_img = depth_img.reshape(-1)
    #     depth_pcd = depth_to_point_cloud(depth_img, intrinsics_camera[0, 0], intrinsics_camera[1, 1], intrinsics_camera[0, 2],
    #                                      intrinsics_camera[1, 2])
    #     depth_pcd = depth_pcd.reshape(-1, 3)
    #     depth_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(depth_pcd))
    #     depth_pcd.colors = o3d.utility.Vector3dVector(rgb_img.reshape(-1, 3) / 255)
    #     if vizualize:
    #         o3d.visualization.draw_geometries([depth_pcd])
    #     return depth_pcd
    #
    # live_pcd = get_live_pcd(vizualize=True)
    # transform points in live_pcd with camera_extrinsics by first extracting the points a numpyu asrray
    # and then transforming them with the extrinsics matrix
    # live_pcd_color = np.asarray(live_pcd.colors)
    # live_pcd_np = np.asarray(live_pcd.points)
    # live_pcd_np = extrinsics_camera[:3, :3] @ live_pcd_np.T + extrinsics_camera[:3, 3][:, np.newaxis]
    # live_pcd_np = live_pcd_np.T
    # live_pcd.points = o3d.utility.Vector3dVector(live_pcd_np)
    # live_pcd.colors = o3d.utility.Vector3dVector(live_pcd_color)
    # live_pcd.transform(extrinsics_camera)
    # curren_eef_pose = controller.get_eef_pose()
    # current_eef_pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.100)
    # current_eef_pose_frame.transform(curren_eef_pose)

    # vizualize live pcd
    # o3d.visualization.draw_geometries([live_pcd, current_eef_pose_frame])
    # o3d.io.write_point_cloud(f'{task_folder}/scene_files/live_image_pcd_0.ply', depth_pcd)
    gripper_state = not gripper_opening[0]
    for j in range(0, interpolated_gripper_poses.shape[0]):

        transf_down = np.eye(4)
        transf_down[2, 3] = -0.065
        # transf_down[1, 3] = .05
        gripper_pose = interpolated_gripper_poses[j]
        # gripper_pose = gripper_pose @ transf_down
        # gripper_pose = extrinsics_camera @ gripper_pose
        gripper_poses.append(gripper_pose)
        gripper_posevec = np.zeros(7)
        gripper_posevec[:3] = gripper_pose[:3, 3]
        rot = R.from_matrix(gripper_pose[:3, :3])
        gripper_posevec[3:] = rot.as_quat()
        pose_message = pose_to_ros_msg(gripper_posevec)
        target_pose_publisher.publish(pose_message)
        start_visual_servo_publisher.publish(start_signal)
        time.sleep(0.25)
        head.set_pan(-4 * np.pi / 180, active_cancellation=True)
        print(gripper_opening[j])
        # Sorry no time now

        # gripper_pose = se3.posevec2pose(gripper_posevec)"""

        # controller.go_to_pose_in_base(gripper_pose, threshold_pos=2e3)
        # head.set_pan(-4 * np.pi / 180, active_cancellation=True)
        video_ims.append(controller.camera.get_rgb())

        if j == 0:
            for _ in range(100):
                target_pose_publisher.publish(pose_message)
                start_visual_servo_publisher.publish(start_signal)
                time.sleep(0.05)
                video_ims.append(controller.camera.get_rgb())

        # live_pcd = get_live_pcd()
        # eef_pose = controller.get_eef_pose()
        # eef_pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.100)
        # eef_pose_frame.transform(eef_pose)
        # gpcd = gripper_pcds[j].transform(extrinsics_camera)#
        # live_pcd.transform(extrinsics_camera)
        # gposeframe = gripper_pose_frame[j]
        # gposeframe.transform(extrinsics_camera)
        # o3d.visualization.draw_geometries([live_pcd, gpcd, gposeframe, eef_pose_frame])
    "Send message to STOP visual servoing"
    start_signal = UInt8()
    start_signal.data = 0
    start_visual_servo_publisher.publish(start_signal)
    time.sleep(.1)

    for _ in range(20):
        time.sleep(0.05)
        video_ims.append(controller.camera.get_rgb())

    np.save(f"{folder}/video_imgs.npy", video_ims)
    np.save(f"{folder}/world_frame_gripper_poses.npy", gripper_poses)

    # generate the video with mediapy

    mediapy.write_video(f"{folder}/live_execution.mp4", video_ims, fps=20)
    #
    # fig = plt.figure()
    # ims = []
    # for i in range(len(video_ims)):
    #     im = plt.imshow(video_ims[i], animated=True)
    #     ims.append([im])
    #
    # ani = animation.ArtistAnimation(fig, ims, interval=300, blit=True, repeat_delay=1000)
    # ani.save(f'{folder}/live_execution.mp4', writer='ffmpeg', fps=20)

def replay_demo(vizualize=False):
    interpolated_gripper_poses = np.load(f"{task_folder}/scene_files/interpolated_gripper_poses.npy")
    print("Number of actions: ", interpolated_gripper_poses.shape[0])
    initial_gripper_pose = copy.deepcopy(interpolated_gripper_poses[0])
    initial_gripper_pose_inv = np.linalg.inv(initial_gripper_pose)
    initial_gripper_pose_pcd = o3d.io.read_point_cloud(f'{task_folder}/scene_files/gripper_pcd_0.ply')
    initial_scene_pcd = o3d.io.read_point_cloud(f'{task_folder}/scene_files/live_image_pcd_0.ply')

    transf_down = np.eye(4)
    transf_down[2, 3] = -0.05
    gripper_pose = interpolated_gripper_poses[0]
    gripper_pose = gripper_pose @ transf_down
    gripper_pose = extrinsics_camera @ gripper_pose

    gripper_posevec = np.zeros(7)
    gripper_posevec[:3] = gripper_pose[:3, 3]
    rot = R.from_matrix(gripper_pose[:3, :3])
    gripper_posevec[3:] = rot.as_quat()
    gripper_pose = se3.posevec2pose(gripper_posevec)
    controller.go_to_pose_in_base(gripper_pose, threshold_pos=2e3)

    input("start smooth controller")
    "Send message to start visual servoing"
    target_pose_publisher = rospy.Publisher("vs_target_pose", Pose, queue_size=10)
    start_visual_servo_publisher = rospy.Publisher("start_vs", UInt8, queue_size=10)
    start_signal = UInt8()
    start_signal.data = 0
    start_visual_servo_publisher.publish(start_signal)
    init_vs_pose = controller.get_eef_posevec()
    init_vs_pose = pose_to_ros_msg(init_vs_pose)
    start_signal = UInt8()
    start_signal.data = 1
    target_pose_publisher.publish(init_vs_pose)
    start_visual_servo_publisher.publish(start_signal)

    for j in range(0, interpolated_gripper_poses.shape[0]):
        transf_down = np.eye(4)
        transf_down[2, 3] = -0.05
        # transf_down[1, 3] = .05
        gripper_pose = interpolated_gripper_poses[j]
        gripper_pose = gripper_pose @ transf_down
        gripper_pose = extrinsics_camera @ gripper_pose

        gripper_posevec = np.zeros(7)
        gripper_posevec[:3] = gripper_pose[:3, 3]
        rot = R.from_matrix(gripper_pose[:3, :3])
        gripper_posevec[3:] = rot.as_quat()
        # gripper_pose = se3.posevec2pose(gripper_posevec)
        pose_message = pose_to_ros_msg(gripper_posevec)
        target_pose_publisher.publish(pose_message)
        start_visual_servo_publisher.publish(start_signal)
        time.sleep(0.05)
        head.set_pan(-4 * np.pi / 180, active_cancellation=True)

        if vizualize:
            idx = int((j - (j % FRAME_STEP)) / FRAME_STEP) + 1

            next_pose_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.100)
            next_pose_coordinate_frame.transform(gripper_pose)

            gripper_pcd = copy.deepcopy(initial_gripper_pose_pcd).transform(
                interpolated_gripper_poses[j] @ initial_gripper_pose_inv)
            gripper_pcd.transform(extrinsics_camera)

            initial_scene_pcd.transform(extrinsics_camera)

            current_rgb = controller.camera.get_rgb()
            current_depth = controller.camera.get_aligned_depth() / 1000
            current_depth_to_pcd = depth_to_point_cloud(current_depth, intrinsics_camera[0, 0], intrinsics_camera[1, 1],
                                                        intrinsics_camera[0, 2], intrinsics_camera[1, 2])
            current_depth_to_pcd = current_depth_to_pcd.reshape(-1, 3)
            current_rgb = current_rgb / 255
            current_rgb = current_rgb.reshape(-1, 3)
            current_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(current_depth_to_pcd))
            current_pcd.colors = o3d.utility.Vector3dVector(current_rgb)
            current_pcd.transform(extrinsics_camera)

            current_eef_pose = controller.get_eef_pose()
            eef_coordinate_frame_reached = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.100)
            eef_coordinate_frame_reached.transform(current_eef_pose)

            gripper_pcd_target_waypoint = o3d.io.read_point_cloud(
                f'{task_folder}/scene_files/gripper_pcd_{idx * FRAME_STEP}.ply')
            live_image_pcd_from_demo = o3d.io.read_point_cloud(
                f'{task_folder}/scene_files/live_image_pcd_{idx * FRAME_STEP}.ply')

            gripper_pcd_target_waypoint.transform(extrinsics_camera)
            gripper_pcd_target_waypoint.paint_uniform_color([0.0, 0.0, 1.0])

            hand_mesh = o3d.io.read_triangle_mesh(f'{SCENE_FILES_FOLDER}/hand_mesh_{idx * FRAME_STEP}.ply')
            hand_mesh_pcd = hand_mesh.sample_points_uniformly(number_of_points=10000)
            hand_mesh_pcd.transform(extrinsics_camera)
            o3d.visualization.draw_geometries(
                [current_pcd, gripper_pcd_target_waypoint, next_pose_coordinate_frame, gripper_pcd, hand_mesh_pcd])
    "Send message to STOP visual servoing"
    start_signal = UInt8()
    start_signal.data = 0
    start_visual_servo_publisher.publish(start_signal)
    time.sleep(.1)

def record_waypoints():

    controller.go_to_neutral()

    record = True
    waypoints = []
    grippers = []

    # start a keyboard listener

    start_time = time.time()

    while time.time() - start_time < 20:
        print("Press enter to record waypoint")
        time.sleep(0.2)
        waypoints.append(controller.get_eef_pose())
        grippers.append(0)
        print("Waypoint recorded")
        print("Press enter to continue recording waypoints or q to stop")

    return np.asarray(waypoints), np.asarray(grippers)

if __name__ == '__main__':
    replay = False
    task = "assets/evaluations/can_0"
    if not replay:
        # actions, gripper_pcds, gripper_opening, gripper_pose_frame = np.asarray(
        #     extract_actions(folder=task, vizualize_3d=False))
        # print("a", actions.shape)
        # # print("pcd", gripper_pcds.shape)
        # print("go",gripper_opening.shape)
        input("Watch saved video and press enter to execute actions...")
        actions, gripper_opening = record_waypoints()
        input("Execute...")
        controller.go_to_neutral()
        execute_actions_world_frame(actions, gripper_opening, folder=task, vizualize=False)
    else:

        if True:
            # show rgbs as video
            for i in range(hands_rgb.shape[0]):
                im = cv2.cvtColor(hands_rgb[i], cv2.COLOR_BGR2RGB)
                # im = cv2.cvtColor(hands_depth[i],cv2.COLOR_GRAY2BGR)
                cv2.imshow("rgb", im)
                cv2.waitKey(1)
                time.sleep(0.01)
            cv2.destroyAllWindows()
        replay_demo(vizualize=False)
