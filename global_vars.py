import numpy as np
import os
os.chdir(os.path.expanduser(
    "/home/george/Desktop/Projects/obi-wan/video_to_actions/"))  # set default experiments location

task_folder = "../assets/demos/grasp_fanta/grasp_fanta_0"
# INTRINSICS_REAL_CAMERA = np.load(f"{task_folder}/intrinsics_iphone.npy")
LOAD_EXISTING_DATA_FOR_PROCESSING = False

INTRINSICS_REAL_CAMERA = np.load("head_cam_intrinsic_matrix_aligned_depth.npy")
START = 10 # which part of the video to use start:end
END = 1000
FRAME_STEP = 2 # use every nth frame
# hands_rgb = np.load(f"{task_folder}/rgb.npy")[START:END]
# hands_depth = np.load(f"{task_folder}/depth.npy")[START:END]
MODEL_MANO_PATH = '/home/george/Desktop/Projects/obi-wan/video_to_actions/_DATA/data/mano'
SCENE_FILES_FOLDER = f"{task_folder}/scene_files"
# make dir if not exists
os.makedirs(SCENE_FILES_FOLDER, exist_ok=True)

INTRINSICS_HAMER_RENDERER = np.eye(4)
# INTRINSICS_HAMER_RENDERER[0, 0] = 918.0
# INTRINSICS_HAMER_RENDERER[1, 1] = 918.0
# INTRINSICS_HAMER_RENDERER[0, 2] = 128.0
# INTRINSICS_HAMER_RENDERER[1, 2] = 96.0
INTRINSICS_HAMER_RENDERER[0 ,0] = 2295.0
INTRINSICS_HAMER_RENDERER[1, 1] = 2295.0
INTRINSICS_HAMER_RENDERER[0, 2] = 320.0
INTRINSICS_HAMER_RENDERER[1, 2] = 240.0
#
# INTRINSICS_HAMER_RENDERER[0 ,0] = 952.5
# INTRINSICS_HAMER_RENDERER[1, 1] = 952.5
# INTRINSICS_HAMER_RENDERER[0, 2] = 320.0
# INTRINSICS_HAMER_RENDERER[1, 2] = 240.0

DISTANCE_BETWEEN_GRIPPERS_FINGERS = 0.08507
T_OPENGL_TO_OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
HUMAN_HAND_COLOR=(0.999, 0.6745, 0.4117)
VIZ_DEMO = True
MANO_HAND_IDS = {"wrist": 0, "index_mcp": 1, "index_pip": 2, "index_dip": 3, "middle_mcp": 4, "middle_pip": 5, "middle_dip": 6, "pinkie_mcp": 7, "pinkie_pip": 8, "pinkie_dip": 9,
                "ring_mcp": 10, "ring_pip": 11, "ring_dip": 12, "thumb_mcp": 13, "thumb_pip": 14, "thumb_dip": 15, "thumb_tip": 16, "index_tip": 17, "middle_tip": 18, "ring_tip": 19,
                "pinky_tip": 20}


