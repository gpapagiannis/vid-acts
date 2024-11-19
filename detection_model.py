from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (
    inference_top_down_pose_model,
    init_pose_model,
    process_mmdet_results,
    vis_pose_result,
)
import numpy as np
import torch
from torch import nn

class DetModel:
    MODEL_DICT = {
        "YOLOX-tiny": {
            "config": "mmdet_configs/configs/yolox/yolox_tiny_8x8_300e_coco.py",
            "model": "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth",
        },
        "YOLOX-s": {
            "config": "mmdet_configs/configs/yolox/yolox_s_8x8_300e_coco.py",
            "model": "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth",
        },
        "YOLOX-l": {
            "config": "mmdet_configs/configs/yolox/yolox_l_8x8_300e_coco.py",
            "model": "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth",
        },
        "YOLOX-x": {
            "config": "mmdet_configs/configs/yolox/yolox_x_8x8_300e_coco.py",
            "model": "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth",
        },

        "Swin-T": {
            "config": "mmdet_configs/configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py",
            "model": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth",
        },
                "Swin-T2": {
            "config": "mmdet_configs/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py",
            "model": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
        },

                        "Swin-T3": {
            "config": "mmdet_configs/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py",
            "model": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
        },
    }

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._load_all_models_once()
        self.model_name = "YOLOX-x"
        self.model = self._load_model(self.model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        d = self.MODEL_DICT[name]
        return init_detector(d["config"], d["model"], device=self.device)

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def detect_and_visualize(self, image: np.ndarray, score_threshold: float) -> tuple[list[np.ndarray], np.ndarray]:
        out = self.detect(image)
        vis = self.visualize_detection_results(image, out[0], score_threshold)
        return out, vis

    def detect(self, image: np.ndarray) -> list[np.ndarray]:
        image = image[:, :, ::-1]  # RGB -> BGR
        out = inference_detector(self.model, image)
        return out

    def visualize_detection_results(
        self, image: np.ndarray, detection_results: list[np.ndarray], score_threshold: float = 0.3
    ) -> np.ndarray:
        person_det = [detection_results[0]] + [np.array([]).reshape(0, 5)] * 79

        image = image[:, :, ::-1]  # RGB -> BGR
        vis = self.model.show_result(
            image, person_det, score_thr=score_threshold, bbox_color=None, text_color=(200, 200, 200), mask_color=None
        )
        return vis[:, :, ::-1]  # BGR -> RGB
    
