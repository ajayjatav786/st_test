import torch
import numpy as np
import cv2

from lightweight_openpose.models.with_mobilenet import PoseEstimationWithMobileNet
from lightweight_openpose.modules.keypoints import extract_keypoints, group_keypoints
from lightweight_openpose.modules.load_state import load_state
from lightweight_openpose.modules.pose import Pose, track_poses
from lightweight_openpose.val import normalize, pad_width


class PoseEstimator:
	def __init__(self,
	             checkpoint_path: str,
	             device: str = 'cuda'):
		self._model = PoseEstimationWithMobileNet()
		checkpoint = torch.load(checkpoint_path, map_location=device)
		load_state(self._model, checkpoint)

		self._model = self._model.eval()
		self._model = self._model.to(device)

		self._device = device

		# Parameters
		self._stride = 8
		self._upsample_ratio = 4
		self._input_height = 256
		self._num_keypoints = Pose.num_kpts

	def estimate(self, input_image: np.ndarray):
		# Run inference
		heatmaps, pafs, scale, pad = self._run_inference(input_image)

		total_keypoints_num = 0
		all_keypoints_by_type = []
		for kpt_idx in range(self._num_keypoints):  # 19th for bg
			total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
			                                         total_keypoints_num)

		pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
		for kpt_id in range(all_keypoints.shape[0]):
			all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self._stride / self._upsample_ratio - pad[1]) / scale
			all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self._stride / self._upsample_ratio - pad[0]) / scale

		estimated_poses = []
		for n in range(len(pose_entries)):
			if len(pose_entries[n]) == 0:
				continue

			pose_keypoints = np.ones((self._num_keypoints, 2), dtype=np.int32) * -1
			for kpt_id in range(self._num_keypoints):
				if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
					pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
					pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])

			pose = Pose(pose_keypoints, pose_entries[n][18])
			estimated_poses.append(pose)

		return estimated_poses

	@staticmethod
	def draw_poses(image: np.ndarray, poses) -> np.ndarray:
		visualization = image.copy()
		for pose in poses:
			pose.draw(visualization)

		return visualization

	def _run_inference(self,
	                   image,
	                   pad_value=(0, 0, 0),
	                   image_mean=np.array([128, 128, 128], np.float32),
	                   image_scale=np.float32(1 / 256)):
		height, width, _ = image.shape
		scale = self._input_height / height

		scaled_image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
		scaled_image = normalize(scaled_image, image_mean, image_scale)
		min_dims = [self._input_height, max(scaled_image.shape[1], self._input_height)]
		padded_image, pad = pad_width(scaled_image, self._stride, pad_value, min_dims)

		image_tensor = torch.from_numpy(padded_image).permute(2, 0, 1).unsqueeze(0).float()
		image_tensor = image_tensor.to(self._device)

		stages_output = self._model(image_tensor)

		stage2_heatmaps = stages_output[-2]
		heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
		heatmaps = cv2.resize(heatmaps, (0, 0),
		                      fx=self._upsample_ratio, fy=self._upsample_ratio,
		                      interpolation=cv2.INTER_CUBIC)

		stage2_pafs = stages_output[-1]
		pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
		pafs = cv2.resize(pafs, (0, 0),
		                  fx=self._upsample_ratio, fy=self._upsample_ratio,
		                  interpolation=cv2.INTER_CUBIC)

		return heatmaps, pafs, scale, pad
