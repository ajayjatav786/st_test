import numpy as np
from lightweight_openpose.modules.pose import Pose

from .pose_estimator import PoseEstimator


class PoseScorer:
	def __init__(self,
	             checkpoint_path: str,
	             device: str = 'cuda'):
		self.pose_estimator = PoseEstimator(checkpoint_path, device=device)

		# Parameters
		self._discrepancy_penalty = 1.0
		self._extra_poses_penalty = 10.0
		self._no_pose_score = 999.0  # when we cannot find any pose

		# Matrix to flip vectors in X direction
		self._flip = np.array([[-1, 0], [0, 1]])

		self._expected_keypoints_mask = [True, True, True,
		                                 False, False,
		                                 True,
		                                 False, False, False, False, False, False, False, False,
		                                 True, True, True, True]

	def compute(self, image: np.ndarray) -> float:
		# Estimate pose(s)
		poses = self.pose_estimator.estimate(image)

		if len(poses) == 0:
			return self._no_pose_score

		pose = poses[0]

		# Get points of interest ([x, y])
		nose = pose.keypoints[0]
		neck = pose.keypoints[1]
		r_sho = pose.keypoints[2]
		l_sho = pose.keypoints[5]
		r_eye = pose.keypoints[14]
		l_eye = pose.keypoints[15]
		r_ear = pose.keypoints[16]
		l_ear = pose.keypoints[17]

		# Compute several pairwise symmetry scores
		eyes_score = self._compute_pairwise_symmetry_score(r_eye, l_eye, nose)
		ears_score = self._compute_pairwise_symmetry_score(r_ear, l_ear, nose)
		shoulers_wrt_neck_score = self._compute_pairwise_symmetry_score(r_sho, l_sho, neck)
		shoulers_wrt_nose_score = self._compute_pairwise_symmetry_score(r_sho, l_sho, nose)

		# Compute extra penalty
		num_discrepancies = self._count_discrepancies(pose)
		discrepancy_penalty = num_discrepancies * self._discrepancy_penalty
		num_extra_poses = len(poses) - 1
		extra_poses_penalty = num_extra_poses * self._extra_poses_penalty

		# Compute final score
		score = eyes_score + ears_score + \
		        shoulers_wrt_neck_score + shoulers_wrt_nose_score + \
		        discrepancy_penalty + extra_poses_penalty
		return score

	def _compute_pairwise_symmetry_score(self, a, b, center) -> float:
		a_rel = a - center
		b_rel = self._flip @ (b - center)

		score = np.linalg.norm(a_rel - b_rel) / np.linalg.norm(a - b)
		return score

	def _count_discrepancies(self, pose: Pose) -> int:
		detected_keypoints_mask = np.all(pose.keypoints >= 0, axis=1)
		num_discrepancies = np.sum(np.logical_xor(self._expected_keypoints_mask, detected_keypoints_mask))
		return num_discrepancies
