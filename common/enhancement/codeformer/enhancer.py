from typing import Optional
from os import path as osp

import numpy as np
import torch
import torchvision.transforms.functional as tvf
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.registry import ARCH_REGISTRY
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

# Import CodeFormer to register it
from .arch import CodeFormer


class CodeFormerEnhancer:
	def __init__(self,
	             base_model_path: str,
	             fidelity_weight: float = 0.5,
	             face_detection_model: str = 'retinaface_resnet50',
	             upscale: int = 1,
	             download_model: bool = False,
	             verbose: bool = False,
	             device: str = 'cuda'):
		self._fidelity_weight = fidelity_weight
		self._verbose = verbose
		self._device = device

		pretrain_model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
		self.restorer = self._configure_code_former(base_model_path,
		                                            pretrain_model_url if download_model else None,
		                                            verbose,
		                                            device)

		self._face_helper = self._configure_face_helper(face_detection_model,
		                                                upscale,
		                                                device)

	def process(self, image: np.ndarray) -> np.ndarray:
		# Clean all the intermediate results to process the next image
		self._face_helper.clean_all()

		# RGB => BGR
		image_bgr = image[..., ::-1].copy()

		self._face_helper.read_image(image_bgr)

		# Get face landmarks for each face
		num_det_faces = self._face_helper.get_face_landmarks_5(
			only_center_face=True,
			resize=640,
			eye_dist_threshold=5
		)

		if self._verbose:
			print(f'Detected {num_det_faces} faces')

		# Align and warp each face
		self._face_helper.align_warp_face()

		# Process each cropped face
		for idx, cropped_face in enumerate(self._face_helper.cropped_faces):
			# Prepare data
			cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
			tvf.normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
			cropped_face_t = cropped_face_t.unsqueeze(0).to(self._device)

			try:
				with torch.no_grad():
					output = self.restorer(cropped_face_t, w=self._fidelity_weight, adain=True)[0]
					restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
				del output
				torch.cuda.empty_cache()
			except Exception as error:
				if self._verbose:
					print(f'Failed inference for CodeFormer: {error}')
				restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

			restored_face = restored_face.astype('uint8')
			self._face_helper.add_restored_face(restored_face)

		# Paste restored faces back
		bg_img = None
		self._face_helper.get_inverse_affine(None)
		restored_img = self._face_helper.paste_faces_to_input_image(upsample_img=None)

		# BGR => RGB
		return restored_img[..., ::-1]

	def _configure_code_former(self,
	                           base_model_path: str,
	                           pretrain_model_url: Optional[str],
	                           verbose: bool,
	                           device: str):
		restorer = ARCH_REGISTRY.get('CodeFormer')(
			dim_embd=512,
			codebook_size=1024,
			n_head=8,
			n_layers=9,
			connect_list=['32', '64', '128', '256']
		).to(device)

		if pretrain_model_url is not None:
			ckpt_path = load_file_from_url(url=pretrain_model_url,
			                               model_dir=base_model_path,
			                               progress=verbose,
			                               file_name=None)
		else:
			ckpt_path = osp.join(base_model_path, 'codeformer.pth')

		checkpoint = torch.load(ckpt_path)['params_ema']
		restorer.load_state_dict(checkpoint)
		restorer.eval()

		return restorer

	def _configure_face_helper(self,
	                           face_detection_model: str,
	                           upscale: int,
	                           device: str):
		return FaceRestoreHelper(
			upscale,
			face_size=512,
			crop_ratio=(1, 1),
			det_model=face_detection_model,
			save_ext='png',
			use_parse=True,
			device=device
		)
