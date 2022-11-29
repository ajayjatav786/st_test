from os import path as osp

import numpy as np

from gfpgan import GFPGANer


class GFPGANEnhancer:
	def __init__(self,
	             base_model_path: str,
	             model_version: str = '1.4',
	             gfp_weight: float = 0.5,
	             upscale: int = 1):
		self.base_model_path = base_model_path
		self.gfp_weight = gfp_weight
		self.upscale = upscale
		self.only_center_face = True  # Only restore the center face
		self.aligned = False  # Input are aligned faces

		# No BG processing for now
		self.bg_upsampler = None

		self.set_model_version(model_version)

	def set_model_version(self, model_version: str):
		self.model_version = model_version
		if model_version == '1':
			self.arch = 'original'
			self.channel_multiplier = 1
			self.model_name = 'GFPGANv1'
			self.url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
		elif model_version == '1.2':
			self.arch = 'clean'
			self.channel_multiplier = 2
			self.model_name = 'GFPGANCleanv1-NoCE-C2'
			self.url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
		elif model_version == '1.3':
			self.arch = 'clean'
			self.channel_multiplier = 2
			self.model_name = 'GFPGANv1.3'
			self.url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
		elif model_version == '1.4':
			self.arch = 'clean'
			self.channel_multiplier = 2
			self.model_name = 'GFPGANv1.4'
			self.url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
		elif model_version == 'RestoreFormer':
			self.arch = 'RestoreFormer'
			self.channel_multiplier = 2
			self.model_name = 'RestoreFormer'
			self.url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
		else:
			raise ValueError(f'Wrong model version {model_version}.')

		model_path = osp.join(self.base_model_path, self.model_name + '.pth')

		self.restorer = GFPGANer(model_path=model_path,
		                         upscale=self.upscale,
		                         arch=self.arch,
		                         channel_multiplier=self.channel_multiplier,
		                         bg_upsampler=self.bg_upsampler)

	def process(self, image: np.ndarray) -> np.ndarray:
		cropped_faces, restored_faces, restored_image = self.restorer.enhance(
			image,
			has_aligned=self.aligned,
			only_center_face=self.only_center_face,
			paste_back=True,
			weight=self.gfp_weight
		)

		return restored_image
