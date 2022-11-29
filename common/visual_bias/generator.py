from typing import Optional, Tuple
import random

import numpy as np
from skimage.io import imread
from skimage.color import rgb2hsv, hsv2rgb, gray2rgb, rgba2rgb


class VisualBiasGenerator:
	def __init__(self,
	             head_path: str,
	             torso_path: str,
	             background_path: str):
		self.head_path = head_path
		self.torso_path = torso_path
		self.bg_path = background_path

		# Read images
		self.bg_image = imread(self.bg_path)
		self.torso_image = imread(self.torso_path)
		self.head_image = imread(self.head_path)

		# Scale images
		self.bg_image = self.bg_image.astype(np.float32) / 255.0
		self.torso_image = self.torso_image.astype(np.float32) / 255.0
		self.head_image = self.head_image.astype(np.float32) / 255.0

	def generate(self, clothes_color: Optional[str] = None) -> np.ndarray:
		# Adjust colors
		new_head_image = VisualBiasGenerator.colorize(self.head_image, saturation_factor=1.0)
		new_bg_image = VisualBiasGenerator.randomize_color(self.bg_image)
		if clothes_color is None or len(clothes_color) != 7 or clothes_color[0] != '#':
			new_torso_image = VisualBiasGenerator.colorize(self.torso_image, saturation_factor=0.5)
			new_torso_image = VisualBiasGenerator.randomize_color(new_torso_image)
		else:
			hue, saturation, value = VisualBiasGenerator.hex_to_hsv(clothes_color)
			new_torso_image = VisualBiasGenerator.colorize(self.torso_image,
			                                               hue=hue,
			                                               saturation=saturation,
			                                               value=value)

		# Compose
		composition = VisualBiasGenerator.compose(new_torso_image, new_bg_image)
		composition = VisualBiasGenerator.compose(new_head_image, composition)

		return composition

	def __iter__(self):
		return self

	def __next__(self) -> np.ndarray:
		return self.generate()

	@staticmethod
	def colorize(image: np.ndarray, *,
	             hue: Optional[float] = None,
	             saturation: Optional[float] = None,
	             value: Optional[float] = None,
	             saturation_factor: Optional[float] = None) -> np.ndarray:
		assert image.dtype == np.float32

		if hue is None and saturation is None and saturation_factor is None:
			return np.array(image)

		alpha = None

		# Convert gray images to RGB
		if len(image.shape) == 2 or image.shape[2] == 1:
			image = gray2rgb(image)

		# Separate alpha channel, if present
		elif image.shape[2] == 4:
			alpha = image[:, :, 3:]
			image = image[:, :, :3]

		# RGB -> HSV
		image_hsv = rgb2hsv(image)

		# Set Hue, if value is provided
		if hue is not None:
			image_hsv[:, :, 0] = np.clip(hue, 0.0, 1.0)

		# Set Saturation, if value is provided
		if saturation is not None:
			image_hsv[:, :, 1] = np.clip(saturation, 0.0, 1.0)

		# Set Value, if value is provided
		if value is not None:
			image_hsv[:, :, 2] = np.clip(value, 0.0, 1.0)

		# Adjust Saturation, if factor is provided
		if saturation_factor is not None:
			image_hsv[:, :, 1] = np.clip(image_hsv[:, :, 1] * saturation_factor, 0.0, 1.0)

		# HSV -> RGB
		image = hsv2rgb(image_hsv)

		# Add alpha back, if needed
		if alpha is not None:
			image = np.concatenate([image, alpha], axis=2)

		return image.astype(np.float32)

	@staticmethod
	def randomize_color(image: np.ndarray, saturation_factor: Optional[float] = None) -> np.ndarray:
		assert image.dtype == np.float32

		hue = random.random()
		saturation_factor = random.gauss(1.0, 1 / 3) if saturation_factor is None else saturation_factor

		return VisualBiasGenerator.colorize(image, hue=hue, saturation_factor=saturation_factor)

	@staticmethod
	def compose(foreground: np.ndarray, background: np.ndarray) -> np.ndarray:
		assert foreground.shape[0] == background.shape[0] and foreground.shape[1] == background.shape[1]
		assert foreground.dtype == background.dtype

		if foreground.shape[2] == 3:
			return foreground

		if background.shape[2] == 4:
			background = rgba2rgb(background)

		alpha = foreground[:, :, 3:4]
		return alpha * foreground[:, :, :3] + (1.0 - alpha) * background

	@staticmethod
	def hex_to_hsv(hex_value: str) -> Tuple[float]:
		rgb = tuple(int(hex_value[i:i + 2], 16) for i in (1, 3, 5))
		return tuple(rgb2hsv(np.array(rgb) / 255.0))
