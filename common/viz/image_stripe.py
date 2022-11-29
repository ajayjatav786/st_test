from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt


def image_stripe(images: List[np.ndarray], titles: Optional[List[str]] = None, scale: float = 8.0):
	if titles is None:
		titles = [''] * len(images)

	assert len(images) == len(titles)

	num_images = len(images)
	fig = plt.figure(figsize=(scale * num_images, scale * num_images))
	for i, (image, title) in enumerate(zip(images, titles)):
		ax = fig.add_subplot(num_images, 1, i + 1)
		imgplot = plt.imshow(image)
		ax.set_title(title)
		ax.get_xaxis().set_ticks([])
		ax.get_yaxis().set_ticks([])
