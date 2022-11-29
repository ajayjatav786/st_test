from typing import List

import PIL.Image


def image_grid(images: List[PIL.Image.Image], rows, cols):
	assert len(images) <= rows * cols

	w, h = images[0].size
	grid = PIL.Image.new('RGB', size=(cols * w, rows * h))

	for i, img in enumerate(images):
		grid.paste(img, box=(i % cols * w, i // cols * h))

	return grid
