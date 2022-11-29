from typing import List, Tuple

from .schedule import Schedule


class StepSchedule(Schedule):
	def __init__(self,
	             initial_value: float,
	             steps: List[Tuple[float, float]],
	             num_train_timesteps: int = 1000):
		super().__init__(initial_value, num_train_timesteps)
		self._steps = sorted(steps, reverse=True)

	def get_value(self, timestamp: int) -> float:
		t = timestamp / self._num_train_timesteps
		value = self._initial_value
		for s, v in self._steps:
			if t <= s:
				value = v
			elif t > s:
				break
		return value
