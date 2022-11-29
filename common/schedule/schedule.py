from typing import Optional
from abc import ABC, abstractmethod


class Schedule(ABC):
	def __init__(self,
	             initial_value: float,
	             num_train_timesteps: int = 1000):
		assert initial_value >= 0.0
		self._initial_value = initial_value
		self._num_train_timesteps = num_train_timesteps

	def __call__(self, timestamp: Optional[int] = None) -> float:
		if timestamp is None:
			return self._initial_value
		else:
			return self.get_value(timestamp)

	def __eq__(self, other: float) -> bool:
		return self._initial_value == other

	def __ne__(self, other: float) -> bool:
		return self._initial_value != other

	def __lt__(self, other: float) -> bool:
		return self._initial_value < other

	def __le__(self, other: float) -> bool:
		return self._initial_value <= other

	def __gt__(self, other: float) -> bool:
		return self._initial_value > other

	def __ge__(self, other: float) -> bool:
		return self._initial_value >= other

	@abstractmethod
	def get_value(self, timestamp: int) -> float:
		raise NotImplementedError()
