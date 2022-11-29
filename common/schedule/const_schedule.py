from .schedule import Schedule


class ConstSchedule(Schedule):
	def __init__(self,
	             initial_value: float,
	             num_train_timesteps: int = 1000):
		super().__init__(initial_value, num_train_timesteps)

	def get_value(self, timestamp: int) -> float:
		return self._initial_value
