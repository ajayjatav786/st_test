from .schedule import Schedule


class LinearSchedule(Schedule):
	def __init__(self,
	             t_a: float,
	             v_a: float,
	             t_b: float,
	             v_b: float,
	             num_train_timesteps: int = 1000):
		assert t_a >= t_b
		super().__init__(v_a, num_train_timesteps)
		self._t_a = t_a
		self._v_a = v_a
		self._t_b = t_b
		self._v_b = v_b

	def get_value(self, timestamp: int) -> float:
		t = timestamp / self._num_train_timesteps
		if t > self._t_a:
			return self._initial_value
		elif t < self._t_b:
			return self._v_b
		return self._v_a + (t - self._t_a) / (self._t_a - self._t_b) * (self._v_a - self._v_b)
