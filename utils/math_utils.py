
class Math:
	@staticmethod
	def make_divisible(v, divisor=8, min_value=None):
		if not min_value:
			min_value = divisor
		# new_v = max(min_value, int(np.ceil(v * 1. / divisor) * divisor))
		new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
		if new_v < 0.9 * v:
			new_v += divisor
		return new_v

	@staticmethod
	# make the value ranged in [min_val, val] or [min_val, max_val] if val > max_val
	def bound_fn(min_val, max_val, value):
		return max(min_val, min(max_val, value))
