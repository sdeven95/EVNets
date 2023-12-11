
class Constants:
	DEFAULT_IMAGE_WIDTH = DEFAULT_IMAGE_HEIGHT = 224
	DEFAULT_IMAGE_CHANNELS = 3
	DEFAULT_VIDEO_FRAMES = 8

	DEFAULT_LOG_FREQ = 500

	DEFAULT_ITERATIONS = 300000   # default iterations
	DEFAULT_EPOCHS = 300  # default epochs

	# as big as possible to avoid interrupt,
	# when iteration based training, set epochs to big number
	# when epoch based training, set iterations to big number
	DEFAULT_MAX_ITERATIONS = DEFAULT_MAX_EPOCHS = 10000000
