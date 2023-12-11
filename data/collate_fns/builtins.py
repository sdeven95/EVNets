import torch
from utils.entry_utils import Entry
from utils import Util


@Entry.register_entry(Entry.Collate)
def default_collate_fn(batch):
	"""Default collate function"""
	batch_size = len(batch)

	# per element of batch is dict, such as { 'image' : imgObj, 'label': labelInt }
	# get keys list
	keys = list(batch[0].keys())

	# create a dict, such as { 'image‘， imageObjList, 'label', labelList }
	new_batch = {k: [] for k in keys}
	for b in range(batch_size):
		for k in keys:
			new_batch[k].append(batch[b][k])

	# stack the keys
	# -> { 'image' : imageTensor NxCxHxW, 'label': one dimensional tensor }
	for k in keys:
		batch_elements = new_batch.pop(k)

		if isinstance(batch_elements[0], (int, float)):
			# for label, list of ints or floats, convert to tensor
			batch_elements = torch.as_tensor(batch_elements)
		else:
			# for image Object list, to stack tensors (including 0-dimensional actually stack will create a new dimension = 0)
			try:
				batch_elements = torch.stack(batch_elements, dim=0).contiguous()
			except Exception as e:
				raise TypeError("Unable to stack the tensors. Error: {}".format(e))

		# convert image batch (NxCxHxW) memory format to channels_last
		if k == "image" and Util.channel_last():
			batch_elements = batch_elements.to(memory_format=torch.channels_last)

		new_batch[k] = batch_elements

	return new_batch
