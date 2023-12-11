# ======= The code should be moved to __init__.py ==============

# the process likes following:
# first, sampler make sample indices of one batch likes [(index, ...), ...]
# second, send the simple index (index, ...) one by one to dataset __getitem__ ,
#         get the specified index sample likes { 'image', imageObject [C,H,W], 'label', labelInt, ... }
# third, send list [ {'image', imageObject, 'label', labelInt, ... }, ...] to collate function
#        get the dict { 'image', [N,C,H,W], 'label': list(int), ...]
# the last result could be got by enumerating dataloader

# Note: if you want to customize transform for image, you can do it in __getitem__ of dataset


from utils.type_utils import Opt
from .sampler.utils import SplUtil
from .dataloader import EVNetsDataloader
from utils.entry_utils import Entry


class DataLoaderUtil:
	@staticmethod
	def get_eval_loader():

		# ---- dataset ----------
		eval_dataset = Entry.get_entity(Entry.Dataset, is_traing=False)
		n_eval_samples = len(eval_dataset)
		data_workers = eval_dataset.workers
		prefetch_factor = eval_dataset.prefetch_factor if data_workers > 0 else None
		persistent_workers = eval_dataset.persistent_workers if data_workers > 0 else False
		pin_memory = eval_dataset.pin_memory

		# ----- sampler ---------
		Opt.set_by_cfg_path(
			Entry.Sampler,
			"val_batch_size",
			Opt.get_by_cfg_path(Entry.Sampler, "eval_batch_size")
		)

		# crop_size_h, crop_size_w = image_size_from_opts()
		# # ------------------ !!!! Don't forget the video sampler -----------
		# setattr(opts, _sampler_path + ".name", "BatchSampler")
		# setattr(opts, _sampler_path + ".BatchSampler.crop_size_height", crop_size_h)
		# setattr(opts, _sampler_path + ".BatchSampler.crop_size_width", crop_size_w)

		eval_sampler = Entry.get_entity(Entry.Sampler, n_data_samples=n_eval_samples, is_training=False)

		# ------- collate function ------------
		eval_collate_fn = Entry.get_entity(Entry.Collate, label_name="eval")

		# ------- data loader ----------------
		return EVNetsDataloader(
			dataset=eval_dataset,
			batch_size=1,  # handled in sampler
			batch_sampler=eval_sampler,
			num_workers=data_workers,
			pin_memory=pin_memory,
			persistent_workers=persistent_workers,
			collate_fn=eval_collate_fn,
			prefetch_factor=prefetch_factor,
		)

	@staticmethod
	def get_train_loader():
		# ------ dataset --------
		train_dataset = Entry.get_entity(Entry.Dataset, is_training=True)
		data_workers = train_dataset.workers
		persistent_workers = train_dataset.persistent_workers if data_workers > 0 else False
		prefetch_factor = train_dataset.prefetch_factor if data_workers > 0 else None
		pin_memory = train_dataset.pin_memory

		# ------ sampler ----------
		n_train_samples = len(train_dataset)
		train_sampler = Entry.get_entity(Entry.Sampler, n_data_samples=n_train_samples, is_training=True)

		# ------- collate function ---------
		train_collate_fn = Entry.get_entity(Entry.Collate, label_name="train")

		# -------- data loader ------------
		train_loader = EVNetsDataloader(
			dataset=train_dataset,
			batch_size=1,  # Handled inside data sampler
			num_workers=data_workers,
			pin_memory=pin_memory,
			batch_sampler=train_sampler,
			persistent_workers=persistent_workers,
			collate_fn=train_collate_fn,
			prefetch_factor=prefetch_factor,
		)

		return train_loader

	@staticmethod
	def get_val_loader():

		# ------ dataset --------
		val_dataset = Entry.get_entity(Entry.Dataset, is_traing=False)
		data_workers = val_dataset.workers
		persistent_workers = val_dataset.persistent_workers if data_workers > 0 else False
		prefetch_factor = val_dataset.prefetch_factor if data_workers > 0 else None
		pin_memory = val_dataset.pin_memory

		# ------ sampler ----------
		n_val_samples = len(val_dataset)
		val_sampler = Entry.get_entity(Entry.Sampler, n_data_samples=n_val_samples, is_training=False)

		# ------- collate function ---------
		val_collate_fn = Entry.get_entity(Entry.Collate, label_name="val")

		# -------- data loader ------------
		val_loader = EVNetsDataloader(
			dataset=val_dataset,
			batch_size=1,
			batch_sampler=val_sampler,
			num_workers=data_workers,
			pin_memory=pin_memory,
			persistent_workers=persistent_workers,
			collate_fn=val_collate_fn,
			prefetch_factor=prefetch_factor,
		)

		return val_loader

	@staticmethod
	def get_train_val_loader():

		# ------ dataset --------
		train_dataset = Entry.get_entity(Entry.Dataset, is_training=True)
		val_dataset = Entry.get_entity(Entry.Dataset, is_training=False)
		data_workers = train_dataset.workers
		persistent_workers = train_dataset.persistent_workers
		pin_memory = train_dataset.pin_memory
		prefetch_factor = train_dataset.prefetch_factor if data_workers > 0 else None

		# ------ sampler ----------
		n_train_samples = len(train_dataset)
		train_sampler = Entry.get_entity(Entry.Sampler, n_data_samples=n_train_samples, is_training=True)
		n_val_samples = len(val_dataset)
		val_sampler = Entry.get_entity(Entry.Sampler, n_data_samples=n_val_samples, is_training=False)

		# ------- collate function ---------
		train_collate_fn = Entry.get_entity(Entry.Collate, label_name="train")
		val_collate_fn = Entry.get_entity(Entry.Collate, label_name="val")

		# -------- data loader ------------
		train_loader = EVNetsDataloader(
			dataset=train_dataset,
			batch_size=1,  # Handled inside data sampler
			num_workers=data_workers,
			pin_memory=pin_memory,
			batch_sampler=train_sampler,
			persistent_workers=persistent_workers,
			collate_fn=train_collate_fn,
			prefetch_factor=prefetch_factor,
		)

		val_loader = EVNetsDataloader(
			dataset=val_dataset,
			batch_size=1,
			batch_sampler=val_sampler,
			num_workers=data_workers,
			pin_memory=pin_memory,
			persistent_workers=persistent_workers,
			collate_fn=val_collate_fn,
			prefetch_factor=prefetch_factor,
		)

		return train_loader, val_loader, train_sampler
