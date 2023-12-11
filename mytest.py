#
# class A:
# 	_data = 789
#
# 	@classmethod
# 	def _print_data(cls):
# 		print(cls._data)
#
# 	@classmethod
# 	def print_data(cls):
# 		for middle_cls in cls.mro()[:-1][::-1]:
# 			middle_cls._print_data()
#
# class B(A):
# 	pass
#
# class C(B):
# 	_data = 456
#
#
# obj = C()
#
# print(type(A))

# class MetaClass(type):
# 	def __new__(mcs, name, bases, attrs):
# 		if "__slots__" in attrs:
# 			attrs.pop("__slots__")
# 		return type.__new__(mcs, name, bases, attrs)
#
#
# class Disp(metaclass=MetaClass):
# 	__slots__ = ["_disp_keys_"]
#
# 	_disp_ = None
#
# 	def __init__(self):
# 		if not hasattr(self, "_disp_keys_"):
# 			self.__class__.add_disp_keys()
#
# 	@classmethod
# 	def _add_disp_keys_(cls, tar_dict):
# 		if cls._disp_:
# 			tar_dict.update(cls._disp_)
#
# 	@classmethod
# 	def add_disp_keys(cls):
# 		cls._disp_keys_ = set()
# 		for middle_cls in cls.mro()[:-1][::-1]:
# 			if issubclass(middle_cls, Disp):
# 				middle_cls._add_disp_keys_(cls._disp_keys_)
#
# 	def __repr__(self):
# 		if not self._disp_keys_:
# 			return self.__class__.__name__ + "()"
#
# 		tar_keys = self._disp_keys_
#
# 		tpl = self.__class__.__name__ + "("
# 		tpl_keys = list(map(lambda key: key + "={}", tar_keys))
# 		tpl += ", ".join(tpl_keys) + ")"
# 		return tpl.format(*[self.__dict__[key_] for key_ in tar_keys])
#
# 	def _repr_by_line(self):
# 		if not self._disp_keys_:
# 			return self.__class__.__name__ + "()\n"
#
# 		tar_keys = self._disp_keys_
#
# 		tpl = self.__class__.__name__ + "(\n"
# 		tpl_keys = map(lambda key: "\t" + key + " = {}", tar_keys)
# 		tpl += ",\n".join(tpl_keys) + "\n)"
# 		return tpl.format(*[self.__dict__[key_] for key_ in tar_keys])
#
#
# class A(Disp):
# 	_disp_ = ["a", "b", "c"]
#
#
# class B(A):
# 	_disp_ = ["A", "B", "C"]
#
#
# obj = B()
#
# obj.a, obj.b, obj.c = 1, 2, 3
# obj.A, obj.B, obj.C = 4, 5, 6
#
# print(obj._disp_keys_)
# print(repr(obj))


# def myFunc(a, b):
# 	return a + b
#
# class myClass:
# 	pass
#
# print(isinstance(myFunc, type))

# import torch
# x = torch.tensor([[1, 2], [3, 4]])
# y = torch.tensor([[5, 6], [7, 8]])
#
# sort_indices = torch.argsort(x, dim=-1)
# print(sort_indices)

# print(x.sum(dim=(0, 1)))
#
# # z = torch.stack([x, y], dim=1)
# # print(z.shape)
# # print(z)


# from data.transform import RandomHorizontalFlip, Normalize
#
# t = RandomHorizontalFlip()
#
# print(t)

# from torch import nn
# from layer.utils.init_params_util import InitParaUtil
#
# md = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 3))
# # for idx, m in enumerate(md.named_parameters()):
# # 	print(idx, "->", m)
# s_d = md.state_dict()
# print(s_d["0.weight"])
# print(s_d["0.bias"])
# print(s_d["1.weight"])
# print(s_d["1.bias"])

# x = {"a": 123, "b": 456}
#
# print(x.get("c", 789))

# from torch import nn
#
# ms = nn.Sequential(nn.Linear(2, 3), nn.Linear(4, 5))
# ms[1] = nn.Linear(7, 8)
#
# print(ms)

# class A:
# 	pass
#
# x = A()
#
# print(x.__class__.mro())

import timm.models
import torchvision.models
