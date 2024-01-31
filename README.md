# EVNet maybe

This is an initial work which is adaptive from Apple CVNets:

https://github.com/apple/ml-cvnets

Please refer to license of cvnets.

The target is making model training and testing completely configurable(I like configure file instead of command line arguments). Although it supports command line arguments, you can do everything in configure file.

In cvnets, you need write many codes to add, get and manage parameters. This work provides a simple framework to complete them automatically(include generating configure file template).

To train or test your model, an optional steps as following:

   * (1)Define your own model(if you want to add automatic arguments, refer to existed models)
   * (2)Define metrics for loss and evaluation if they don't exist
   * (3)Prepare your data and data accessing code
   * (4)Set model class in main.py and run it to print configure file template
   * (5)Edit template and complete your own configure file
   * (6)Set configure path in model definition and run for training or testing

Note, it only moves some works of cvnets not all and still need many tests. Latter it maybe wrapper some other frameworks.


Note, Mobile Aggregate Net and SoftReLU come from the following paper:

   * Lightweight food recognition via aggregation block and feature encoding

the paper provides an encoding block similar to positional encoding but with almost negligible parameters and computation which is especially effective to food recognition. you can add this block to any model to get smaller network in depth and number of parameters with pretty much the same performance.
the paper also proposes a new activation function, it's effective especially for small or medium-sized network and enables faster network convergence.

And Mobile Global Shuffle Net come from the following paper:

   * Efficient food image recognition with global shuffle convolution

the paper provides a novel convolution to replace transformer and verifies it on a parallel network. 

Any bug, problem, issue, suggestion, mail to: sdeven95@live.cn

Thanks






