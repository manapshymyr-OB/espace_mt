import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import torch

# torch.cuda.is_available()
#
# torch.cuda.device_count()
#
# torch.cuda.current_device()
#
# torch.cuda.device(0)

print(torch.cuda.get_device_name(0))