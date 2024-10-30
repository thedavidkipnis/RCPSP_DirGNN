import torch

print(torch.__version__)

# model input vector always consists of attributes from the most recently completed predecessor
# and whichever attributes from the target node we deem as useful