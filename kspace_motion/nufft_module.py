import os
import sys
import torch
import numpy as np
from torch.nn.parameter import Parameter

# optional local checkout of pytorch-finufft, e.g. export PYTORCH_FINUFFT_PATH=/path/to/pytorch-finufft
if os.environ.get("PYTORCH_FINUFFT_PATH"):
    sys.path.append(os.environ["PYTORCH_FINUFFT_PATH"])
import pytorch_finufft

###################################################################################################
class KSpaceMotion(torch.nn.Module):

  def __init__(self, data_shape, device="cpu"):
    super().__init__()
    self.device = device
    self.data_shape = data_shape
    self.rotations = self._init_rotations()
    self.translation = self._init_translations()

  def forward(self, input):
    out = pytorch_finufft.functional.finufft_type2(self.rotations, input,
                                                    modeord=0, isign=-1)
    out = out * torch.exp(-1j * self.translation)
    out = out.reshape(input.shape)#(self.data_shape)
    #out = torch.abs(torch.fft.ifftshift(torch.fft.ifftn(out)))
    return out

  def _init_rotations(self):
    return Parameter(torch.zeros((3, np.prod(self.data_shape)), device=self.device))
    #return Parameter(torch.rand((3, np.prod(self.data_shape)), device=self.device))

  def _init_translations(self):
    return Parameter(torch.zeros(np.prod(self.data_shape), device=self.device))
    #return Parameter(torch.rand(np.prod(self.data_shape), device=self.device))

############################################################################################
class InverseFFT(torch.nn.Module):
  
  def forward(self, input):
    return torch.abs(torch.fft.ifftshift(torch.fft.ifftn(input)))
  
