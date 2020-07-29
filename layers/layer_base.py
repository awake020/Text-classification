"""abstract base class for all type of layers"""
import torch.nn as nn


class LayerBase(nn.Module):
    """Abstract base class for all type of layers."""

    def __init__(self):
        super(LayerBase, self).__init__()

    def apply_mask(self, input_tensor, mask_tensor):
        input_tensor = self.tensor_ensure_gpu(input_tensor)
        mask_tensor = self.tensor_ensure_gpu(mask_tensor)
        return input_tensor * mask_tensor.unsqueeze(-1).expand_as(input_tensor)

    def get_seq_len_list_from_mask_tensor(self, mask_tensor):
        batch_size = mask_tensor.shape[0]
        return [int(mask_tensor[k].sum().item()) for k in range(batch_size)]
