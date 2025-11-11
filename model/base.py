import torch


class BaseModule(torch.nn.Module):
    """Common helpers shared by Grad-TTS modules."""

    def __init__(self):
        super().__init__()

    def relocate_input(self, x: list):
        """
        Relocates provided tensors to the same device set for the module.
        """
        device = next(self.parameters()).device
        for i in range(len(x)):
            if isinstance(x[i], torch.Tensor) and x[i].device != device:
                x[i] = x[i].to(device)
        return x
