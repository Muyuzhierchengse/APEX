import torch


class NFECounter:
    def __init__(self):
        self.value = 0
        self._hook_handle = None

    def register(self, model: torch.nn.Module):
        self.remove()

        def _hook(module, input, output):
            self.value += 1
        self._hook_handle = model.register_forward_hook(_hook)

    def reset(self):
        self.value = 0

    def remove(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
