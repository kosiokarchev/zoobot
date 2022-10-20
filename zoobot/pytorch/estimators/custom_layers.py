
import wandb
from torch import Tensor, nn

class PermaDropout(nn.modules.dropout._DropoutNd):
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/dropout.html#Dropout
    def forward(self, x: Tensor) -> Tensor:
            return nn.functional.dropout(x, self.p, True, self.inplace)  # simply replaced self.training with True

class WandBLog(nn.Module):

    def __init__(self, log_string, batch_to_image=False):
        super().__init__()
        self.log_string = log_string
        self.batch_to_image = batch_to_image

    # log and then return
    def forward(self, x) -> Tensor:
        if self.batch_to_image:
            # wandb.log({self.log_string: x.transpose(3, 1)})
            wandb.log({self.log_string: wandb.Image(x)})
        else:    
            wandb.log({self.log_string: x})
        return x
