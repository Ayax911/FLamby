# Weighted focal loss, as in Fed-ISIC2019 (He et al., https://arxiv.org/abs/1708.02002).
import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class BaselineLoss(_Loss):
    """Weighted focal loss for the benign / malignant task.

    Attributes
    ----------
    alpha: torch.tensor of size 2, class weights (benign, malignant)
    gamma: float, for gamma = 0 focal loss is the same as CE loss, increasing gamma
    reduces the loss of the "easy to classify" examples
    """

    def __init__(
        self,
        # N_train / n_class on the train partition: 6671/4393 and 6671/2278
        alpha=torch.tensor([1.5186, 2.9284]),
        gamma=2.0,
    ):
        super(BaselineLoss, self).__init__()
        self.alpha = alpha.to(torch.float)
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Parameters
        ----------
        inputs : torch.tensor of size (N, 2), logits output by the model (pre-softmax)
        targets : torch.tensor of size (N,), int, 0 (benign) or 1 (malignant)
        """
        targets = targets.view(-1, 1).type_as(inputs)
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.long())
        logpt = logpt.view(-1)
        pt = logpt.exp()
        self.alpha = self.alpha.to(targets.device)
        at = self.alpha.gather(0, targets.data.view(-1).long())
        logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt

        return loss.mean()
