import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)

def hinge_d_loss(logits_real, logits_fake):
    loss_real = F.relu(1. - logits_real).mean()
    loss_fake = F.relu(1. + logits_fake).mean()
    return 0.5 * (loss_real + loss_fake)

def vanilla_d_loss(logits_real, logits_fake):
    return 0.5 * (F.softplus(-logits_real).mean() + F.softplus(logits_fake).mean())


class PatchDiscriminator(nn.Module):
    def __init__(self, num_layers: int = 3, num_channels: int = 64):
        super().__init__()
        channels = [num_channels * min(2 ** i, 8) for i in range(num_layers + 1)]  # gradually increase the number of filters
        strides = [2] * (num_layers - 1) + [1]  # last layer has stride 1

        sequence = [nn.Conv2d(3, num_channels, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        for in_channels, out_channels, stride in zip(channels[:-1], channels[1:], strides):
            sequence += [
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(out_channels, 1, kernel_size=4, stride=1, padding=1)]  # output 1 channel prediction map

        self.layers = nn.Sequential(*sequence)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def g_loss(self, fake: torch.Tensor, rec_loss: torch.Tensor, last_layer: nn.Module) -> torch.Tensor:
        g_loss = -torch.mean(self(fake))  # loss goes down when prob_real goes up
        try:
            rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
            d_weight = torch.norm(rec_grads) / (torch.norm(g_grads) + 1e-4)
            d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
            return d_weight * g_loss
        except RuntimeError as e:
            raise RuntimeError("last_layer has no grads, make sure you're not running your model with torch.no_grad") from e

    def d_loss(self, real: torch.Tensor, fake: torch.Tensor, loss: str) -> torch.Tensor:
        d_loss_fns = {'hinge': hinge_d_loss, 'vanilla': vanilla_d_loss}
        assert loss in d_loss_fns, f'`loss` arg must be one of {d_loss_fns.keys()}'
        logits_real = self(real.contiguous().detach())
        logits_fake = self(fake.contiguous().detach())
        return d_loss_fns[loss](logits_real, logits_fake)
