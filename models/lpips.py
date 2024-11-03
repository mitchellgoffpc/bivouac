import torch
import torch.nn as nn
from torchvision import models
from helpers import CHECKPOINT_DIR, download_file

LPIPS_URL = "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
LPIPS_PATH = CHECKPOINT_DIR / "lpips.ckpt"


def normalize(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    return x / (torch.sum(x ** 2, dim=1, keepdim=True).sqrt() + eps)

def avg_pool(x: torch.Tensor) -> torch.Tensor:
    return x.mean((2, 3), keepdim=True)

def LinLayer(in_channels: int, out_channels: int = 1, use_dropout: bool = False) -> nn.Module:
    layers = [nn.Dropout()] if use_dropout else []
    layers += [nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)]
    return nn.ModuleDict({'model': nn.Sequential(*layers)})


class VGG16(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        vgg_pretrained_features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*vgg_pretrained_features[:4])
        self.slice2 = nn.Sequential(*vgg_pretrained_features[4:9])
        self.slice3 = nn.Sequential(*vgg_pretrained_features[9:16])
        self.slice4 = nn.Sequential(*vgg_pretrained_features[16:23])
        self.slice5 = nn.Sequential(*vgg_pretrained_features[23:30])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = h1 = self.slice1(x)
        h = h2 = self.slice2(h)
        h = h3 = self.slice3(h)
        h = h4 = self.slice4(h)
        h = h5 = self.slice5(h)
        return h1, h2, h3, h4, h5


class LPIPS(nn.Module):
    def __init__(self, use_dropout: bool = True):
        super().__init__()
        self.net = VGG16()
        self.lin0 = LinLayer(64, use_dropout=use_dropout)
        self.lin1 = LinLayer(128, use_dropout=use_dropout)
        self.lin2 = LinLayer(256, use_dropout=use_dropout)
        self.lin3 = LinLayer(512, use_dropout=use_dropout)
        self.lin4 = LinLayer(512, use_dropout=use_dropout)
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self) -> None:
        download_file(LPIPS_URL, LPIPS_PATH)
        self.load_state_dict(torch.load(LPIPS_PATH, map_location=torch.device("cpu"), weights_only=True), strict=False)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        input_feats = self.net((inputs - self.shift) / self.scale)
        target_feats = self.net((targets - self.shift) / self.scale)
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        diffs = [(normalize(x1) - normalize(x2)).square() for x1, x2 in zip(input_feats, target_feats)]
        return sum(avg_pool(lin.model(diff)) for lin, diff in zip(lins, diffs))
