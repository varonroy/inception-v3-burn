import os
import torch
from torch.hub import load_state_dict_from_url
from torchvision.models.inception import Inception3, InceptionA, InceptionC, InceptionE
import argparse

FID_WEIGHTS_URL = "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth"

parser = argparse.ArgumentParser(
    prog="InceptionV3 FID Weights Downloader",
    description="Converts the weights provided by the source from the old to the new PyTorch format",
)

parser.add_argument(
    "--file",
    default="~/.cache/inception-v3-burn/pt_inception-2015-12-05-6726825d.pth",
    help="Path to the weights file",
)

args = parser.parse_args()
file = args.file

file = os.path.expanduser(args.file)

dir = os.path.dirname(file)

if not os.path.exists(dir):
    os.makedirs(dir)
    print(f"Directory `{dir}` created.")

model = Inception3(num_classes=1008, aux_logits=False)

state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)

model.Mixed_5b = InceptionA(192, pool_features=32)
model.Mixed_5c = InceptionA(256, pool_features=64)
model.Mixed_5d = InceptionA(288, pool_features=64)
model.Mixed_6b = InceptionC(768, channels_7x7=128)
model.Mixed_6c = InceptionC(768, channels_7x7=160)
model.Mixed_6d = InceptionC(768, channels_7x7=160)
model.Mixed_6e = InceptionC(768, channels_7x7=192)
model.Mixed_7b = InceptionE(1280)
model.Mixed_7c = InceptionE(2048)

model.load_state_dict(state_dict)
model_weights = model.state_dict()
torch.save(model_weights, file)
print(f"saved to `{file}`.")
