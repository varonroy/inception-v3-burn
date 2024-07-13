# InceptionV3 - Burn
This project provides an implementation for the InceptionV3 as described in the [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) paper.

The implementation is almost a one-to-one translation of the [PyTorch](https://pytorch.org/) [implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py) (also see the [hub page](https://pytorch.org/hub/pytorch_vision_inception_v3/)).

Pre-trained weights for this model can be either downloaded from PyTorch (using torchvision), or from [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid).

## Downloading FID Weights
The FID weights provided by [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid) use the legacy version of PyTorch's serialization which is not supported by Burn (or more precisely, by Candle which Burn uses in the background). Therefore, the script `download_fid_weights.py` is provided. This script downloads the weights, and re-saves them in the current PyTorch format.

To run the script:
```sh
# If no arguments are provided, the weights file will be saved to the default location:
# `~/.cache/inception-v3-burn/pt_inception-2015-12-05-6726825d.pth`
python download_fid_weights.py

# Alternatively, you can provide a custom path.
python download_fid_weights.py --file PATH_TO_FILE
```

Then, add the model to your dependencies:
```toml
[dependencies]
inception-v3-burn = { git = "https://github.com/varonroy/inception-v3-burn", features = ["pretrained"] }
```

And initialize it using the weights that were prepared in the previous steps.
```rust
use inception_v3_burn::model::{
    weights::{downloader::InceptionV3PretrainedLoader, WeightsSource},
    InceptionV3,
};

fn main() {
    type B = burn::backend::NdArray;
    let device = burn::backend::ndarray::NdArrayDevice::default();

    // If you have saved the model to a location other than the default one,
    // replace None, with `Some(<fid-weights-file-path>)`.
    let (config, model) = InceptionV3::<B>::pretrained(WeightsSource::fid(None), &device).unwrap();
}
```

## License
This implementation is licensed under the MIT license.

For the pre-trained weights' licenses, please refer to their original sources:
- [PyTorch](https://pytorch.org/hub/pytorch_vision_inception_v3/)
- [FID](https://github.com/mseitzer/pytorch-fid)
