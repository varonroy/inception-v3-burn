use burn::prelude::*;
use nn::pool::{MaxPool2d, MaxPool2dConfig};

use super::basic_block::{BasicConv2d, BasicConv2dConfig};

#[derive(Debug, Config)]
pub struct InceptionBConfig {
    pub branch3x3: BasicConv2dConfig,

    pub branch3x3dbl_1: BasicConv2dConfig,
    pub branch3x3dbl_2: BasicConv2dConfig,
    pub branch3x3dbl_3: BasicConv2dConfig,

    pub max_pool: MaxPool2dConfig,
}

impl InceptionBConfig {
    pub fn default_with_channels(in_channels: usize) -> Self {
        Self {
            branch3x3: BasicConv2dConfig::create(in_channels, 384, 3).with_stride(2),

            branch3x3dbl_1: BasicConv2dConfig::create(in_channels, 64, 1),
            branch3x3dbl_2: BasicConv2dConfig::create(64, 96, 3).with_padding(1),
            branch3x3dbl_3: BasicConv2dConfig::create(96, 96, 3).with_stride(2),

            max_pool: MaxPool2dConfig::new([3, 3]).with_strides([2, 2]),
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> InceptionB<B> {
        InceptionB {
            branch3x3: self.branch3x3.init(device),

            branch3x3dbl_1: self.branch3x3dbl_1.init(device),
            branch3x3dbl_2: self.branch3x3dbl_2.init(device),
            branch3x3dbl_3: self.branch3x3dbl_3.init(device),

            max_pool: self.max_pool.init(),
        }
    }
}

#[derive(Debug, Module)]
pub struct InceptionB<B: Backend> {
    pub branch3x3: BasicConv2d<B>,

    pub branch3x3dbl_1: BasicConv2d<B>,
    pub branch3x3dbl_2: BasicConv2d<B>,
    pub branch3x3dbl_3: BasicConv2d<B>,

    pub max_pool: MaxPool2d,
}

impl<B: Backend> InceptionB<B> {
    pub fn forward_to_arr(&self, x: Tensor<B, 4>) -> [Tensor<B, 4>; 3] {
        let branch3x3 = self.branch3x3.forward(x.clone());

        let branch3x3dbl = self.branch3x3dbl_1.forward(x.clone());
        let branch3x3dbl = self.branch3x3dbl_2.forward(branch3x3dbl);
        let branch3x3dbl = self.branch3x3dbl_3.forward(branch3x3dbl);

        let branch_pool = self.max_pool.forward(x.clone());

        [branch3x3, branch3x3dbl, branch_pool]
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let output = self.forward_to_arr(x);
        Tensor::cat(output.to_vec(), 1)
    }
}
