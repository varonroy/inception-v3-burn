use burn::prelude::*;
use nn::pool::{AvgPool2d, AvgPool2dConfig};

use super::basic_block::{BasicConv2d, BasicConv2dConfig};

#[derive(Debug, Config)]
pub struct InceptionAConfig {
    pub branch1x1: BasicConv2dConfig,

    pub branch5x5_1: BasicConv2dConfig,
    pub branch5x5_2: BasicConv2dConfig,

    pub branch3x3dbl_1: BasicConv2dConfig,
    pub branch3x3dbl_2: BasicConv2dConfig,
    pub branch3x3dbl_3: BasicConv2dConfig,

    pub avg_pool2d: AvgPool2dConfig,
    pub branch_pool: BasicConv2dConfig,
}

impl InceptionAConfig {
    pub fn default_with_channels(in_channels: usize, pool_features: usize) -> Self {
        Self {
            branch1x1: BasicConv2dConfig::create(in_channels, 64, 1),

            branch5x5_1: BasicConv2dConfig::create(in_channels, 48, 1),
            branch5x5_2: BasicConv2dConfig::create(48, 64, 5).with_padding(2),

            branch3x3dbl_1: BasicConv2dConfig::create(in_channels, 64, 1),
            branch3x3dbl_2: BasicConv2dConfig::create(64, 96, 3).with_padding(1),
            branch3x3dbl_3: BasicConv2dConfig::create(96, 96, 3).with_padding(1),

            avg_pool2d: AvgPool2dConfig::new([3, 3])
                .with_strides([1, 1])
                .with_padding(nn::PaddingConfig2d::Explicit(1, 1)),
            branch_pool: BasicConv2dConfig::create(in_channels, pool_features, 1),
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> InceptionA<B> {
        InceptionA {
            branch1x1: self.branch1x1.init(device),

            branch5x5_1: self.branch5x5_1.init(device),
            branch5x5_2: self.branch5x5_2.init(device),

            branch3x3dbl_1: self.branch3x3dbl_1.init(device),
            branch3x3dbl_2: self.branch3x3dbl_2.init(device),
            branch3x3dbl_3: self.branch3x3dbl_3.init(device),

            avg_pool2d: self.avg_pool2d.init(),
            branch_pool: self.branch_pool.init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct InceptionA<B: Backend> {
    pub branch1x1: BasicConv2d<B>,

    pub branch5x5_1: BasicConv2d<B>,
    pub branch5x5_2: BasicConv2d<B>,

    pub branch3x3dbl_1: BasicConv2d<B>,
    pub branch3x3dbl_2: BasicConv2d<B>,
    pub branch3x3dbl_3: BasicConv2d<B>,

    pub branch_pool: BasicConv2d<B>,
    pub avg_pool2d: AvgPool2d,
}

impl<B: Backend> InceptionA<B> {
    pub fn forward_to_arr(&self, x: Tensor<B, 4>) -> [Tensor<B, 4>; 4] {
        let branch1x1 = self.branch1x1.forward(x.clone());

        let branch5x5 = self.branch5x5_1.forward(x.clone());
        let branch5x5 = self.branch5x5_2.forward(branch5x5);

        let branch3x3dbl = self.branch3x3dbl_1.forward(x.clone());
        let branch3x3dbl = self.branch3x3dbl_2.forward(branch3x3dbl);
        let branch3x3dbl = self.branch3x3dbl_3.forward(branch3x3dbl);

        let branch_pool = self.avg_pool2d.forward(x.clone());
        let branch_pool = self.branch_pool.forward(branch_pool);

        [branch1x1, branch5x5, branch3x3dbl, branch_pool]
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let output = self.forward_to_arr(x);
        Tensor::cat(output.to_vec(), 1)
    }
}
