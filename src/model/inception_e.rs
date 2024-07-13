use burn::prelude::*;
use nn::pool::{AvgPool2d, AvgPool2dConfig};

use super::basic_block::{BasicConv2d, BasicConv2dConfig};

#[derive(Debug, Config)]
pub struct InceptionEConfig {
    pub branch1x1: BasicConv2dConfig,

    pub branch3x3_1: BasicConv2dConfig,
    pub branch3x3_2a: BasicConv2dConfig,
    pub branch3x3_2b: BasicConv2dConfig,

    pub branch3x3dbl_1: BasicConv2dConfig,
    pub branch3x3dbl_2: BasicConv2dConfig,
    pub branch3x3dbl_3a: BasicConv2dConfig,
    pub branch3x3dbl_3b: BasicConv2dConfig,

    pub avg_pool: AvgPool2dConfig,
    pub branch_pool: BasicConv2dConfig,
}

impl InceptionEConfig {
    pub fn default_with_channels(in_channels: usize) -> Self {
        Self {
            branch1x1: BasicConv2dConfig::create(in_channels, 320, 1),

            branch3x3_1: BasicConv2dConfig::create(in_channels, 384, 1),
            branch3x3_2a: BasicConv2dConfig::create2(384, 384, [1, 3]).with_padding2([0, 1]),
            branch3x3_2b: BasicConv2dConfig::create2(384, 384, [3, 1]).with_padding2([1, 0]),

            branch3x3dbl_1: BasicConv2dConfig::create(in_channels, 448, 1),
            branch3x3dbl_2: BasicConv2dConfig::create(448, 384, 3).with_padding(1),
            branch3x3dbl_3a: BasicConv2dConfig::create2(384, 384, [1, 3]).with_padding2([0, 1]),
            branch3x3dbl_3b: BasicConv2dConfig::create2(384, 384, [3, 1]).with_padding2([1, 0]),

            avg_pool: AvgPool2dConfig::new([3, 3])
                .with_strides([1, 1])
                .with_padding(nn::PaddingConfig2d::Explicit(1, 1)),
            branch_pool: BasicConv2dConfig::create(in_channels, 192, 1),
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> InceptionE<B> {
        InceptionE {
            branch1x1: self.branch1x1.init(device),

            branch3x3_1: self.branch3x3_1.init(device),
            branch3x3_2a: self.branch3x3_2a.init(device),
            branch3x3_2b: self.branch3x3_2b.init(device),

            branch3x3dbl_1: self.branch3x3dbl_1.init(device),
            branch3x3dbl_2: self.branch3x3dbl_2.init(device),
            branch3x3dbl_3a: self.branch3x3dbl_3a.init(device),
            branch3x3dbl_3b: self.branch3x3dbl_3b.init(device),

            avg_pool: self.avg_pool.init(),
            branch_pool: self.branch_pool.init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct InceptionE<B: Backend> {
    pub branch1x1: BasicConv2d<B>,

    pub branch3x3_1: BasicConv2d<B>,
    pub branch3x3_2a: BasicConv2d<B>,
    pub branch3x3_2b: BasicConv2d<B>,

    pub branch3x3dbl_1: BasicConv2d<B>,
    pub branch3x3dbl_2: BasicConv2d<B>,
    pub branch3x3dbl_3a: BasicConv2d<B>,
    pub branch3x3dbl_3b: BasicConv2d<B>,

    pub avg_pool: AvgPool2d,
    pub branch_pool: BasicConv2d<B>,
}

impl<B: Backend> InceptionE<B> {
    pub fn forward_to_arr(&self, x: Tensor<B, 4>) -> [Tensor<B, 4>; 4] {
        let branch1x1 = self.branch1x1.forward(x.clone());

        let branch3x3 = self.branch3x3_1.forward(x.clone());
        let branch3x3 = [
            self.branch3x3_2a.forward(branch3x3.clone()),
            self.branch3x3_2b.forward(branch3x3.clone()),
        ];
        let branch3x3 = Tensor::cat(branch3x3.to_vec(), 1);

        let branch3x3dbl = self.branch3x3dbl_1.forward(x.clone());
        let branch3x3dbl = self.branch3x3dbl_2.forward(branch3x3dbl);
        let branch3x3dbl = [
            self.branch3x3dbl_3a.forward(branch3x3dbl.clone()),
            self.branch3x3dbl_3b.forward(branch3x3dbl.clone()),
        ];
        let branch3x3dbl = Tensor::cat(branch3x3dbl.to_vec(), 1);

        let branch_pool = self.avg_pool.forward(x);
        let branch_pool = self.branch_pool.forward(branch_pool);

        [branch1x1, branch3x3, branch3x3dbl, branch_pool]
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let output = self.forward_to_arr(x);
        Tensor::cat(output.to_vec(), 1)
    }
}
