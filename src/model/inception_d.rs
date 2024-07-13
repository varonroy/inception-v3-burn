use burn::prelude::*;
use nn::pool::{MaxPool2d, MaxPool2dConfig};

use super::basic_block::{BasicConv2d, BasicConv2dConfig};

#[derive(Debug, Config)]
pub struct InceptionDConfig {
    pub branch3x3_1: BasicConv2dConfig,
    pub branch3x3_2: BasicConv2dConfig,

    pub branch7x7x3_1: BasicConv2dConfig,
    pub branch7x7x3_2: BasicConv2dConfig,
    pub branch7x7x3_3: BasicConv2dConfig,
    pub branch7x7x3_4: BasicConv2dConfig,

    pub max_pool: MaxPool2dConfig,
}

impl InceptionDConfig {
    pub fn default_with_channels(in_channels: usize) -> Self {
        Self {
            branch3x3_1: BasicConv2dConfig::create(in_channels, 192, 1),
            branch3x3_2: BasicConv2dConfig::create(192, 320, 3).with_stride(2),

            branch7x7x3_1: BasicConv2dConfig::create(in_channels, 192, 1),
            branch7x7x3_2: BasicConv2dConfig::create2(192, 192, [1, 7]).with_padding2([0, 3]),
            branch7x7x3_3: BasicConv2dConfig::create2(192, 192, [7, 1]).with_padding2([3, 0]),
            branch7x7x3_4: BasicConv2dConfig::create(192, 192, 3).with_stride(2),

            max_pool: MaxPool2dConfig::new([3, 3]).with_strides([2, 2]),
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> InceptionD<B> {
        InceptionD {
            branch3x3_1: self.branch3x3_1.init(device),
            branch3x3_2: self.branch3x3_2.init(device),

            branch7x7x3_1: self.branch7x7x3_1.init(device),
            branch7x7x3_2: self.branch7x7x3_2.init(device),
            branch7x7x3_3: self.branch7x7x3_3.init(device),
            branch7x7x3_4: self.branch7x7x3_4.init(device),

            max_pool: self.max_pool.init(),
        }
    }
}

#[derive(Debug, Module)]
pub struct InceptionD<B: Backend> {
    pub branch3x3_1: BasicConv2d<B>,
    pub branch3x3_2: BasicConv2d<B>,

    pub branch7x7x3_1: BasicConv2d<B>,
    pub branch7x7x3_2: BasicConv2d<B>,
    pub branch7x7x3_3: BasicConv2d<B>,
    pub branch7x7x3_4: BasicConv2d<B>,

    pub max_pool: MaxPool2d,
}

impl<B: Backend> InceptionD<B> {
    pub fn forward_to_arr(&self, x: Tensor<B, 4>) -> [Tensor<B, 4>; 3] {
        let branch3x3 = self.branch3x3_1.forward(x.clone());
        let branch3x3 = self.branch3x3_2.forward(branch3x3);

        let branch7x7x3 = self.branch7x7x3_1.forward(x.clone());
        let branch7x7x3 = self.branch7x7x3_2.forward(branch7x7x3);
        let branch7x7x3 = self.branch7x7x3_3.forward(branch7x7x3);
        let branch7x7x3 = self.branch7x7x3_4.forward(branch7x7x3);

        let branch_pool = self.max_pool.forward(x);
        [branch3x3, branch7x7x3, branch_pool]
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let output = self.forward_to_arr(x);
        Tensor::cat(output.to_vec(), 1)
    }
}
