use burn::prelude::*;
use nn::pool::{AvgPool2d, AvgPool2dConfig};

use super::basic_block::{BasicConv2d, BasicConv2dConfig};

#[derive(Debug, Config)]
pub struct InceptionCConfig {
    pub branch1x1: BasicConv2dConfig,

    pub branch7x7_1: BasicConv2dConfig,
    pub branch7x7_2: BasicConv2dConfig,
    pub branch7x7_3: BasicConv2dConfig,

    pub branch7x7dbl_1: BasicConv2dConfig,
    pub branch7x7dbl_2: BasicConv2dConfig,
    pub branch7x7dbl_3: BasicConv2dConfig,
    pub branch7x7dbl_4: BasicConv2dConfig,
    pub branch7x7dbl_5: BasicConv2dConfig,

    pub avg_pool: AvgPool2dConfig,
    pub branch_pool: BasicConv2dConfig,
}

impl InceptionCConfig {
    pub fn default_with_channels(in_channels: usize, channels_7x7: usize) -> Self {
        let c7 = channels_7x7;

        Self {
            branch1x1: BasicConv2dConfig::create(in_channels, 192, 1),

            branch7x7_1: BasicConv2dConfig::create(in_channels, c7, 1),
            branch7x7_2: BasicConv2dConfig::create2(c7, c7, [1, 7]).with_padding2([0, 3]),
            branch7x7_3: BasicConv2dConfig::create2(c7, 192, [7, 1]).with_padding2([3, 0]),

            branch7x7dbl_1: BasicConv2dConfig::create(in_channels, c7, 1),
            branch7x7dbl_2: BasicConv2dConfig::create2(c7, c7, [7, 1]).with_padding2([3, 0]),
            branch7x7dbl_3: BasicConv2dConfig::create2(c7, c7, [1, 7]).with_padding2([0, 3]),
            branch7x7dbl_4: BasicConv2dConfig::create2(c7, c7, [7, 1]).with_padding2([3, 0]),
            branch7x7dbl_5: BasicConv2dConfig::create2(c7, 192, [1, 7]).with_padding2([0, 3]),

            avg_pool: AvgPool2dConfig::new([3, 3])
                .with_strides([1, 1])
                .with_padding(nn::PaddingConfig2d::Explicit(1, 1)),
            branch_pool: BasicConv2dConfig::create(in_channels, 192, 1),
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> InceptionC<B> {
        InceptionC {
            branch1x1: self.branch1x1.init(device),

            branch7x7_1: self.branch7x7_1.init(device),
            branch7x7_2: self.branch7x7_2.init(device),
            branch7x7_3: self.branch7x7_3.init(device),

            branch7x7dbl_1: self.branch7x7dbl_1.init(device),
            branch7x7dbl_2: self.branch7x7dbl_2.init(device),
            branch7x7dbl_3: self.branch7x7dbl_3.init(device),
            branch7x7dbl_4: self.branch7x7dbl_4.init(device),
            branch7x7dbl_5: self.branch7x7dbl_5.init(device),

            avg_pool: self.avg_pool.init(),
            branch_pool: self.branch_pool.init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct InceptionC<B: Backend> {
    pub branch1x1: BasicConv2d<B>,

    pub branch7x7_1: BasicConv2d<B>,
    pub branch7x7_2: BasicConv2d<B>,
    pub branch7x7_3: BasicConv2d<B>,

    pub branch7x7dbl_1: BasicConv2d<B>,
    pub branch7x7dbl_2: BasicConv2d<B>,
    pub branch7x7dbl_3: BasicConv2d<B>,
    pub branch7x7dbl_4: BasicConv2d<B>,
    pub branch7x7dbl_5: BasicConv2d<B>,

    pub avg_pool: AvgPool2d,
    pub branch_pool: BasicConv2d<B>,
}

impl<B: Backend> InceptionC<B> {
    pub fn forward_to_arr(&self, x: Tensor<B, 4>) -> [Tensor<B, 4>; 4] {
        let branch1x1 = self.branch1x1.forward(x.clone());

        let branch7x7 = self.branch7x7_1.forward(x.clone());
        let branch7x7 = self.branch7x7_2.forward(branch7x7);
        let branch7x7 = self.branch7x7_3.forward(branch7x7);

        let branch7x7dbl = self.branch7x7dbl_1.forward(x.clone());
        let branch7x7dbl = self.branch7x7dbl_2.forward(branch7x7dbl);
        let branch7x7dbl = self.branch7x7dbl_3.forward(branch7x7dbl);
        let branch7x7dbl = self.branch7x7dbl_4.forward(branch7x7dbl);
        let branch7x7dbl = self.branch7x7dbl_5.forward(branch7x7dbl);

        let branch_pool = self.avg_pool.forward(x);
        let branch_pool = self.branch_pool.forward(branch_pool);

        [branch1x1, branch7x7, branch7x7dbl, branch_pool]
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let output = self.forward_to_arr(x);
        Tensor::cat(output.to_vec(), 1)
    }
}
