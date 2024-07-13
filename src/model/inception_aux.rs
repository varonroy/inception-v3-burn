use burn::prelude::*;
use nn::{
    pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, AvgPool2d, AvgPool2dConfig},
    Linear, LinearConfig,
};

use super::basic_block::{BasicConv2d, BasicConv2dConfig};

// TODO: implement debug when `AdaptiveAvgPool2dConfig` also implements it
#[derive(Config)]
pub struct InceptionAuxConfig {
    pub conv0: BasicConv2dConfig,
    pub conv1: BasicConv2dConfig,
    pub fc: LinearConfig,
    pub avg_pool: AvgPool2dConfig,
    pub adaptive_avg_pool: AdaptiveAvgPool2dConfig,
}

impl InceptionAuxConfig {
    pub fn default_with_channels(in_channels: usize, num_classes: usize) -> Self {
        Self {
            conv0: BasicConv2dConfig::create(in_channels, 128, 1),
            conv1: BasicConv2dConfig::create(128, 768, 5),
            fc: LinearConfig::new(768, num_classes),
            avg_pool: AvgPool2dConfig::new([5, 5]).with_strides([3, 3]),
            adaptive_avg_pool: AdaptiveAvgPool2dConfig::new([1, 1]),
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> InceptionAux<B> {
        InceptionAux {
            conv0: self.conv0.init(device),
            conv1: self.conv1.init(device),
            fc: self.fc.init(device),

            avg_pool: self.avg_pool.init(),
            adaptive_avg_pool: self.adaptive_avg_pool.init(),
        }
    }
}

#[derive(Debug, Module)]
pub struct InceptionAux<B: Backend> {
    conv0: BasicConv2d<B>,
    conv1: BasicConv2d<B>,
    fc: Linear<B>,
    avg_pool: AvgPool2d,
    adaptive_avg_pool: AdaptiveAvgPool2d,
}

impl<B: Backend> InceptionAux<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        // N x 768 x 17 x 17
        let x = self.avg_pool.forward(x);
        // N x 768 x 5 x 5
        let x = self.conv0.forward(x);
        // N x 128 x 5 x 5
        let x = self.conv1.forward(x);
        // N x 768 x 1 x 1
        // Adaptive average pooling
        let x = self.adaptive_avg_pool.forward(x);
        // N x 768 x 1 x 1
        let x = x.flatten::<2>(1, 3);
        // N x 768
        let x = self.fc.forward(x);
        // N x 1000
        x
    }
}
