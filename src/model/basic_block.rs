use burn::prelude::*;
use nn::{
    conv::{Conv2d, Conv2dConfig},
    BatchNorm, BatchNormConfig, Relu,
};

#[derive(Debug, Config)]
pub struct BasicConv2dConfig {
    pub conv: Conv2dConfig,
    pub bn: BatchNormConfig,
}

impl BasicConv2dConfig {
    pub fn create(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self {
            conv: Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
                .with_bias(false),
            bn: BatchNormConfig::new(out_channels).with_epsilon(0.001),
        }
    }

    pub fn create2(in_channels: usize, out_channels: usize, kernel_size: [usize; 2]) -> Self {
        Self {
            conv: Conv2dConfig::new([in_channels, out_channels], kernel_size).with_bias(false),
            bn: BatchNormConfig::new(out_channels).with_epsilon(0.001),
        }
    }

    pub fn with_padding(mut self, padding: usize) -> Self {
        self.conv = self
            .conv
            .with_padding(nn::PaddingConfig2d::Explicit(padding, padding));
        self
    }

    pub fn with_padding2(mut self, padding: [usize; 2]) -> Self {
        self.conv = self
            .conv
            .with_padding(nn::PaddingConfig2d::Explicit(padding[0], padding[1]));
        self
    }

    pub fn with_stride(mut self, stride: usize) -> Self {
        self.conv = self.conv.with_stride([stride, stride]);
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> BasicConv2d<B> {
        BasicConv2d {
            conv: self.conv.init(device),
            bn: self.bn.init(device),
            relu: Relu::new(),
        }
    }
}

#[derive(Debug, Module)]
pub struct BasicConv2d<B: Backend> {
    pub conv: Conv2d<B>,
    pub bn: BatchNorm<B, 2>,
    pub relu: Relu,
}

impl<B: Backend> BasicConv2d<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = self.bn.forward(x);
        let x = self.relu.forward(x);
        x
    }
}
