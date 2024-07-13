// use burn::prelude::*;
//
// use super::{
//     basic_block::{BasicConv2d, BasicConv2dConfig},
//     inception_a::{InceptionA, InceptionAConfig},
//     inception_aux::InceptionAuxConfig,
//     inception_b::InceptionCConfig,
//     inception_c::InceptionBConfig,
//     inception_d::InceptionDConfig,
//     inception_e::InceptionEConfig,
// };
//
// #[derive(Debug, Clone, Copy)]
// pub enum BlockTy {
//     BasicConv2d,
//     InceptionA,
//     InceptionB,
//     InceptionC,
//     InceptionD,
//     InceptionE,
//     InceptionAux,
// }
//
// impl BlockTy {
//     pub fn create_config(self, in_channels: usize, pool_features: usize) -> BlockConfig {
//         let i = in_channels;
//         let p = pool_features;
//         match self {
//             Self::BasicConv2d => {
//                 BlockConfig::BasicConv2d(Box::new(BasicConv2dConfig::create(i, p)))
//             }
//             Self::InceptionA => {
//                 BlockConfig::InceptionA(InceptionAConfig::default_with_channels(i, p))
//             }
//             Self::InceptionB => InceptionBConfig::default_with_channels(i),
//             Self::InceptionC => InceptionCConfig::default_with_channels(i, p),
//             Self::InceptionD => InceptionDConfig::default_with_channels(i),
//             Self::InceptionE => InceptionEConfig::default_with_channels(i),
//             Self::InceptionAux => InceptionAuxConfig::default_with_channels(i, p),
//         }
//     }
// }
//
// #[derive(Debug, Config)]
// pub enum BlockConfig {
//     BasicConv2d(Box<BasicConv2dConfig>),
//     // InceptionA(Box<InceptionAConfig>),
//     // InceptionB(Box<InceptionBConfig>),
//     // InceptionC(Box<InceptionCConfig>),
//     // InceptionD(Box<InceptionDConfig>),
//     // InceptionE(Box<InceptionEConfig>),
//     // InceptionAux(Box<InceptionAuxConfig>),
// }
//
// impl BlockConfig {
//     pub fn init<B: Backend>(&self, device: &B::Device) -> Block<B> {
//         match self {
//             Self::BasicConv2d(config) => Block::BasicConv2d(Box::new(config.init(device))),
//             // Self::InceptionA(config) => Block::InceptionA(Box::new(config.init(device))),
//             // Self::InceptionB(config) => Block::InceptionB(Box::new(config.init(device))),
//             // Self::InceptionC(config) => Block::InceptionC(Box::new(config.init(device))),
//             // Self::InceptionD(config) => Block::InceptionD(Box::new(config.init(device))),
//             // Self::InceptionE(config) => Block::InceptionE(Box::new(config.init(device))),
//             // Self::InceptionAux(config) => Block::InceptionAux(Box::new(config.init(device))),
//         }
//     }
// }
//
// #[derive(Debug, Module)]
// pub enum Block<B: Backend> {
//     BasicConv2d(BasicConv2d<B>),
//     // InceptionA(InceptionA<B>),
//     // InceptionB(InceptionB<B>),
//     // InceptionC(InceptionC<B>),
//     // InceptionD(InceptionD<B>),
//     // InceptionE(InceptionE<B>),
//     // InceptionAux(InceptionAux<B>),
// }
//
// impl<B: Backend> Block<B> {
//     pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
//         match self {
//             Self::BasicConv2d(layer) => layer.forward(x),
//             // Self::InceptionA(layer) => layer.forward(x),
//             //Self::  InceptionB(layer) => layer.forward(x),
//             //Self::  InceptionC(layer) => layer.forward(x),
//             //Self::  InceptionD(layer) => layer.forward(x),
//             //Self::  InceptionE(layer) => layer.forward(x),
//             //Self::  InceptionAux(layer) => layer.forward(x),
//         }
//     }
// }
