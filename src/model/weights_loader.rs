use {
    super::InceptionV3Record,
    burn::{
        prelude::*,
        record::{FullPrecisionSettings, Recorder, RecorderError},
        tensor::Device,
    },
    burn_import::pytorch::{LoadArgs, PyTorchFileRecorder},
};

/// Load specified pre-trained PyTorch weights as a record.
pub(super) fn load_weights_record<B: Backend>(
    weights: &super::weights::WeightsSource,
    device: &Device<B>,
) -> Result<InceptionV3Record<B>, RecorderError> {
    let load_args = LoadArgs::new(weights.file_path.clone())
        // Convert `Conv2d` to lowercase
        .with_key_remap("Conv2d(.+)", "conv2d$1")
        // Convert `Mixed` to lowercase
        .with_key_remap("Mixed(.+)", "mixed$1")
        // Convert `AuxLogits` to lowercase
        .with_key_remap("AuxLogits(.+)", "aux_logits$1");

    let record = PyTorchFileRecorder::<FullPrecisionSettings>::new().load(load_args, device)?;

    Ok(record)
}
