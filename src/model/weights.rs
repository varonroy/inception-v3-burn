use std::path::PathBuf;

/// Pre-trained weights metadata.
#[derive(Debug, Clone)]
pub struct WeightsSource {
    pub file_path: PathBuf,
    pub num_classes: usize,
    pub fid_configuration: bool,
}

#[cfg(feature = "pretrained")]
pub mod downloader {
    use super::*;
    use burn::data::network::downloader;
    use std::fs::{create_dir_all, File};
    use std::io::Write;
    use std::path::{Path, PathBuf};

    const IMAGENET_1K_V1_URL: &'static str =
        "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth";

    // const FID_URL: &'static str = "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth";

    /// Download the pre-trained weights to a specific directory.
    pub fn download_to(url: &str, model_dir: impl AsRef<Path>) -> Result<PathBuf, std::io::Error> {
        let model_dir = model_dir.as_ref();

        if !model_dir.exists() {
            create_dir_all(&model_dir)?;
        }

        let file_base_name = url.rsplit_once('/').unwrap().1;
        let file_name = model_dir.join(file_base_name);
        if !file_name.exists() {
            // Download file content
            let bytes = downloader::download_file_as_bytes(url, file_base_name);

            // Write content to file
            let mut output_file = File::create(&file_name)?;
            let bytes_written = output_file.write(&bytes)?;

            if bytes_written != bytes.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Failed to write the whole model weights file.",
                ));
            }
        }

        Ok(file_name)
    }

    /// Download the pre-trained weights to the local cache directory.
    pub fn download(url: &str) -> Result<PathBuf, std::io::Error> {
        let model_dir = dirs::home_dir()
            .expect("Should be able to get home directory")
            .join(".cache")
            .join("inception-v3-burn");
        download_to(url, model_dir)
    }

    pub trait InceptionV3PretrainedLoader {
        /// (source)[https://pytorch.org/vision/main/models/generated/torchvision.models.inception_v3.html#torchvision.models.Inception_V3_Weights] weights.
        /// metrics:
        /// - ImageNet-1K:
        ///   - acc@1: 77.294
        ///   - acc@5: 93.450
        /// file size: 103.903
        fn imagenet1k_download() -> WeightsSource;

        /// (source)[https://pytorch.org/vision/main/models/generated/torchvision.models.inception_v3.html#torchvision.models.Inception_V3_Weights] weights.
        /// metrics:
        /// - ImageNet-1K:
        ///   - acc@1: 77.294
        ///   - acc@5: 93.450
        /// file size: 103.903
        fn imagenet1k(path: impl Into<PathBuf>) -> WeightsSource;

        /// Weights taken from the [`mseitzer/pytorch-fid`](https://github.com/mseitzer/pytorch-fid/)
        /// repository.
        /// These weights need to be converted using PyTorch before they can be used,
        fn fid(path: Option<&Path>) -> WeightsSource;
    }

    impl InceptionV3PretrainedLoader for WeightsSource {
        fn imagenet1k_download() -> WeightsSource {
            WeightsSource {
                file_path: download(IMAGENET_1K_V1_URL).unwrap(),
                num_classes: 1000,
                fid_configuration: false,
            }
        }

        fn imagenet1k(path: impl Into<PathBuf>) -> WeightsSource {
            WeightsSource {
                file_path: path.into(),
                num_classes: 1000,
                fid_configuration: false,
            }
        }

        fn fid(path: Option<&Path>) -> WeightsSource {
            WeightsSource {
                file_path: path.map(|path| path.to_path_buf()).unwrap_or_else(|| {
                    dirs::home_dir()
                        .expect("Should be able to get home directory")
                        .join(".cache")
                        .join("inception-v3-burn")
                        .join("pt_inception-2015-12-05-6726825d.pth")
                }),
                num_classes: 1008,
                fid_configuration: false,
            }
        }
    }
}
