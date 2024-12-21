use ndarray::{Array2, s};
use mnist::MnistBuilder;
use std::path::Path;
use super::dataset::Dataset;

pub struct MnistData {
    pub train_images: Array2<f32>,
    pub train_labels: Array2<f32>,
    pub test_images: Array2<f32>,
    pub test_labels: Array2<f32>,
}

impl Dataset for MnistData {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        if !Path::new("data").exists() {
            std::fs::create_dir("data")?;
        }

        // Download the dataset if needed
        if !Path::new("data/train-images-idx3-ubyte").exists() {
            println!("Downloading MNIST dataset...");
            std::process::Command::new("curl")
                .args(&[
                    "-O",
                    "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
                ])
                .output()?;
            std::process::Command::new("curl")
                .args(&[
                    "-O",
                    "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
                ])
                .output()?;
            std::process::Command::new("curl")
                .args(&[
                    "-O",
                    "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
                ])
                .output()?;
            std::process::Command::new("curl")
                .args(&[
                    "-O",
                    "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
                ])
                .output()?;

            // Unzip files
            for file in &[
                "train-images-idx3-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
                "t10k-images-idx3-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz",
            ] {
                std::process::Command::new("gunzip")
                    .arg("-f")
                    .arg(file)
                    .output()?;
            }

            // Move files to data directory
            for file in &[
                "train-images-idx3-ubyte",
                "train-labels-idx1-ubyte",
                "t10k-images-idx3-ubyte",
                "t10k-labels-idx1-ubyte",
            ] {
                std::fs::rename(file, format!("data/{}", file))?;
            }
        }

        let mnist = MnistBuilder::new()
            .label_format_one_hot()
            .training_set_length(50_000)
            .validation_set_length(10_000)
            .test_set_length(10_000)
            .finalize();

        let train_images = Array2::from_shape_vec(
            (50_000, 784),
            mnist.trn_img.iter().map(|&x| x as f32 / 255.0).collect()
        )?;

        let train_labels = Array2::from_shape_vec(
            (50_000, 10),
            mnist.trn_lbl.iter().map(|&x| x as f32).collect()
        )?;

        let test_images = Array2::from_shape_vec(
            (10_000, 784),
            mnist.tst_img.iter().map(|&x| x as f32 / 255.0).collect()
        )?;

        let test_labels = Array2::from_shape_vec(
            (10_000, 10),
            mnist.tst_lbl.iter().map(|&x| x as f32).collect()
        )?;

        Ok(MnistData {
            train_images,
            train_labels,
            test_images,
            test_labels,
        })
    }

    fn get_batch(&self, start: usize, batch_size: usize) -> (Array2<f32>, Array2<f32>) {
        let end = start + batch_size;
        let batch_images = self.train_images.slice(s![start..end, ..]).to_owned();
        let batch_labels = self.train_labels.slice(s![start..end, ..]).to_owned();
        (batch_images, batch_labels)
    }

    fn get_train_size(&self) -> usize { 50_000 }
    fn get_test_size(&self) -> usize { 10_000 }
    fn get_input_size(&self) -> usize { 784 }
    fn get_num_classes(&self) -> usize { 10 }
}