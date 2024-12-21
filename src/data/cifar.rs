use ndarray::{Array2, s};
use std::path::Path;
use super::dataset::Dataset;

pub struct Cifar10Data {
    pub train_images: Array2<f32>,
    pub train_labels: Array2<f32>,
    pub test_images: Array2<f32>,
    pub test_labels: Array2<f32>,
}

impl Dataset for Cifar10Data {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        if !Path::new("data").exists() {
            std::fs::create_dir("data")?;
        }

        let base_url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";

        // Download and extract CIFAR-10
        if !Path::new("data/cifar-10-batches-bin").exists() {
            println!("Downloading CIFAR-10 dataset...");
            std::process::Command::new("curl")
                .args(&["-O", base_url])
                .output()?;

            std::process::Command::new("tar")
                .args(&["-xzf", "cifar-10-binary.tar.gz"])
                .output()?;

            std::fs::rename("cifar-10-batches-bin", "data/cifar-10-batches-bin")?;
        }

        // Load training data (5 batches)
        let mut train_images = Vec::new();
        let mut train_labels = Vec::new();

        for i in 1..=5 {
            let path = format!("data/cifar-10-batches-bin/data_batch_{}.bin", i);
            let data = std::fs::read(path)?;

            for chunk in data.chunks(3073) {
                let label = chunk[0] as usize;
                let pixels = &chunk[1..];

                let mut one_hot = vec![0.0; 10];
                one_hot[label] = 1.0;
                train_labels.extend(one_hot);
                train_images.extend(pixels.iter().map(|&x| x as f32 / 255.0));
            }
        }

        // Load test data
        let test_data = std::fs::read("data/cifar-10-batches-bin/test_batch.bin")?;
        let mut test_images = Vec::new();
        let mut test_labels = Vec::new();

        for chunk in test_data.chunks(3073) {
            let label = chunk[0] as usize;
            let pixels = &chunk[1..];

            let mut one_hot = vec![0.0; 10];
            one_hot[label] = 1.0;
            test_labels.extend(one_hot);
            test_images.extend(pixels.iter().map(|&x| x as f32 / 255.0));
        }

        Ok(Cifar10Data {
            train_images: Array2::from_shape_vec((50_000, 3072), train_images)?,
            train_labels: Array2::from_shape_vec((50_000, 10), train_labels)?,
            test_images: Array2::from_shape_vec((10_000, 3072), test_images)?,
            test_labels: Array2::from_shape_vec((10_000, 10), test_labels)?,
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
    fn get_input_size(&self) -> usize { 3072 }
    fn get_num_classes(&self) -> usize { 10 }
}