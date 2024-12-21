pub mod dataset;  // The trait definition
pub mod mnist;
pub mod cifar;

pub use dataset::Dataset;
pub use mnist::MnistData;
pub use cifar::Cifar10Data;