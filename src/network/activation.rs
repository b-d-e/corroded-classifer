use std::f32;

pub trait Activation {
    fn forward(&self, x: f32) -> f32;
    fn derivative(&self, x: f32) -> f32;
}

/// ReLU activation function
#[derive(Debug)]
pub struct ReLU;

impl Activation for ReLU {
    fn forward(&self, x: f32) -> f32 {
        x.max(0.0)
    }

    fn derivative(&self, x: f32) -> f32 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
}

/// Sigmoid activation function
#[derive(Debug)]
pub struct Sigmoid;

impl Activation for Sigmoid {
    fn forward(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derivative(&self, x: f32) -> f32 {
        let sig = self.forward(x);
        sig * (1.0 - sig)
    }
}