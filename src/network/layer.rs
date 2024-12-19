use ndarray::{Array1, Array2, ArrayView1};
use rand_distr::{Distribution, Normal}; // Changed from rand::distributions
use rand::thread_rng;
use super::activation::Activation;

pub struct Layer {
    pub weights: Array2<f32>,  // Made public
    pub biases: Array1<f32>,   // Made public
    activation: Box<dyn Activation>,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, activation: Box<dyn Activation>) -> Self {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, (1.0 / input_size as f32).sqrt())
            .expect("Failed to create normal distribution");

        let weights = Array2::from_shape_fn((output_size, input_size), |_| {
            normal.sample(&mut rng)
        });

        let biases = Array1::zeros(output_size);

        Layer {
            weights,
            biases,
            activation,
        }
    }

    pub fn forward(&self, input: &ArrayView1<f32>) -> Array1<f32> {
        let preactivation = self.weights.dot(input) + &self.biases;
        preactivation.mapv(|x| self.activation.forward(x))
    }

    pub fn backward(&self, input: &ArrayView1<f32>, output: &ArrayView1<f32>, error: &ArrayView1<f32>)
        -> (Array2<f32>, Array1<f32>, Array1<f32>) {
        let derivative = output.mapv(|x| self.activation.derivative(x));
        let delta = error * &derivative;

        let grad_weights = Array2::from_shape_fn((self.weights.nrows(), self.weights.ncols()), |(i, j)| {
            delta[i] * input[j]
        });

        let grad_biases = delta.clone();
        let prev_error = self.weights.t().dot(&delta);

        (grad_weights, grad_biases, prev_error)
    }
}