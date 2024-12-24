use ndarray::{Array1, ArrayView1, Array2, s};
use super::layer::Layer;

pub struct Network {
    layers: Vec<Layer>,
    learning_rate: f32,
}

impl Network {
    pub fn new(learning_rate: f32) -> Self {
        Network {
            layers: Vec::new(),
            learning_rate,
        }
    }

    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    pub fn forward(&self, input: &ArrayView1<f32>) -> Array1<f32> {
        let mut current = input.to_owned();
        for layer in &self.layers {
            current = layer.forward(&current.view());
        }
        current
    }

    pub fn train_step(&mut self, input: &ArrayView1<f32>, target: &ArrayView1<f32>) -> f32 {
        // Forward pass storing activations
        let mut activations = Vec::with_capacity(self.layers.len() + 1);
        activations.push(input.to_owned());

        let mut current = input.to_owned();
        for layer in &self.layers {
            current = layer.forward(&current.view());
            activations.push(current.clone());
        }

        // Compute output error
        let output_error = &activations[activations.len() - 1] - target;
        let loss = output_error.mapv(|x| x * x).sum() / 2.0;

        // Backward pass
        let mut current_error = output_error;

        // iterate through layers in reverse
        // need access to the activations for each layer
        for (layer_idx, layer) in self.layers.iter_mut().enumerate().rev() {
            let input_activation = &activations[layer_idx];
            let output_activation = &activations[layer_idx + 1];

            // Get gradients and error for previous layer
            let (weight_gradients, bias_gradients, prev_error) =
                layer.backward(&input_activation.view(), &output_activation.view(), &current_error.view());

            // Update weights and biases
            layer.weights -= &(weight_gradients * self.learning_rate);
            layer.biases -= &(bias_gradients * self.learning_rate);

            // Update error for next iteration
            current_error = prev_error;
        }

        loss
    }

    pub fn calculate_accuracy(&self, images: &Array2<f32>, labels: &Array2<f32>) -> f32 {
        let mut correct = 0;
        let total = images.nrows();

        for i in 0..total {
            let image = images.slice(s![i, ..]);
            let label = labels.slice(s![i, ..]);

            let output = self.forward(&image.view());

            // Get predicted class (max index)
            let pred_idx = output.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap();

            // Get true class
            let true_idx = label.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap();

            if pred_idx == true_idx {
                correct += 1;
            }
        }

        correct as f32 / total as f32
    }
}