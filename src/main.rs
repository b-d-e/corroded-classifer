use ndarray::s;
use indicatif::{ProgressBar, ProgressStyle};
use corroded_classifier::{
    network::network::Network,
    network::layer::Layer,
    network::activation::{ReLU, Sigmoid},
    data::mnist::MnistData,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Training parameters
    let batch_size = 128;
    let epochs = 5;
    let hidden_size = 128;
    let learning_rate = 0.1;

    // Load MNIST data
    let mnist = MnistData::new()?;

    // Create network
    let mut network = Network::new(learning_rate);

    // Add layers
    network.add_layer(Layer::new(784, hidden_size, Box::new(ReLU)));
    network.add_layer(Layer::new(hidden_size, 10, Box::new(Sigmoid)));


    println!("Starting training...");

    // Training loop
    for epoch in 0..epochs {
        println!("Epoch {}/{}", epoch + 1, epochs);
        let mut total_loss = 0.0;
        let num_batches = mnist.train_images.nrows() / batch_size;

        let progress_bar = ProgressBar::new(num_batches as u64);
        progress_bar.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .unwrap()
            .progress_chars("=>-"));

        for batch in 0..num_batches {
            let start = batch * batch_size;
            let (batch_images, batch_labels) = mnist.get_batch(start, batch_size);

            let mut batch_loss = 0.0;
            for i in 0..batch_size {
                let image = batch_images.slice(s![i, ..]);
                let label = batch_labels.slice(s![i, ..]);
                let loss = network.train_step(&image, &label);
                batch_loss += loss;
            }

            let avg_batch_loss = batch_loss / batch_size as f32;
            total_loss += batch_loss;

            // Calculate batch accuracy
            let batch_accuracy = network.calculate_accuracy(&batch_images, &batch_labels) * 100.0;

            progress_bar.set_message(format!("Loss: {:.4} | Acc: {:.2}%", avg_batch_loss, batch_accuracy));
            progress_bar.inc(1);
        }

        let epoch_loss = total_loss / (num_batches * batch_size) as f32;

        // Calculate epoch accuracy on whole training set
        let epoch_train_accuracy = network.calculate_accuracy(&mnist.train_images, &mnist.train_labels) * 100.0;

        let epoch_val_accuracy = network.calculate_accuracy(&mnist.test_images, &mnist.test_labels) * 100.0;

        progress_bar.finish_with_message(
            format!("Epoch avg loss: {:.4} | Train Accuracy: {:.2}% | Val Accuracy: {:.2}%", epoch_loss, epoch_train_accuracy, epoch_val_accuracy)
        );

        // Early stopping check
        if epoch_val_accuracy > 98.0 {
            println!("Reached target val accuracy of 98%! Stopping training.");
            break;
        }
    }

    Ok(())
}