use ndarray::Array2;

pub trait Dataset {
    fn new() -> Result<Self, Box<dyn std::error::Error>> where Self: Sized;
    fn get_batch(&self, start: usize, batch_size: usize) -> (Array2<f32>, Array2<f32>);
    fn get_train_size(&self) -> usize;
    fn get_test_size(&self) -> usize;
    fn get_num_classes(&self) -> usize;
    fn get_input_size(&self) -> usize;
}