# 🦀 Corroded Classifier

MNIST MLP implemented in Rust

To use:

`cargo build --release` (will be significantly slower if you just `cargo run`)
`./target/release/corroded_classifier`

Training _should_ look like:
```
❯ ./target/release/corroded_classifier
Starting training...
Epoch 1/5
[00:00:06] ========================================     390/390     Epoch avg loss: 0.0558 | Train Accuracy: 95.15% | Val Accuracy: 94.99%      Epoch 2/5
[00:00:06] ========================================     390/390     Epoch avg loss: 0.0293 | Train Accuracy: 96.57% | Val Accuracy: 95.67%      Epoch 3/5
[00:00:06] ========================================     390/390     Epoch avg loss: 0.0223 | Train Accuracy: 96.68% | Val Accuracy: 95.68%      Epoch 4/5
[00:00:06] ========================================     390/390     Epoch avg loss: 0.0181 | Train Accuracy: 98.01% | Val Accuracy: 96.91%      Epoch 5/5
[00:00:06] ========================================     390/390     Epoch avg loss: 0.0153 | Train Accuracy: 98.00% | Val Accuracy: 96.82%
```