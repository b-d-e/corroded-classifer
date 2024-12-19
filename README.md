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
[00:00:06] ========================================     390/390     Epoch avg loss: 0.0549 | Train Accuracy: 96.26% | Val Accuracy: 95.86%
Epoch 2/5
[00:00:06] ========================================     390/390     Epoch avg loss: 0.0291 | Train Accuracy: 97.11% | Val Accuracy: 96.44%
Epoch 3/5
[00:00:06] ========================================     390/390     Epoch avg loss: 0.0221 | Train Accuracy: 97.90% | Val Accuracy: 96.92%
Epoch 4/5
[00:00:06] ========================================     390/390     Epoch avg loss: 0.0183 | Train Accuracy: 97.99% | Val Accuracy: 96.89%
Epoch 5/5
[00:00:06] ========================================     390/390     Epoch avg loss: 0.0149 | Train Accuracy: 98.44% | Val Accuracy: 97.24%
```
and log some ascii graphs at the end.