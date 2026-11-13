# Rust/Burn

Rust implementation of the two class classifier using [burn](https://burn.dev/). Build with `cargo build`, to run: `./target/debug/two_class_nn_burn ../two_class_nn/data/model.bin`. For building with `CUDA`, `cargo run --locked --release --features cuda -- --cuda`. See `-h` for usage; for example data, use [generate.py](../two_class_nn/data/generate.py).