mod data;

use std::{env, fs, path::PathBuf};

use burn::nn::Linear;
use burn::nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig};
use burn::optim::{GradientsParams, SgdConfig};
use burn::prelude::*;
use burn::tensor::activation::relu;
use data::{Batch, Dataset, Result};

//const MODEL_DATA: &[u8] = include_bytes!("../data/model.bin");


const USAGE: &str = "
    Usage: two_class_nn_burn [--cuda] [model.bin] [epochs] [learning_rate]
        Defaults: CPU, bundled model.bin, 500 epochs, learning rate 0.3";

struct Options {
    cuda: bool,
    path: Option<PathBuf>,
    epochs: usize,
    learning_rate: f64,
}

impl Options {
    fn parse(args: Vec<String>) -> Result<Self> {
        let cuda = args.first().is_some_and(|arg| arg == "--cuda");
        let args = &args[usize::from(cuda)..];
        if args.len() > 3 || args.iter().any(|arg| arg.starts_with("--")) {
            return Err(USAGE.into());
        }
        let epochs = args.get(1).map_or(Ok(500), |arg| arg.parse::<usize>())?;
        let lr = args.get(2).map_or(Ok(0.3), |arg| arg.parse::<f32>())?;
        if !(1..=1_000_000).contains(&epochs) || !lr.is_finite() || lr <= 0.0 {
            return Err("expected 1–1,000,000 epochs and a finite positive learning rate".into());
        }
        Ok(Self {
            cuda,
            path: args.first().map(PathBuf::from),
            epochs,
            learning_rate: f64::from(lr),
        })
    }

    fn device(&self) -> Result<Device> {
        if !self.cuda {
            return Ok(Device::flex());
        }
        #[cfg(feature = "cuda")]
        return Ok(Device::cuda(0));
        #[cfg(not(feature = "cuda"))]
        Err("CUDA requires building with --features cuda".into())
    }
}

#[derive(Module, Debug)]
struct Classifier {
    hidden: Linear,
    output: Linear,
}

impl Classifier {
    /// Map [batch, 2] features to [batch, 2] logits through 16 ReLU units.
    fn forward(&self, inputs: Tensor<2>) -> Tensor<2> {
        self.output.forward(relu(self.hidden.forward(inputs)))
    }
}

/// Evaluate the mean loss and accuracy from a single forward pass.
fn report(
    split: &str,
    epoch: usize,
    model: &Classifier,
    batch: &Batch,
    criterion: &CrossEntropyLoss,
) {
    let logits = model.forward(batch.inputs.clone()).detach();
    let loss: f32 = criterion
        .forward(logits.clone(), batch.labels.clone())
        .into_scalar();
    let accuracy: f32 = logits
        .argmax(1)
        .squeeze_dim::<1>(1)
        .equal(batch.labels.clone())
        .float()
        .mean()
        .into_scalar();
    println!(
        "{split} epoch {epoch} loss {loss:.6} accuracy {:.2}%",
        100.0 * accuracy
    );
}

/// Each epoch computes the mean loss over the full training split and takes one SGD step.
fn train(dataset: Dataset, options: &Options, device: &Device) {
    let mut model = Classifier {
        hidden: dataset.hidden,
        output: dataset.output,
    };
    let mut optimizer = SgdConfig::new().init();
    let criterion = CrossEntropyLossConfig::new().init(device);
    report("train", 0, &model, &dataset.train, &criterion);
    for epoch in 1..=options.epochs {
        let logits = model.forward(dataset.train.inputs.clone());
        let loss = criterion.forward(logits, dataset.train.labels.clone());
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(options.learning_rate, model, grads);
        if epoch % 100 == 0 || epoch == options.epochs {
            report("train", epoch, &model, &dataset.train, &criterion);
        }
    }
    report("test", options.epochs, &model, &dataset.test, &criterion);
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!("{USAGE}");
        return Ok(());
    }

    let options = Options::parse(args)?;
    let device = options.device()?.autodiff();
    let custom_data = options.path.as_ref().map(fs::read).transpose()?;
    if let Some(data_ref) = custom_data.as_deref() { //  .as_deref().unwrap(); //(MODEL_DATA)
        let data = Dataset::load(data_ref, &device)?;
        println!("Device: {device:?}");
        train(data, &options, &device);
        Ok(())
    } else {
        Err(USAGE.into())
    }


    //Ok(())
}
