//! Reader for the original classifier's little-endian NN2CLS01 file.
use std::io::{self, Cursor, Read};

use burn::module::Param;
use burn::nn::Linear;
use burn::prelude::*;

pub const INPUTS: usize = 2;
pub const HIDDEN: usize = 16;
pub const CLASSES: usize = 2;

const MAX_SAMPLES: usize = 1_000_000;
const HEADER_BYTES: usize = 28;
const PARAMS: usize = HIDDEN * INPUTS + HIDDEN + CLASSES * HIDDEN + CLASSES;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub struct Batch {
    pub inputs: Tensor<2>,
    pub labels: Tensor<1, Int>,
}

pub struct Dataset {
    pub train: Batch,
    pub test: Batch,
    pub hidden: Linear,
    pub output: Linear,
}

impl Dataset {
    /// Load both splits and initialize trainable layers from the shared file.
    pub fn load(bytes: &[u8], device: &Device) -> Result<Self> {
        let mut reader = Cursor::new(bytes);
        let (ntrain, ntest) = read_header(&mut reader)?;
        let train = read_batch(&mut reader, ntrain, device)?;
        let test = read_batch(&mut reader, ntest, device)?;
        let hidden = read_layer(&mut reader, INPUTS, HIDDEN, device)?;
        let output = read_layer(&mut reader, HIDDEN, CLASSES, device)?;

        Ok(Self {
            train,
            test,
            hidden,
            output,
        })
    }
}

/// Validate the architecture and exact payload length before allocating tensors.
fn read_header(reader: &mut Cursor<&[u8]>) -> io::Result<(usize, usize)> {
    let mut magic = [0; 8];
    reader.read_exact(&mut magic)?;
    if &magic != b"NN2CLS01" {
        return Err(invalid_data("expected the NN2CLS01 file signature"));
    }

    let ntrain = read_u32(reader)? as usize;
    let ntest = read_u32(reader)? as usize;
    let dims = [read_u32(reader)?, read_u32(reader)?, read_u32(reader)?];
    if dims != [INPUTS as u32, HIDDEN as u32, CLASSES as u32]
        || !(1..=MAX_SAMPLES).contains(&ntrain)
        || !(1..=MAX_SAMPLES).contains(&ntest)
    {
        return Err(invalid_data(
            "expected a 2-16-2 model and two nonempty splits",
        ));
    }

    let expected = HEADER_BYTES + 4 * ((ntrain + ntest) * (INPUTS + 1) + PARAMS);
    if reader.get_ref().len() != expected {
        return Err(invalid_data("dataset size disagrees with its header"));
    }
    Ok((ntrain, ntest))
}

/// Read one split as f32 features [samples, inputs] and integer class IDs [samples].
fn read_batch(reader: &mut impl Read, samples: usize, device: &Device) -> Result<Batch> {
    let inputs = read_floats(reader, samples * INPUTS)?;
    let labels = (0..samples)
        .map(|_| {
            let label = read_u32(reader)?;
            if label >= CLASSES as u32 {
                return Err(invalid_data("class IDs must be 0 or 1"));
            }
            Ok(label as i32)
        })
        .collect::<io::Result<Vec<_>>>()?;

    Ok(Batch {
        inputs: Tensor::from_data(TensorData::new(inputs, [samples, INPUTS]), device),
        labels: Tensor::from_data(TensorData::new(labels, [samples]), device),
    })
}

/// Convert the shared output-major weights to Burn's [inputs, outputs] layout.
fn read_layer(
    reader: &mut impl Read,
    inputs: usize,
    outputs: usize,
    device: &Device,
) -> Result<Linear> {
    let output_major = read_floats(reader, outputs * inputs)?;
    let weights = (0..inputs)
        .flat_map(|input| (0..outputs).map(move |output| output * inputs + input))
        .map(|index| output_major[index])
        .collect::<Vec<_>>();
    let bias = read_floats(reader, outputs)?;

    Ok(Linear {
        weight: Param::from_tensor(Tensor::from_data(
            TensorData::new(weights, [inputs, outputs]),
            device,
        )),
        bias: Some(Param::from_tensor(Tensor::from_data(
            TensorData::new(bias, [outputs]),
            device,
        ))),
    })
}

fn read_floats(reader: &mut impl Read, count: usize) -> io::Result<Vec<f32>> {
    (0..count)
        .map(|_| {
            let value = f32::from_bits(read_u32(reader)?);
            if !value.is_finite() {
                return Err(invalid_data("features and weights must be finite"));
            }
            Ok(value)
        })
        .collect()
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn invalid_data(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
