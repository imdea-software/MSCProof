#![allow(non_snake_case)]
// #![allow(unused_imports)]
// #![allow(unused_variables)]
#![allow(non_upper_case_globals)]

use ark_ff::Field;
use ark_serialize::{CanonicalSerialize, SerializationError, CanonicalDeserialize};
use ark_std::io::{Write, Read};

use crate::utils::{
    Matrix,
    DimIC,
    DimKC,
    DimOC,
    DimID,
    DimKD,
    DimOD,
};

pub mod matmul;
pub mod mlsumcheck;
pub mod general_prover;
pub mod general_verifier;
pub mod CNN;
pub mod ipformlsumcheck;
pub mod data_structures;

#[macro_use]
pub mod utils;
pub mod conv;



pub type ModelExecution<T> = Vec<LayerInfo<T>>;

#[derive(Clone)]
pub enum LayerInfo<F: Field> {
    LIC(LayerInfoConv<F>),
    LID(LayerInfoDense<F>),
}

// Implemented to be able to store the layer information in a file
impl<F: Field> CanonicalSerialize for LayerInfo<F> {
    fn serialize<W: Write>(&self, mut writer: W) -> Result<(), SerializationError> {
        match self {
            LayerInfo::LIC(l) => l.serialize(&mut writer)?,
            LayerInfo::LID(l) => l.serialize(&mut writer)?,
        }
        Ok(())
    }

    fn serialized_size(&self) -> usize {
        match self {
            LayerInfo::LIC(l) => l.serialized_size(),
            LayerInfo::LID(l) => l.serialized_size(),
        }
    }
}

impl<F: Field> CanonicalDeserialize for LayerInfo<F> {
    fn deserialize<R: Read>(mut reader: R) -> Result<Self, SerializationError> {
        match LayerInfoConv::<F>::deserialize(&mut reader) {
            Ok(l) => {return Ok(LayerInfo::LIC(l));},
            Err(_) => {},
        }
        match LayerInfoDense::<F>::deserialize(&mut reader) {
            Ok(l) => {return Ok(LayerInfo::LID(l));},
            Err(_) => {},
        }
        Err(SerializationError::InvalidData)
    }
}


#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct LayerInfoConv<F: Field> {
    pub name: String,

    pub input: Vec<Matrix<F>>,
    pub kernel: Vec<Matrix<F>>,
    pub output: Vec<Matrix<F>>,

    pub dim_input: DimIC,
    pub dim_kernel: DimKC,
    pub dim_output: DimOC,

    pub padding: (usize, usize),
    pub strides: (usize, usize),
}

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct LayerInfoDense<F: Field> {
    pub name: String,

    pub input: Vec<Matrix<F>>,
    pub kernel: Matrix<F>,

    pub output: Matrix<F>, // Should be a matrix of dim (1, nb_neurons)

    pub dim_input: DimID,
    pub dim_kernel: DimKD,
    pub dim_output: DimOD,
}
