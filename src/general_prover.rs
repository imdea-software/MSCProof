use ark_ec::PairingEngine;
// //! Prover
use ark_ff::{Field, PrimeField};

use ark_std::vec::Vec;
use ark_linear_sumcheck::rng::Blake2s512Rng;

// use ark_serialize::Read;
// use ark_serialize::SerializationError;
// use ark_serialize::Write;

use crate::{LayerInfo, LayerInfoConv, ModelExecution};
use crate::matmul::{ProverMatMul, ProverMatMulOutput};

use crate::CNN::prover::{ProverCNN, ProverCNNOutput};

use crate::utils::conv_layer_output_to_flatten;

use std::collections::HashMap;
// use std::time::Instant;

use derive_new::new;


// The goal is to have a general prover whose purpose is to prove
// the general relation output = f(input)
//
// To do so we want to chain small provers whose purpose is to prove
// layer_output = f(layer_input)
//
// In general, we want an equivalence between the GeneralProver and
// the function (e.g. NN) we want to prove
//
// Something like:
//
//    Function                      General Prover
//  _____________
// |   output    | ------------>   XXXXXXXXXXXXXXX
// |_____________|
// |  mat mul    | ------------>   1. MatMul Prover
// |_____________|
// |  mat mul    | ------------>   2. MatMul Prover
// |_____________|
// |   input     | ------------>   XXXXXXXXXXXXXXX
// |_____________|
//
// Where each MatMul prover proves the result of a matrix multiplication
// and the General Prover puts them together to prove the whole function
//
// We also want a description of the fuction that allows us to extract
// all the modules that we need and combine them
//

/// Kernel of size 1 are not supported yet



#[derive(new)]
pub struct GeneralProver<'a, E: PairingEngine<Fr=F>, F: Field> {
    pub model_exec: ModelExecution<F>,
    // Created from first layer to last
    pub prover_modules: Vec<ProverModule<'a, E, F>>,
    pub output_randomness: Vec<F>,
    pub fs_rng: Option<&'a mut Blake2s512Rng>,
    #[new(default)]
    pub times: HashMap<String, u128>,
    #[new(default)]
    pub general_prover_output: GeneralProverOutput<E,F>,
}

pub type GeneralProverOutput<E,F> = Vec<ProverOutput<E,F>>;


#[derive(Clone)]
pub enum ProverOutput<E: PairingEngine<Fr=F>, F:Field> {
    CNNOutput(ProverCNNOutput<E, F>),
    DenseOutput(ProverMatMulOutput<F>),
}

// #[derive(Clone)]
pub enum ProverModule<'a, E: PairingEngine<Fr=F>, F:Field> {
    CNN(ProverCNN<'a, E, F>),
    MatMul(ProverMatMul<'a, F>),
}

// General structure for prover messages
#[derive(Debug)]
pub struct ProverMessages<F: Field>{
    /// evaluations on P(0), P(1), P(2), ...
    pub evaluations: Vec<F>,
}


impl<'a, E: PairingEngine<Fr=F>, F: Field + PrimeField> GeneralProver<'a, E, F> {

    // Creating the prover submodules
    // the prover stores the execution twice 
    // could be reduced to one to save memory
    // the prover modules could also be generated on the fly 
    // to skip this function
    pub fn setup(&mut self) {

        let mut prover_modules = Vec::<ProverModule<E, F>>::new();
        let mut cnn_module = Vec::<LayerInfoConv<F>>::new();

        let mut initial_randomness = None;

        for (i, layer_info) in self.model_exec.iter().enumerate() {

            if i == self.model_exec.len() - 1 {
                initial_randomness = Some(self.output_randomness.clone());
            }

            match &layer_info {

                // l takes the value of what is inside the enum
                LayerInfo::LIC(l) => {
                    cnn_module.push(l.clone());
                },

                // l takes the value of what is inside the enum
                LayerInfo::LID(l) => {

                    if cnn_module.len() >= 1 {
                        prover_modules.push(
                            ProverModule::CNN(ProverCNN::new(
                                cnn_module, 
                                None, 
                                None
                            ))
                        );
                        cnn_module = Vec::<LayerInfoConv<F>>::new();
                    }

                    let dim_input = (l.input.len(), l.input[0].len(), l.input[0][0].len());
                    // Flattening the input of the convolution
                    let mat_L = conv_layer_output_to_flatten(&l.input, dim_input);
                    let dim_input_flatten = (1,mat_L.len());

                    let mut mm_module = ProverMatMul::new(
                        vec![mat_L],
                        l.kernel.clone(), 
                        dim_input_flatten, 
                        l.dim_kernel, 
                        initial_randomness,
                        None,
                    );

                    mm_module.name = l.name.clone();

                    prover_modules.push(
                        ProverModule::MatMul(mm_module)
                    );
                    initial_randomness = None;
                }
            }
            
        }
        
        if cnn_module.len() >= 1 {
            prover_modules.push(
                ProverModule::CNN(
                    ProverCNN::new(
                        cnn_module, 
                        initial_randomness, 
                        None
                        )
                    )
            );
        }

        self.prover_modules = prover_modules;

    }

    pub fn prove_model(&mut self) -> GeneralProverOutput<E,F>{

        let mut fs_rng =self.fs_rng.take();
        let mut init_randomness = Vec::<F>::new();
        let mut gp_output = Vec::new();

        for prover_module in self.prover_modules.iter_mut().rev() {
            
            
            match prover_module {

                ProverModule::CNN(pm) => {

                    println!("PROVING CNN LAYER(S)");
                    
                    if pm.output_randomness == None {
                        pm.output_randomness = Some(init_randomness);
                    }

                    pm.fs_rng = fs_rng.take();
                    pm.prove_CNN();

                    gp_output.push(
                        ProverOutput::CNNOutput(pm.layer_outputs.clone().unwrap())
                    );

                    fs_rng = pm.fs_rng.take();
                    init_randomness = pm.input_randomness.clone().unwrap();

                    self.times.extend(pm.times.clone());


                },

                ProverModule::MatMul(pm) => {

                    println!("PROVING DENSE LAYER");

                    if pm.initial_randomness == None {
                        pm.initial_randomness = Some(init_randomness);
                    }

                    pm.fs_rng = fs_rng.take();
                    let result = pm.prove();

                    gp_output.push(
                        ProverOutput::DenseOutput(result.unwrap())
                    );

                    fs_rng = pm.fs_rng.take();
                    init_randomness = pm.input_randomness.clone().unwrap();

                    self.times.extend(pm.times.clone());

                }
            }

        }
        
        gp_output
    }
}