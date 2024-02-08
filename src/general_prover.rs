use ark_ec::PairingEngine;
// //! Prover
use ark_ff::{Field, PrimeField};

use ark_std::vec::Vec;
use ark_linear_sumcheck::rng::{Blake2s512Rng, FeedableRNG};

// use ark_serialize::Read;
// use ark_serialize::SerializationError;
// use ark_serialize::Write;

use crate::{LayerInfo, LayerInfoConv, ModelExecution, LayerProverOutput};
use crate::matmul::ProverMatMul;
use crate::conv::{ProverConv2D, ProverConvOutput};


use crate::CNN::prover::{ProverCNN, ProverCNNOutput};

use crate::utils::{
    conv_layer_output_to_flatten,
    matrix_reshape_predicate,
    predicate_processing,
};
use crate::log2i;

use std::collections::HashMap;
use std::time::Instant;

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
    // List from first layer to last
    pub model_exec: ModelExecution<F>,
    // Created from first layer to last
    pub prover_modules: Vec<ProverModule<'a, E, F>>,
    pub output_randomness: Vec<F>,
    #[new(default)]
    pub times: HashMap<String, u128>,
    #[new(default)]
    pub general_prover_output: GeneralProverOutput<F>,
}

pub type GeneralProverOutput<F> = Vec<ProverOutput<F>>;


#[derive(Clone)]
pub enum ProverOutput<F:Field> {
    CNNOutput(ProverCNNOutput<F>),
    ConvOutput(ProverConvOutput<F>),
    DenseOutput(LayerProverOutput<F>),
}

// #[derive(Clone)]
pub enum ProverModule<'a, E: PairingEngine<Fr=F>, F:Field> {
    CNN(ProverCNN<'a, E, F>),
    MatMul(ProverMatMul<F>),
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
                                // None
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
                        initial_randomness.unwrap(),
                        // None,
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
                        // None
                        )
                    )
            );
        }

        self.prover_modules = prover_modules;

    }

    pub fn prove_model(&mut self) -> GeneralProverOutput<F>{

        let mut fs_rng = Blake2s512Rng::setup();

        let mut init_randomness = Vec::<F>::new();
        let mut gp_output = Vec::new();

        for prover_module in self.prover_modules.iter_mut().rev() {
            
            
            match prover_module {

                ProverModule::CNN(pm) => {

                    println!("PROVING CNN LAYER(S)");
                    
                    if pm.output_randomness == None {
                        pm.output_randomness = Some(init_randomness);
                    }

                    // pm.fs_rng = fs_rng.take();
                    pm.prove_CNN(&mut fs_rng);

                    gp_output.push(
                        ProverOutput::CNNOutput(pm.layer_outputs.clone().unwrap())
                    );

                    // fs_rng = pm.fs_rng.take();
                    init_randomness = pm.input_randomness.clone().unwrap();

                    self.times.extend(pm.times.clone());


                },

                ProverModule::MatMul(pm) => {

                    println!("PROVING DENSE LAYER");

                    if pm.initial_randomness.len() == 0 {
                        pm.initial_randomness = init_randomness;
                    }

                    // pm.fs_rng = fs_rng.take();
                    let result = pm.prove(&mut fs_rng);

                    gp_output.push(
                        ProverOutput::DenseOutput(result.unwrap())
                    );

                    // fs_rng = pm.fs_rng.take();
                    init_randomness = pm.input_randomness.clone().unwrap();

                    self.times.extend(pm.times.clone());

                }
            }

        }
        
        gp_output
    }

    // Prove function consuming the prover to avoid memory issues
    pub fn streaming_prove(mut self, fs_rng: &mut Blake2s512Rng) -> (GeneralProverOutput<F>, HashMap<String, u128>,) {

        
        let mut cnn_module = Vec::<LayerInfoConv<F>>::new();
        let mut gp_output: Vec<ProverOutput<F>> = Vec::new();

        let mut initial_randomness = Some(self.output_randomness);

        for layer_info in self.model_exec.into_iter().rev() {

            match layer_info {

                // l takes the value of what is inside the enum
                LayerInfo::LIC(l) => {
                    cnn_module.push(l);
                },

                // l takes the value of what is inside the enum
                LayerInfo::LID(l) => {

                    if cnn_module.len() >= 1 {

                        // reversing list as the prover cnn take layer info
                        // from input to output
                        let rev_cnn_module = cnn_module
                            .into_iter()
                            .rev()
                            .collect();

                        let mut pm = ProverCNN::<E, _>::new(
                                rev_cnn_module, 
                                initial_randomness, 
                                // None
                        );
                        
                        println!("PROVING CNN LAYER(S)");
    
                        // pm.fs_rng = fs_rng.take();
                        pm.prove_CNN(fs_rng);
    
                        gp_output.push(
                            ProverOutput::CNNOutput(pm.layer_outputs.unwrap())
                        );
    
                        // fs_rng = pm.fs_rng.take();
                        initial_randomness = pm.input_randomness;
                        
                        cnn_module = Vec::<LayerInfoConv<F>>::new();
                    }

                    let dim_input = (l.input.len(), l.input[0].len(), l.input[0][0].len());
                    // Flattening the input of the convolution
                    let mat_L = conv_layer_output_to_flatten(&l.input, dim_input);
                    let dim_input_flatten = (1,mat_L.len());

                    let mut pm = ProverMatMul::new(
                        vec![mat_L],
                        l.kernel, 
                        dim_input_flatten, 
                        l.dim_kernel, 
                        initial_randomness.unwrap(),
                        // None,
                    );
                    pm .name = l.name;


                    println!("PROVING DENSE LAYER");

                    // pm.fs_rng = fs_rng.take();
                    // pm.fs_rng = Some(&mut fs_rng);
                    let result = pm.prove(fs_rng);

                    gp_output.push(
                        ProverOutput::DenseOutput(result.unwrap())
                    );

                    // fs_rng = pm.fs_rng.take();
                    initial_randomness = pm.input_randomness;

                    self.times.extend(pm.times.clone());
                    // initial_randomness = None;

                }
            }
            
        }
        
        if cnn_module.len() >= 1 {
            
            let rev_cnn_module = cnn_module
                .into_iter()
                .rev()
                .collect();
                
            let mut pm = ProverCNN::<E, _>::new(
                rev_cnn_module, 
                initial_randomness, 
                // None
            );

            println!("PROVING CNN LAYER(S)");
                    
            // pm.fs_rng = fs_rng.take();
            pm.prove_CNN(fs_rng);

            gp_output.push(
                ProverOutput::CNNOutput(pm.layer_outputs.unwrap())
            );

            self.times.extend(pm.times.clone());

        }

        (gp_output, self.times)
    }

    // Prove function consuming the prover to avoid memory issues
    // Proves each layers independantly
    pub fn streaming_prove_all_layers(mut self, fs_rng: &mut Blake2s512Rng) -> (GeneralProverOutput<F>, HashMap<String, u128>,){

        
        let mut gp_output: Vec<ProverOutput<F>> = Vec::new();
        let mut initial_randomness = self.output_randomness;

        for layer_info in self.model_exec.into_iter().rev() {

            match layer_info {

                // l takes the value of what is inside the enum
                LayerInfo::LIC(l) => {

                    let now_layer = Instant::now();

                
                    // predicate processing
                    let (predicate, dim_input_reshape) = matrix_reshape_predicate(
                        l.dim_input,
                        l.dim_kernel, 
                        l.padding,
                        l.strides, 
                    );
                    let predicate_mle = predicate_processing::<F>(
                        predicate, 
                        l.dim_input, 
                        dim_input_reshape
                    );

                    let mut layer_prover = ProverConv2D::<E, F>::new(
                        l.input.clone(),
                        l.kernel.clone(),
                        l.strides,
                        l.padding,
                        l.dim_input,
                        l.dim_kernel,
                        Vec::<F>::new(),
                        initial_randomness.clone(), 
                    );
                    layer_prover.name = l.name.clone();
                    layer_prover.mles = HashMap::from([("predicate_mle", predicate_mle)]);
                    
                    println!("PROVING CONV LAYER {}", l.name.clone());

                    let layer_output = layer_prover.prove(fs_rng).unwrap();

                    let elapsed_time_layer = now_layer.elapsed();
                    self.times.insert(format!("time_for_proving_layer_{}", l.name), elapsed_time_layer.as_micros());
                    
                    self.times.extend(layer_prover.times.into_iter());

                    gp_output.push(
                        ProverOutput::ConvOutput(layer_output)
                    );

                    let conv_sc_rand = layer_prover.prover_state_conv.clone().unwrap().randomness;
                    let reshape_sc_rand = layer_prover.prover_state_reshape.clone().unwrap().randomness;
                    let (_, rsigma) = conv_sc_rand.split_at(log2i!(dim_input_reshape.2));
                    initial_randomness = Vec::from([reshape_sc_rand, rsigma.into()].concat());
                },

                // l takes the value of what is inside the enum
                LayerInfo::LID(l) => {


                    let dim_input = (l.input.len(), l.input[0].len(), l.input[0][0].len());
                    // Flattening the input of the convolution
                    let mat_L = conv_layer_output_to_flatten(&l.input, dim_input);
                    let dim_input_flatten = (1,mat_L.len());

                    let mut layer_prover = ProverMatMul::new(
                        vec![mat_L],
                        l.kernel, 
                        dim_input_flatten, 
                        l.dim_kernel, 
                        initial_randomness,
                    );
                    layer_prover .name = l.name;


                    println!("PROVING DENSE LAYER");

                    // pm.fs_rng = fs_rng.take();
                    let result = layer_prover.prove(fs_rng);

                    gp_output.push(
                        ProverOutput::DenseOutput(result.unwrap())
                    );

                    // fs_rng = pm.fs_rng.take();
                    initial_randomness = layer_prover.input_randomness.unwrap();

                    // self.times.extend(pm.times.clone());
                    // initial_randomness = None;

                }
            }
            
        }
        
        let rev_gp_output = gp_output
            .into_iter()
            .rev()
            .collect();

        (rev_gp_output, self.times)
    }


    // Prove function using a layer description to prove a layer
    // independantly to avoid memory issues
    // Updates the general prover randomness for successive uses
    pub fn prove_next_layer(&mut self, layer_info: LayerInfo<F>, fs_rng: &mut Blake2s512Rng) -> (ProverOutput<F>, HashMap<String, u128>,){

        let layer_proof: ProverOutput<F>;
        let initial_randomness = self.output_randomness.clone();

        match layer_info {

            // l takes the value of what is inside the enum
            LayerInfo::LIC(l) => {

                let now_layer = Instant::now();

                // predicate processing
                let (predicate, dim_input_reshape) = matrix_reshape_predicate(
                    l.dim_input,
                    l.dim_kernel, 
                    l.padding,
                    l.strides, 
                );
                let predicate_mle = predicate_processing::<F>(
                    predicate, 
                    l.dim_input, 
                    dim_input_reshape
                );

                let mut layer_prover = ProverConv2D::<E, F>::new(
                    l.input.clone(),
                    l.kernel.clone(),
                    l.strides,
                    l.padding,
                    l.dim_input,
                    l.dim_kernel,
                    Vec::<F>::new(),
                    initial_randomness.clone(), 
                );
                layer_prover.name = l.name.clone();
                layer_prover.mles = HashMap::from([("predicate_mle", predicate_mle)]);
                
                println!("PROVING CONV LAYER {}", l.name.clone());

                let layer_output = layer_prover.prove(fs_rng).unwrap();

                let elapsed_time_layer = now_layer.elapsed();
                self.times.insert(format!("time_for_proving_layer_{}", l.name), elapsed_time_layer.as_micros());
                
                self.times.extend(layer_prover.times.into_iter());

                layer_proof = ProverOutput::ConvOutput(layer_output);
                
                let conv_sc_rand = layer_prover.prover_state_conv.clone().unwrap().randomness;
                let reshape_sc_rand = layer_prover.prover_state_reshape.clone().unwrap().randomness;
                let (_, rsigma) = conv_sc_rand.split_at(log2i!(dim_input_reshape.2));
                self.output_randomness = Vec::from([reshape_sc_rand, rsigma.into()].concat());
            },

            // l takes the value of what is inside the enum
            LayerInfo::LID(l) => {


                let dim_input = (l.input.len(), l.input[0].len(), l.input[0][0].len());
                // Flattening the input of the convolution
                let mat_L = conv_layer_output_to_flatten(&l.input, dim_input);
                let dim_input_flatten = (1,mat_L.len());

                let mut layer_prover = ProverMatMul::new(
                    vec![mat_L],
                    l.kernel, 
                    dim_input_flatten, 
                    l.dim_kernel, 
                    initial_randomness,
                );
                layer_prover .name = l.name;


                println!("PROVING DENSE LAYER");

                let result = layer_prover.prove(fs_rng);

                layer_proof = ProverOutput::DenseOutput(result.unwrap());

                self.output_randomness = layer_prover.input_randomness.unwrap();

            }
        }

        (layer_proof, self.times.clone())
    }
}