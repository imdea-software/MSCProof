#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(non_snake_case)]
use ark_bls12_381::Bls12_381;
use ark_ec::PairingEngine;
use ark_linear_sumcheck::rng::{Blake2s512Rng, FeedableRNG};
use ark_poly::DenseMultilinearExtension;


use ark_std::{vec::Vec, UniformRand, One};

// use csv::Writer;

use rand::thread_rng;

use modgkr_lib::utils::*;

use modgkr_lib::log2i;

use std::time::Instant;

use std::fs::File;
use std::io::{BufWriter, Error, Read, Write};
use std::collections::HashMap;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError};

use subroutines::MultilinearKzgPCS;
use subroutines::PolynomialCommitmentScheme;

use modgkr_lib::CNN::verifier::VerifierCNN;
use modgkr_lib::CNN::prover::ProverCNN;
use modgkr_lib::{LayerInfoConv, LayerInfoDense, LayerInfo};
use modgkr_lib::general_prover::*;
use modgkr_lib::general_verifier::*;

use modgkr_lib::data_structures::DenseOrSparseMultilinearExtension;

use std::thread;
use std::time::Duration;
use jemalloc_ctl::{stats, epoch};

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

// Set to choose the number of repetitions of the proof and verify
const NUM_PROVE_REPETITIONS: usize = 1;

// Prove the execution of the convolutional layers using the general prover
macro_rules! vgg11conv_general_prover_bench {
    () => {

        let mut rng = thread_rng();
        // type E = Bls12_381;
        // type Fr = <E as PairingEngine>::Fr;

        let mut time_prove: u128 = 0;
        for _ in 0..NUM_PROVE_REPETITIONS {

            let fs_rng = Blake2s512Rng::setup();

            /*--------------------------------------------------------------*/
            /*--------------- Loading the execution of VGG11 ---------------*/
            /*--------------------------------------------------------------*/


            let mut layers_conv = Vec::<LayerInfoConv<Fr>>::new();

            for i in 1..=8 {
                let mut save_file = File::open(format!("VGG11_exec/VGG11_conv{}_execution_description.txt", i)).unwrap();
                let mut file_content = Vec::new();
                save_file
                    .read_to_end(&mut file_content)
                    .unwrap();
                layers_conv.push(LayerInfoConv::<Fr>::deserialize(file_content.as_slice()).unwrap());
            }


            /*--------------------------------------------------------*/
            /*--------------- Computing the output MLE ---------------*/
            /*--------------------------------------------------------*/


            let cnn_output = &layers_conv.last().unwrap().output;
            let dim_output = layers_conv.last().unwrap().dim_output;
            let MLE_cnn_output = layer_output_to_mle(&cnn_output, layers_conv.last().unwrap().dim_output, false);

            let mut rand_output =
                Vec::with_capacity(log2i!(dim_output.0) + log2i!(dim_output.1 * dim_output.2));
            for i in 0..(log2i!(dim_output.0) + log2i!(dim_output.1 * dim_output.2)) {
                rand_output.push(Fr::rand(&mut rng));
            }
            
            /*--------------------------------------------------------*/
            /*--------------- Instanciating the prover ---------------*/
            /*--------------------------------------------------------*/

            let mut layers_info = Vec::new();
            for l in layers_conv {
                layers_info.push(
                    LayerInfo::<_>::LIC(l)
                );
            }

            epoch::advance().unwrap();

            let allocated = stats::allocated::read().unwrap();
            let resident = stats::resident::read().unwrap();
            println!("\n{} bytes allocated/{} bytes resident", allocated, resident);

            let GProver: GeneralProver<'_, E, Fr> = GeneralProver {
                model_exec: layers_info,
                prover_modules: Vec::new(),
                output_randomness: rand_output.clone(),
                times: HashMap::new(),
                general_prover_output: Vec::new(),
            };

            epoch::advance().unwrap();

            let allocated = stats::allocated::read().unwrap();
            let resident = stats::resident::read().unwrap();
            println!("{} bytes allocated/{} bytes resident", allocated, resident);

            /*--------------------------------------------------------*/
            /*----------------- Running the prover -------------------*/
            /*--------------------------------------------------------*/
        
    
            let mut fs_rng = Blake2s512Rng::setup();
            println!("Starting proof");
            let now_prove = Instant::now();

            let (gp_output, time) = GProver.streaming_prove_all_layers(&mut fs_rng);

            epoch::advance().unwrap();

            let allocated = stats::allocated::read().unwrap();
            let resident = stats::resident::read().unwrap();
            println!("{} bytes allocated/{} bytes resident", allocated, resident);
            
            let elapsed_time_prove = now_prove.elapsed();
            time_prove += elapsed_time_prove.as_micros();
        
            let MLE_eval_cnn_output = MLE_cnn_output.fix_variables(rand_output.as_slice())[0];

            println!("\nLayer: Conv8_512");
            println!("Time to perform the fix_var for the conv op: {} microsec.", time.get("time_fix_var_conv_Conv8_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", time.get("time_mlsc_conv_Conv8_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", time.get("time_fix_var_reshape_Conv8_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", time.get("time_mlsc_reshape_Conv8_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv8_512 without reshape op: {} microsec.", (time.get("time_for_proving_layer_Conv8_512").unwrap() - time.get("time_input_reshape_Conv8_512").unwrap()) / NUM_PROVE_REPETITIONS as u128);

            println!("\nLayer: Conv7_512");
            println!("Time to perform the fix_var for the conv op: {} microsec.", time.get("time_fix_var_conv_Conv7_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", time.get("time_mlsc_conv_Conv7_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", time.get("time_fix_var_reshape_Conv7_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", time.get("time_mlsc_reshape_Conv7_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv7_512 without reshape op: {} microsec.", (time.get("time_for_proving_layer_Conv7_512").unwrap() - time.get("time_input_reshape_Conv7_512").unwrap()) / NUM_PROVE_REPETITIONS as u128);

            println!("\nLayer: Conv6_512");
            println!("Time to perform the fix_var for the conv op: {} microsec.", time.get("time_fix_var_conv_Conv6_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", time.get("time_mlsc_conv_Conv6_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", time.get("time_fix_var_reshape_Conv6_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", time.get("time_mlsc_reshape_Conv6_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv6_512 without reshape op: {} microsec.", (time.get("time_for_proving_layer_Conv6_512").unwrap() - time.get("time_input_reshape_Conv6_512").unwrap()) / NUM_PROVE_REPETITIONS as u128);

            println!("\nLayer: Conv5_512");
            println!("Time to perform the fix_var for the conv op: {} microsec.", time.get("time_fix_var_conv_Conv5_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", time.get("time_mlsc_conv_Conv5_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", time.get("time_fix_var_reshape_Conv5_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", time.get("time_mlsc_reshape_Conv5_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv5_512 without reshape op: {} microsec.", (time.get("time_for_proving_layer_Conv5_512").unwrap() - time.get("time_input_reshape_Conv5_512").unwrap()) / NUM_PROVE_REPETITIONS as u128);

            println!("\nLayer: Conv4_256");
            println!("Time to perform the fix_var for the conv op: {} microsec.", time.get("time_fix_var_conv_Conv4_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", time.get("time_mlsc_conv_Conv4_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", time.get("time_fix_var_reshape_Conv4_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", time.get("time_mlsc_reshape_Conv4_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv4_256 without reshape op: {} microsec.", (time.get("time_for_proving_layer_Conv4_256").unwrap() - time.get("time_input_reshape_Conv4_256").unwrap()) / NUM_PROVE_REPETITIONS as u128);

            println!("\nLayer: Conv3_256");
            println!("Time to perform the fix_var for the conv op: {} microsec.", time.get("time_fix_var_conv_Conv3_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", time.get("time_mlsc_conv_Conv3_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", time.get("time_fix_var_reshape_Conv3_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", time.get("time_mlsc_reshape_Conv3_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv3_256 without reshape op: {} microsec.", (time.get("time_for_proving_layer_Conv3_256").unwrap() - time.get("time_input_reshape_Conv3_256").unwrap()) / NUM_PROVE_REPETITIONS as u128);

            println!("\nLayer: Conv2_128");
            println!("Time to perform the fix_var for the conv op: {} microsec.", time.get("time_fix_var_conv_Conv2_128").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", time.get("time_mlsc_conv_Conv2_128").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", time.get("time_fix_var_reshape_Conv2_128").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", time.get("time_mlsc_reshape_Conv2_128").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv2_128 without reshape op: {} microsec.", (time.get("time_for_proving_layer_Conv2_128").unwrap() - time.get("time_input_reshape_Conv2_128").unwrap()) / NUM_PROVE_REPETITIONS as u128);


            println!("\nLayer: Conv1_64");
            println!("Time to perform the fix_var for the conv op: {} microsec.", time.get("time_fix_var_conv_Conv1_64").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", time.get("time_mlsc_conv_Conv1_64").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", time.get("time_fix_var_reshape_Conv1_64").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", time.get("time_mlsc_reshape_Conv1_64").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv1_64 without reshape op: {} microsec.", (time.get("time_for_proving_layer_Conv1_64").unwrap() - time.get("time_input_reshape_Conv1_64").unwrap()) / NUM_PROVE_REPETITIONS as u128);


            println!("\nTime for the FULL PROVER: {} microsec.", time_prove / NUM_PROVE_REPETITIONS as u128);


            /*----------------------------------------------------*/
            /*--------------- Verifier Computation ---------------*/
            /*----------------------------------------------------*/

            let mut layers_conv = Vec::<LayerInfoConv<Fr>>::new();

            for i in 1..=8 {
                let mut save_file = File::open(format!("VGG11_exec/VGG11_conv{}_execution_description.txt", i)).unwrap();
                let mut file_content = Vec::new();
                save_file
                    .read_to_end(&mut file_content)
                    .unwrap();
                layers_conv.push(LayerInfoConv::<Fr>::deserialize(file_content.as_slice()).unwrap());
            }

            let mut layers_info = Vec::new();
            for l in layers_conv {
                layers_info.push(
                    LayerInfo::<_>::LIC(l)
                );
            }

            let mut fs_rng = Blake2s512Rng::setup();
            let GVerifier: GeneralVerifier<'_,Fr> = GeneralVerifier {
                model_exec: layers_info,
                model_output: MLE_eval_cnn_output,
                verifier_modules: Vec::new(),
                output_randomness: rand_output.clone(),
                general_prover_output: gp_output,
            };
        
            GVerifier.streaming_verify_all_layers(&mut fs_rng);

        }
    }
}


// Prove the execution of the full network one layer at a time using the general prover
macro_rules! vgg11_general_prover_bench {
    () => {

        let mut rng = thread_rng();


        let mut time_prove: u128 = 0;
        let mut output_layer = true;
        for _ in 0..NUM_PROVE_REPETITIONS {


            /*--------------------------------------------------------------------------*/
            /*--------------- Loading and Proving the execution of VGG11 ---------------*/
            /*--------------------------------------------------------------------------*/
            
            let mut fs_rng = Blake2s512Rng::setup();
            let mut GProver: GeneralProver<'_, E, Fr> = GeneralProver {
                model_exec: Vec::new(),
                prover_modules: Vec::new(),
                output_randomness: Vec::new(),
                times: HashMap::new(),
                general_prover_output: Vec::new(),
            };

            let mut layer_outputs = Vec::new();
            let mut MLE_dense_output_dos = DenseOrSparseMultilinearExtension::from(
                DenseMultilinearExtension::from_evaluations_vec(1,vec![Fr::one(); 2])
            );
            let mut rand_output = Vec::new();
            for i in (1..=2).rev() {

                epoch::advance().unwrap();
                let allocated = stats::allocated::read().unwrap();
                let resident = stats::resident::read().unwrap();
                println!("\n{} bytes allocated/{} bytes resident", allocated, resident);

                let mut save_file = File::open(format!("VGG11_exec/VGG11_dense4096_{}_execution_description.txt", i)).unwrap();
                let mut file_content = Vec::new();
                save_file
                    .read_to_end(&mut file_content)
                    .unwrap();
                let layer_dense = LayerInfoDense::<Fr>::deserialize(file_content.as_slice()).unwrap();
                if output_layer {
                    let cnn_output = layer_dense.output.clone();
                    let dim_output = layer_dense.dim_output.clone();
                    let (MLE_dense_output, _, _) = matrix_to_mle(
                        cnn_output.clone(), 
                        dim_output, 
                        false
                    );
                    MLE_dense_output_dos = DenseOrSparseMultilinearExtension::from(MLE_dense_output);
                    for i in 0..(log2i!(dim_output.0) + log2i!(dim_output.1)) {
                        rand_output.push(Fr::rand(&mut rng));
                    }
                    output_layer = false;
                    GProver.output_randomness = rand_output.clone();
                }
                
                epoch::advance().unwrap();
                let allocated = stats::allocated::read().unwrap();
                let resident = stats::resident::read().unwrap();
                println!("{} bytes allocated/{} bytes resident", allocated, resident);
                let now_prove = Instant::now();
                let (layer_output, time) = GProver.prove_next_layer(LayerInfo::LID(layer_dense), &mut fs_rng);
                time_prove += now_prove.elapsed().as_micros();
                layer_outputs.push(layer_output);

            }

            for i in (1..=8).rev() {
                let mut save_file = File::open(format!("VGG11_exec/VGG11_conv{}_execution_description.txt", i)).unwrap();
                let mut file_content = Vec::new();
                save_file
                    .read_to_end(&mut file_content)
                    .unwrap();
                let layer_conv = LayerInfoConv::<Fr>::deserialize(file_content.as_slice()).unwrap();

                let now_prove = Instant::now();
                let (layer_output, time) = GProver.prove_next_layer(LayerInfo::LIC(layer_conv), &mut fs_rng);
                time_prove += now_prove.elapsed().as_micros();
                layer_outputs.push(layer_output);

            }

            println!("\nTime for the FULL PROVER: {} microsec.", time_prove / NUM_PROVE_REPETITIONS as u128);

            // /*----------------------------------------------------*/
            // /*--------------- Verifier Computation ---------------*/
            // /*----------------------------------------------------*/

            let MLE_eval_cnn_output = MLE_dense_output_dos.fix_variables(rand_output.as_slice())[0];

            let mut fs_rng = Blake2s512Rng::setup();
            let mut GVerifier: GeneralVerifier<'_,Fr> = GeneralVerifier {
                model_exec: Vec::new(),
                model_output: MLE_eval_cnn_output,
                verifier_modules: Vec::new(),
                output_randomness: rand_output.clone(),
                general_prover_output: Vec::new(),
            };

            for p in layer_outputs {
                GVerifier.verify_next_layer(p, &mut fs_rng)
            }

        }
    }
}


// Prove the execution of a reduced version of the full network using the general prover
macro_rules! vgg11full_cnnprover_bench {
    () => {

        let mut rng = thread_rng();
        // type E = Bls12_381;
        // type Fr = <E as PairingEngine>::Fr;

        let mut time_prove: u128 = 0;
        for _ in 0..NUM_PROVE_REPETITIONS {

            let fs_rng = Blake2s512Rng::setup();

            /*--------------------------------------------------------------*/
            /*--------------- Loading the execution of VGG11 ---------------*/
            /*--------------------------------------------------------------*/


            let mut layers_conv = Vec::<LayerInfoConv<Fr>>::new();
            for i in 1..=8 {
                let mut save_file = File::open(format!("VGG11_exec/VGG11_conv{}_execution_description.txt", i)).unwrap();
                let mut file_content = Vec::new();
                save_file
                    .read_to_end(&mut file_content)
                    .unwrap();
                layers_conv.push(LayerInfoConv::<Fr>::deserialize(file_content.as_slice()).unwrap());
            }

            let layers_dense = Vec::<LayerInfoDense<Fr>>::new();
            let mut save_file = File::open("VGG11_exec/VGG11_dense2048_execution_description.txt").unwrap();
            let mut file_content = Vec::new();
            save_file
                .read_to_end(&mut file_content)
                .unwrap();
            let layers_dense = Vec::<LayerInfoDense<Fr>>::deserialize(file_content.as_slice()).unwrap();


            /*--------------------------------------------------------*/
            /*--------------- Computing the output MLE ---------------*/
            /*--------------------------------------------------------*/


            let cnn_output = &layers_dense.last().unwrap().output;
            let dim_output = layers_dense.last().unwrap().dim_output;
            let (MLE_dense_output, _, _) = matrix_to_mle(cnn_output.clone(), dim_output, false);
            let MLE_dense_output = DenseOrSparseMultilinearExtension::from(MLE_dense_output);

            // let cnn_output = &layers_conv.last().unwrap().output;
            // let dim_output = layers_conv.last().unwrap().dim_output;
            // let MLE_cnn_output = layer_output_to_mle(&cnn_output, layers_conv.last().unwrap().dim_output, false);
            
            let mut rand_output =
                Vec::with_capacity(log2i!(dim_output.0) + log2i!(dim_output.1));
            for i in 0..(log2i!(dim_output.0) + log2i!(dim_output.1)) {
                rand_output.push(Fr::rand(&mut rng));
            }

            /*--------------------------------------------------------*/
            /*--------------- Instanciating the prover ---------------*/
            /*--------------------------------------------------------*/

            let mut layers_info = Vec::new();
            for l in layers_conv {
                layers_info.push(
                    LayerInfo::<_>::LIC(l)
                );
            }
            for l in layers_dense {
                layers_info.push(
                    LayerInfo::<_>::LID(l)
                );
            }

            epoch::advance().unwrap();

            let allocated = stats::allocated::read().unwrap();
            let resident = stats::resident::read().unwrap();
            println!("\n{} bytes allocated/{} bytes resident", allocated, resident);

            let mut fs_rng = Blake2s512Rng::setup();
            let GProver: GeneralProver<'_, E, Fr> = GeneralProver {
                model_exec: layers_info,
                prover_modules: Vec::new(),
                output_randomness: rand_output.clone(),
                times: HashMap::new(),
                general_prover_output: Vec::new(),
            };

            epoch::advance().unwrap();

            let allocated = stats::allocated::read().unwrap();
            let resident = stats::resident::read().unwrap();
            println!("{} bytes allocated/{} bytes resident", allocated, resident);


            /*--------------------------------------------------------*/
            /*----------------- Running the prover -------------------*/
            /*--------------------------------------------------------*/
        
    
            println!("Starting proof");
            let now_prove = Instant::now();

            let (gp_output, time) = GProver.streaming_prove(&mut fs_rng);

            epoch::advance().unwrap();

            let allocated = stats::allocated::read().unwrap();
            let resident = stats::resident::read().unwrap();
            println!("{} bytes allocated/{} bytes resident", allocated, resident);
            
            let elapsed_time_prove = now_prove.elapsed();
            time_prove += elapsed_time_prove.as_micros();
        
            // let MLE_eval_cnn_output = MLE_cnn_output.fix_variables(rand_output.as_slice())[0];
            let MLE_eval_cnn_output = MLE_dense_output.fix_variables(rand_output.as_slice())[0];

            println!("\nLayer: Conv8_512");
            println!("Time to perform the fix_var for the conv op: {} microsec.", time.get("time_fix_var_conv_Conv8_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", time.get("time_mlsc_conv_Conv8_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", time.get("time_fix_var_reshape_Conv8_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", time.get("time_mlsc_reshape_Conv8_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv8_512 without reshape op: {} microsec.", (time.get("time_for_proving_layer_Conv8_512").unwrap() - time.get("time_input_reshape_Conv8_512").unwrap()) / NUM_PROVE_REPETITIONS as u128);

            println!("\nLayer: Conv7_512");
            println!("Time to perform the fix_var for the conv op: {} microsec.", time.get("time_fix_var_conv_Conv7_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", time.get("time_mlsc_conv_Conv7_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", time.get("time_fix_var_reshape_Conv7_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", time.get("time_mlsc_reshape_Conv7_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv7_512 without reshape op: {} microsec.", (time.get("time_for_proving_layer_Conv7_512").unwrap() - time.get("time_input_reshape_Conv7_512").unwrap()) / NUM_PROVE_REPETITIONS as u128);

            println!("\nLayer: Conv6_512");
            println!("Time to perform the fix_var for the conv op: {} microsec.", time.get("time_fix_var_conv_Conv6_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", time.get("time_mlsc_conv_Conv6_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", time.get("time_fix_var_reshape_Conv6_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", time.get("time_mlsc_reshape_Conv6_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv6_512 without reshape op: {} microsec.", (time.get("time_for_proving_layer_Conv6_512").unwrap() - time.get("time_input_reshape_Conv6_512").unwrap()) / NUM_PROVE_REPETITIONS as u128);

            println!("\nLayer: Conv5_512");
            println!("Time to perform the fix_var for the conv op: {} microsec.", time.get("time_fix_var_conv_Conv5_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", time.get("time_mlsc_conv_Conv5_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", time.get("time_fix_var_reshape_Conv5_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", time.get("time_mlsc_reshape_Conv5_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv5_512 without reshape op: {} microsec.", (time.get("time_for_proving_layer_Conv5_512").unwrap() - time.get("time_input_reshape_Conv5_512").unwrap()) / NUM_PROVE_REPETITIONS as u128);

            println!("\nLayer: Conv4_256");
            println!("Time to perform the fix_var for the conv op: {} microsec.", time.get("time_fix_var_conv_Conv4_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", time.get("time_mlsc_conv_Conv4_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", time.get("time_fix_var_reshape_Conv4_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", time.get("time_mlsc_reshape_Conv4_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv4_256 without reshape op: {} microsec.", (time.get("time_for_proving_layer_Conv4_256").unwrap() - time.get("time_input_reshape_Conv4_256").unwrap()) / NUM_PROVE_REPETITIONS as u128);

            println!("\nLayer: Conv3_256");
            println!("Time to perform the fix_var for the conv op: {} microsec.", time.get("time_fix_var_conv_Conv3_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", time.get("time_mlsc_conv_Conv3_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", time.get("time_fix_var_reshape_Conv3_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", time.get("time_mlsc_reshape_Conv3_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv3_256 without reshape op: {} microsec.", (time.get("time_for_proving_layer_Conv3_256").unwrap() - time.get("time_input_reshape_Conv3_256").unwrap()) / NUM_PROVE_REPETITIONS as u128);

            println!("\nLayer: Conv2_128");
            println!("Time to perform the fix_var for the conv op: {} microsec.", time.get("time_fix_var_conv_Conv2_128").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", time.get("time_mlsc_conv_Conv2_128").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", time.get("time_fix_var_reshape_Conv2_128").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", time.get("time_mlsc_reshape_Conv2_128").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv2_128 without reshape op: {} microsec.", (time.get("time_for_proving_layer_Conv2_128").unwrap() - time.get("time_input_reshape_Conv2_128").unwrap()) / NUM_PROVE_REPETITIONS as u128);


            println!("\nLayer: Conv1_64");
            println!("Time to perform the fix_var for the conv op: {} microsec.", time.get("time_fix_var_conv_Conv1_64").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", time.get("time_mlsc_conv_Conv1_64").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", time.get("time_fix_var_reshape_Conv1_64").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", time.get("time_mlsc_reshape_Conv1_64").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv1_64 without reshape op: {} microsec.", (time.get("time_for_proving_layer_Conv1_64").unwrap() - time.get("time_input_reshape_Conv1_64").unwrap()) / NUM_PROVE_REPETITIONS as u128);


            println!("\nTime for the FULL PROVER: {} microsec.", time_prove / NUM_PROVE_REPETITIONS as u128);


            /*----------------------------------------------------*/
            /*--------------- Verifier Computation ---------------*/
            /*----------------------------------------------------*/

            let mut layers_conv = Vec::<LayerInfoConv<Fr>>::new();
            for i in 1..=8 {
                let mut save_file = File::open(format!("VGG11_exec/VGG11_conv{}_execution_description.txt", i)).unwrap();
                let mut file_content = Vec::new();
                save_file
                    .read_to_end(&mut file_content)
                    .unwrap();
                layers_conv.push(LayerInfoConv::<Fr>::deserialize(file_content.as_slice()).unwrap());
            }
            let layers_dense = Vec::<LayerInfoDense<Fr>>::new();
            let mut save_file = File::open("VGG11_exec/VGG11_dense2048_execution_description.txt").unwrap();
            let mut file_content = Vec::new();
            save_file
                .read_to_end(&mut file_content)
                .unwrap();
            let layers_dense = Vec::<LayerInfoDense<Fr>>::deserialize(file_content.as_slice()).unwrap();

            let mut layers_info = Vec::new();
            for l in layers_conv {
                layers_info.push(
                    LayerInfo::<_>::LIC(l)
                );
            }
            for l in layers_dense {
                layers_info.push(
                    LayerInfo::<_>::LID(l)
                );
            }

            let mut fs_rng = Blake2s512Rng::setup();
            let mut GVerifier: GeneralVerifier<'_, Fr> = GeneralVerifier {
                model_exec: layers_info,
                model_output: MLE_eval_cnn_output,
                verifier_modules: Vec::new(),
                output_randomness: rand_output.clone(),
                general_prover_output: gp_output,
            };
        
            GVerifier.setup();
            GVerifier.verify_model(&mut fs_rng);

        }
    }
}


// Prove the execution of the convolutional layers using dedicated CNN prover
macro_rules! vgg11conv_prover_bench {
    () => {


        let mut rng = thread_rng();
        // type E = Bls12_381;
        // type Fr = <E as PairingEngine>::Fr;


        let mut time_prove: u128 = 0;
        for _ in 0..NUM_PROVE_REPETITIONS {

            let mut fs_rng = Blake2s512Rng::setup();

            /*--------------------------------------------------------------*/
            /*--------------- Loading the execution of VGG11 ---------------*/
            /*--------------------------------------------------------------*/
            
            let mut save_file = File::open("VGG11_exec/VGG11_conv_execution_description.txt").unwrap();
            let mut file_content = Vec::new();
            save_file
                .read_to_end(&mut file_content)
                .unwrap();
            let layers = Vec::<LayerInfoConv<Fr>>::deserialize(file_content.as_slice()).unwrap();
        

            /*--------------------------------------------------------*/
            /*--------------- Computing the output MLE ---------------*/
            /*--------------------------------------------------------*/

            let cnn_output = &layers.last().unwrap().output;
            let dim_output = &layers.last().unwrap().dim_output;
            let MLE_cnn_output = layer_output_to_mle(&cnn_output, layers.last().unwrap().dim_output, false);

            let mut rand_output =
                Vec::with_capacity(log2i!(dim_output.0) + log2i!(dim_output.1 * dim_output.2));
            for i in 0..(log2i!(dim_output.0) + log2i!(dim_output.1 * dim_output.2)) {
                rand_output.push(Fr::rand(&mut rng));
            }


            /*--------------------------------------------------------*/
            /*--------------- Instanciating the prover ---------------*/
            /*--------------------------------------------------------*/

            let mut provercnn: ProverCNN< E, Fr> = ProverCNN::new(
                layers.clone(),
                Some(rand_output.clone()),
            );

            /*--------------------------------------------------------------*/
            /*----------------- Preprocessing the prover -------------------*/
            /*--------------------------------------------------------------*/

            provercnn.preprocessing();

            /*--------------------------------------------------------*/
            /*----------------- Running the prover -------------------*/
            /*--------------------------------------------------------*/
        
    
            println!("Starting proof");
            let now_prove = Instant::now();
            
            provercnn.prove_CNN(&mut fs_rng);
    
            let elapsed_time_prove = now_prove.elapsed();
            time_prove += elapsed_time_prove.as_micros();

            let prover_output = provercnn.layer_outputs.unwrap();
        
            let MLE_eval_cnn_output = MLE_cnn_output.fix_variables(rand_output.as_slice())[0];



            println!("\nLayer: Conv8_512");
            println!("Time to perform the fix_var for the conv op: {} microsec.", provercnn.times.get("time_fix_var_conv_Conv8_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", provercnn.times.get("time_mlsc_conv_Conv8_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", provercnn.times.get("time_fix_var_reshape_Conv8_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", provercnn.times.get("time_mlsc_reshape_Conv8_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv8_512 without reshape op: {} microsec.", (provercnn.times.get("time_for_proving_layer_Conv8_512").unwrap() - provercnn.times.get("time_input_reshape_Conv8_512").unwrap()) / NUM_PROVE_REPETITIONS as u128);

            println!("\nLayer: Conv7_512");
            println!("Time to perform the fix_var for the conv op: {} microsec.", provercnn.times.get("time_fix_var_conv_Conv7_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", provercnn.times.get("time_mlsc_conv_Conv7_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", provercnn.times.get("time_fix_var_reshape_Conv7_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", provercnn.times.get("time_mlsc_reshape_Conv7_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv7_512 without reshape op: {} microsec.", (provercnn.times.get("time_for_proving_layer_Conv7_512").unwrap() - provercnn.times.get("time_input_reshape_Conv7_512").unwrap()) / NUM_PROVE_REPETITIONS as u128);

            println!("\nLayer: Conv6_512");
            println!("Time to perform the fix_var for the conv op: {} microsec.", provercnn.times.get("time_fix_var_conv_Conv6_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", provercnn.times.get("time_mlsc_conv_Conv6_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", provercnn.times.get("time_fix_var_reshape_Conv6_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", provercnn.times.get("time_mlsc_reshape_Conv6_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv6_512 without reshape op: {} microsec.", (provercnn.times.get("time_for_proving_layer_Conv6_512").unwrap() - provercnn.times.get("time_input_reshape_Conv6_512").unwrap()) / NUM_PROVE_REPETITIONS as u128);

            println!("\nLayer: Conv5_512");
            println!("Time to perform the fix_var for the conv op: {} microsec.", provercnn.times.get("time_fix_var_conv_Conv5_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", provercnn.times.get("time_mlsc_conv_Conv5_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", provercnn.times.get("time_fix_var_reshape_Conv5_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", provercnn.times.get("time_mlsc_reshape_Conv5_512").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv5_512 without reshape op: {} microsec.", (provercnn.times.get("time_for_proving_layer_Conv5_512").unwrap() - provercnn.times.get("time_input_reshape_Conv5_512").unwrap()) / NUM_PROVE_REPETITIONS as u128);

            println!("\nLayer: Conv4_256");
            println!("Time to perform the fix_var for the conv op: {} microsec.", provercnn.times.get("time_fix_var_conv_Conv4_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", provercnn.times.get("time_mlsc_conv_Conv4_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", provercnn.times.get("time_fix_var_reshape_Conv4_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", provercnn.times.get("time_mlsc_reshape_Conv4_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv4_256 without reshape op: {} microsec.", (provercnn.times.get("time_for_proving_layer_Conv4_256").unwrap() - provercnn.times.get("time_input_reshape_Conv4_256").unwrap()) / NUM_PROVE_REPETITIONS as u128);

            println!("\nLayer: Conv3_256");
            println!("Time to perform the fix_var for the conv op: {} microsec.", provercnn.times.get("time_fix_var_conv_Conv3_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", provercnn.times.get("time_mlsc_conv_Conv3_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", provercnn.times.get("time_fix_var_reshape_Conv3_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", provercnn.times.get("time_mlsc_reshape_Conv3_256").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv3_256 without reshape op: {} microsec.", (provercnn.times.get("time_for_proving_layer_Conv3_256").unwrap() - provercnn.times.get("time_input_reshape_Conv3_256").unwrap()) / NUM_PROVE_REPETITIONS as u128);

            println!("\nLayer: Conv2_128");
            println!("Time to perform the fix_var for the conv op: {} microsec.", provercnn.times.get("time_fix_var_conv_Conv2_128").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", provercnn.times.get("time_mlsc_conv_Conv2_128").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", provercnn.times.get("time_fix_var_reshape_Conv2_128").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", provercnn.times.get("time_mlsc_reshape_Conv2_128").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv2_128 without reshape op: {} microsec.", (provercnn.times.get("time_for_proving_layer_Conv2_128").unwrap() - provercnn.times.get("time_input_reshape_Conv2_128").unwrap()) / NUM_PROVE_REPETITIONS as u128);


            println!("\nLayer: Conv1_64");
            println!("Time to perform the fix_var for the conv op: {} microsec.", provercnn.times.get("time_fix_var_conv_Conv1_64").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the conv op: {} microsec.", provercnn.times.get("time_mlsc_conv_Conv1_64").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the fix_var for the rehape op: {} microsec.", provercnn.times.get("time_fix_var_reshape_Conv1_64").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to perform the MLSC for the rehape op: {} microsec.", provercnn.times.get("time_mlsc_reshape_Conv1_64").unwrap() / NUM_PROVE_REPETITIONS as u128);
            println!("Time to prove layer_Conv1_64 without reshape op: {} microsec.", (provercnn.times.get("time_for_proving_layer_Conv1_64").unwrap() - provercnn.times.get("time_input_reshape_Conv1_64").unwrap()) / NUM_PROVE_REPETITIONS as u128);


            println!("\nTime for the FULL PROVER: {} microsec.", time_prove / NUM_PROVE_REPETITIONS as u128);


            /*----------------------------------------------------*/
            /*--------------- Verifier Computation ---------------*/
            /*----------------------------------------------------*/

            let mut fs_rng = Blake2s512Rng::setup();
            let mut verifiercnn: VerifierCNN<Fr> = VerifierCNN::new(
                layers,
                prover_output,
                Some(MLE_eval_cnn_output),
                Some(rand_output),
            );
            let verif = verifiercnn.verify_SC(&mut fs_rng).unwrap();
            let verif = verifiercnn.verify_fingerprint().unwrap();

        }
    }
}




fn main() {
    type E = Bls12_381;
    type Fr = <E as PairingEngine>::Fr;

    vgg11conv_general_prover_bench!();

    // vgg11_general_prover_bench!();

    // vgg11full_cnnprover_bench!();
    
    // vgg11conv_prover_bench!();

}