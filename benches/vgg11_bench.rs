#![allow(unused_imports)]
#![allow(unused_variables)]
use ark_bls12_381::Bls12_381;
use ark_ec::PairingEngine;
use ark_linear_sumcheck::rng::{Blake2s512Rng, FeedableRNG};


use ark_std::{vec::Vec, UniformRand};

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

const NUM_PROVE_REPETITIONS: usize = 1;
// const NUM_VERIFY_REPETITIONS: usize = 100;

macro_rules! vgg11_prover_bench {
    ($input_dim:expr, $kernel_dim:expr, $channel_in:expr, $channel_out:expr) => {

        let mut rng = thread_rng();
        type E = Bls12_381;
        type Fr = <E as PairingEngine>::Fr;

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
            for l in layers_conv.clone() {
                layers_info.push(
                    LayerInfo::<_>::LIC(l)
                );
            }
        
            let mut fs_rng = Blake2s512Rng::setup();
            let mut GProver: GeneralProver<'_, E, Fr> = GeneralProver {
                model_exec: layers_info.clone(),
                prover_modules: Vec::new(),
                output_randomness: rand_output.clone(),
                fs_rng: Some(&mut fs_rng),
                times: HashMap::new(),
                general_prover_output: Vec::new(),
            };

            /*--------------------------------------------------------------*/
            /*----------------- Preprocessing the prover -------------------*/
            /*--------------------------------------------------------------*/


            GProver.setup();

            /*--------------------------------------------------------*/
            /*----------------- Running the prover -------------------*/
            /*--------------------------------------------------------*/
        
    
            println!("Starting proof");
            let now_prove = Instant::now();
            
            let gp_output = GProver.prove_model();
            
            let time = GProver.times.clone();
            
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



            let mut GVerifier: GeneralVerifier<'_, E, Fr> = GeneralVerifier {
                model_exec: layers_info.clone(),
                model_output: MLE_eval_cnn_output,
                verifier_modules: Vec::new(),
                output_randomness: rand_output.clone(),
                general_prover_output: gp_output,
                fs_rng: Blake2s512Rng::setup(),
            };
        
            GVerifier.setup();

            GVerifier.verify_model();
        }
    }
}


macro_rules! vgg11conv_prover_bench {
    ($input_dim:expr, $kernel_dim:expr, $channel_in:expr, $channel_out:expr) => {


        let mut rng = thread_rng();
        type E = Bls12_381;
        type Fr = <E as PairingEngine>::Fr;


        let mut time_prove: u128 = 0;
        for _ in 0..NUM_PROVE_REPETITIONS {

            let mut fs_rng = Blake2s512Rng::setup();

            /*--------------------------------------------------------------*/
            /*--------------- Loading the execution of VGG11 ---------------*/
            /*--------------------------------------------------------------*/
            
            let mut save_file = File::open("VGG11_execution_description.txt").unwrap();
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
                Some(&mut fs_rng),
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
            
            provercnn.prove_CNN();
    
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
            let mut verifiercnn: VerifierCNN<E, Fr> = VerifierCNN::new(
                layers,
                prover_output,
                Some(MLE_eval_cnn_output),
                Some(rand_output),
                Some(&mut fs_rng),
            );
            let verif = verifiercnn.verify_SC().unwrap();
            let verif = verifiercnn.verify_fingerprint().unwrap();

        }
    }
}




fn main() {

    vgg11_prover_bench!(dim_input, dim_kernel, dchannelin, dchannelout);
    
    // vgg11conv_prover_bench!(dim_input, dim_kernel, dchannelin, dchannelout);

}