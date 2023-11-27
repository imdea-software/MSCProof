use ark_ec::PairingEngine;
// //! Verifier
use ark_ff::{Field, PrimeField};

use ark_std::vec::Vec;
use ark_linear_sumcheck::rng::Blake2s512Rng;

use crate::CNN::verifier::VerifierCNN;
use crate::general_prover::ProverOutput;
use crate::{LayerInfo, LayerInfoConv,ModelExecution};
use crate::matmul::VerifierMatMul;


pub enum VerifierModule<'a, E: PairingEngine<Fr=F>, F:Field> {
    CNN(VerifierCNN<'a, E, F>),
    MatMul(VerifierMatMul<'a, F>),
}

#[derive(Debug)]
pub struct VerifierMessages<F: Field>(Vec<F>);

pub struct GeneralVerifier<'a, E: PairingEngine<Fr=F>, F: Field> {
    pub model_exec: ModelExecution<F>,
    pub model_output: F,
    // Created from last layer to first, ie in the order of the proof
    pub verifier_modules: Vec<VerifierModule<'a, E, F>>,
    pub output_randomness: Vec<F>,
    pub general_prover_output: Vec<ProverOutput<E, F>>,
    pub fs_rng: Blake2s512Rng,
}



impl<'a, E: PairingEngine<Fr=F>, F: Field + PrimeField> GeneralVerifier<'a, E, F> {

    // Creating the prover submodules
    pub fn setup(&mut self) {

        let mut verifier_modules = Vec::<VerifierModule<E, F>>::new();
        let mut cnn_module = Vec::<LayerInfoConv<F>>::new();
        let mut initial_randomness = None;
        let mut model_output = None;

        let mut gp_output_iter = self.general_prover_output.iter().rev();
        for (i, layer_info) in self.model_exec.iter().enumerate() 
        {

            if i == self.model_exec.len() - 1 {
                initial_randomness = Some(self.output_randomness.clone());
                model_output = Some(self.model_output);
            }

            match &layer_info {

                LayerInfo::LIC(l) => {
                    cnn_module.push(l.clone());
                },

                LayerInfo::LID(l) => {

                    if cnn_module.len() >= 1 {

                        let po = match gp_output_iter.next().unwrap() {
                            ProverOutput::CNNOutput(po) => {po.clone()},
                            _ => {panic!("Wrong type of or missing prover output!")}
                        };

                        verifier_modules.push(
                            VerifierModule::CNN(VerifierCNN::new(
                                cnn_module,
                                po,
                                None,
                                None,
                                None,
                            ))
                        );
                        cnn_module = Vec::<LayerInfoConv<F>>::new();

                    }

                    
                    let po = match gp_output_iter.next().unwrap()  {
                        ProverOutput::DenseOutput(po) => {po.clone()},
                        _ => {panic!("Wrong type of or missing prover output!")}
                    };

                    verifier_modules.push(
                        VerifierModule::MatMul(VerifierMatMul::new(
                            l.clone(),
                            po, 
                            model_output,
                            initial_randomness,
                            None,
                        ))
                    );
                    initial_randomness = None;
                }
            }
        }
        
        if cnn_module.len() >= 1 {
            let po = match gp_output_iter.next().unwrap() {
                ProverOutput::CNNOutput(po) => {po.clone()},
                _ => {panic!("Wrong type of or missing prover output!")}
            };

            verifier_modules.push(
                VerifierModule::CNN(
                    VerifierCNN::new(
                        cnn_module, 
                        po,
                        model_output,
                        initial_randomness, 
                        None
                        )
                    )
            );
        }

        self.verifier_modules = verifier_modules;

    }

    pub fn verify_model(&'a mut self) {


        let mut fs_rng = Some(&mut self.fs_rng);
        let mut init_randomness = self.output_randomness.clone();
        let mut output_eval = self.model_output;

        for verifier_module in self.verifier_modules.iter_mut().rev() {
            
            match verifier_module {

                VerifierModule::CNN(vm) => {

                    println!("VERIFYING CNN LAYER");
                    
                    if vm.output_randomness == None {
                        vm.output_randomness = Some(init_randomness);
                    }

                    if vm.cnn_output_MLE_eval == None {
                        vm.cnn_output_MLE_eval= Some(output_eval);
                    }

                    vm.fs_rng = fs_rng;
                    let _ = vm.verify_SC().unwrap();

                    fs_rng = vm.fs_rng.take();
                    init_randomness = vm.input_randomness.clone().unwrap();
                    output_eval = vm.input_fingerprint.clone().unwrap();

                },

                VerifierModule::MatMul(vm) => {

                    println!("VERIFYING DENSE LAYER");

                    if vm.output_randomness == None {
                        vm.output_randomness = Some(init_randomness);
                    }
                    if vm.output_eval == None {
                        vm.output_eval= Some(output_eval);
                    }

                    vm.fs_rng = fs_rng.take();
                    let _ = vm.verify().unwrap();

                    fs_rng = vm.fs_rng.take();
                    init_randomness = vm.input_randomness.clone().unwrap();
                    output_eval = vm.input_fingerprint.clone().unwrap();
                    
                }
            }
        }
    }
}