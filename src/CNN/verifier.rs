// //! Verifier
use ark_ff::{
    Field, 
    PrimeField
};
use ark_linear_sumcheck::Error;
use ark_linear_sumcheck::rng::Blake2s512Rng;

use ark_poly::MultilinearExtension;

use ark_std::vec::Vec;

use ark_ec::PairingEngine;

use subroutines::{
    MultilinearKzgPCS,
    MultilinearVerifierParam,
    PolynomialCommitmentScheme,
    Commitment,
    MultilinearKzgProof,
};

use std::collections::HashMap;
use std::iter::zip;

use itertools::izip;

use derive_new::new;



use crate::conv::{ProverConvOutput, VerifierConv2D};

use crate::utils::{
    kernel_processing,
    matrix_reshape_predicate,
    predicate_processing,
    input_to_mle,
};

use crate::log2i;

use crate::LayerInfoConv;

/* The CNN verifier is an information-theoretic verifier for the CNN VE prover, meaning:
1) Fingerprints are not recomputed in the main algorithm (only in a separate function for testing and completeness)
2) No commitments and opening proofs are involved.
*/

#[derive(new)]
pub struct VerifierCNN<'a, E: PairingEngine<Fr=F>, F: Field> {
    pub layers_info: Vec<LayerInfoConv<F>>,
    pub prover_output: Vec<(ProverConvOutput<E, F>, ProverConvOutput<E, F>)>,
    pub cnn_output_MLE_eval: Option<F>,
    // Randomness of the output of the Conv block
    // ie received from the previous verifier or the initial randomness
    pub output_randomness: Option<Vec<F>>,

    // Randomness of the input of the Conv block
    // ie to be given to the next verifier if there is one
    #[new(default)]
    pub input_randomness: Option<Vec<F>>,
    #[new(default)]
    pub input_fingerprint: Option<F>,
    // State of the fs_rng
    // To be transfered to the verifiers
    pub fs_rng: Option<&'a mut Blake2s512Rng>,

    #[new(default)]
    pub layer_randomness: Vec<(Vec<F>, Vec<F>)>,
    #[new(default)]
    pub fingerprints: HashMap<&'a str, Vec<F>>,
}


impl<'a, E: PairingEngine<Fr=F>, F: Field + PrimeField> VerifierCNN<'a, E, F> {

    pub fn verify_SC(&mut self) -> Result<(), Error> {

        let mut MLE_eval_layer_output = self.cnn_output_MLE_eval.unwrap();

        let mut kernel_fingerprints = Vec::<F>::new();
        let mut predicate_fingerprints = Vec::<F>::new();

        for (layer_info, prover_output) in zip(
            self.layers_info.iter().rev(), 
            self.prover_output.iter()
        ) {

            println!("VERIFYING LAYER: {:?}", layer_info.name);
            
            let mut fs_rng = self.fs_rng.take().unwrap();
        
            let (prover_output_conv, prover_output_reshape) = prover_output;

            let mut layer_verifier_conv = VerifierConv2D::<E, F>::new(
                MLE_eval_layer_output,
                fs_rng,
            );

            let verifier_messages_conv = layer_verifier_conv.verify(
                &prover_output_conv.polynomial.info(),
                prover_output_conv.proof, 
                &prover_output_conv.prover_msgs
            ).unwrap();
            
            fs_rng = layer_verifier_conv.fs_rng;

            let mut layer_verifier_reshape = VerifierConv2D::<E, F>::new(
                prover_output_conv.proof.0,
                fs_rng,
            );

            let verifier_messages_reshape = layer_verifier_reshape.verify(
                &prover_output_reshape.polynomial.info(),
                prover_output_reshape.proof, 
                &prover_output_reshape.prover_msgs
            ).unwrap();
            
            fs_rng = layer_verifier_reshape.fs_rng;
            

            MLE_eval_layer_output = prover_output_reshape.proof.0;

            kernel_fingerprints.push(prover_output_conv.proof.1);
            predicate_fingerprints.push(prover_output_reshape.proof.1);


            self.layer_randomness.push((verifier_messages_conv, verifier_messages_reshape));
            self.fs_rng = Some(fs_rng);

        }
        
        let first_layer_randomness = self.layer_randomness.last().unwrap();
        let first_layer_info = self.layers_info.last().unwrap();

        let input_randomness = [
            first_layer_randomness.1.clone().as_slice(), 
            first_layer_randomness.0.clone()
                .split_at(
                    log2i!(
                        first_layer_info.dim_kernel.2 * 
                        first_layer_info.dim_kernel.3
                    )
                ).1
            ]
            .concat();

        self.input_randomness = Some(
            input_randomness.clone()
        );

        self.fingerprints.insert("kernel_fingerprints", kernel_fingerprints);
        self.fingerprints.insert("predicate_fingerprints", predicate_fingerprints);
        self.fingerprints.insert(
            "input_fingerprint", 
            vec![self.prover_output.last().unwrap().1.proof.0]
        );

        self.input_fingerprint = Some(self.prover_output.last().unwrap().1.proof.0);

        Ok(())
    }


    // Only for testing purposes
    pub fn verify_fingerprint(&self) -> Result<(), Error> {
        
        let kernel_fingerprints = self.fingerprints.get("kernel_fingerprints").unwrap();
        let predicate_fingerprints = self.fingerprints.get("predicate_fingerprints").unwrap();
        let input_fingerprint = self.fingerprints.get("input_fingerprint").unwrap();

        let input = self.layers_info[0].input.clone();

        let input_mle = input_to_mle(&input, self.layers_info[0].dim_input);

        let mut layer_output_randomness = self.output_randomness.clone().unwrap();

        for (
            layer_info, 
            layer_randomness,
            kernel_fingerprint,
            predicate_fingerprint,
        ) in izip!(
            self.layers_info.iter().rev(), 
            self.layer_randomness.iter(),
            kernel_fingerprints,
            predicate_fingerprints,
        ) {

            println!("VERIFYING FINGERPRINTS LAYER: {:?}", layer_info.name);


            let (verifier_messages_conv, verifier_messages_reshape) = layer_randomness;

            let layer_kernel = layer_info.kernel.clone();
            let layer_dim_kernel = layer_info.dim_kernel;

            let layer_kernel_mle = kernel_processing(layer_kernel, layer_dim_kernel);

            let (layer_predicate, layer_dim_input_reshape) = matrix_reshape_predicate(
                layer_info.dim_input,
                layer_info.dim_kernel, 
                layer_info.padding,
                layer_info.strides, 
            );
            let layer_predicate_mle = predicate_processing::<F>(
                layer_predicate, 
                layer_info.dim_input, 
                layer_dim_input_reshape
            );

            let (init_rand_input, init_rand_kernel) = layer_output_randomness.split_at(log2i!(layer_dim_input_reshape.1));

            let (rxy, rsigma) = verifier_messages_conv.split_at(log2i!(layer_info.dim_kernel.2 * layer_info.dim_kernel.3));
        
            
            let kernel_randomness = Vec::from(
                [
                    init_rand_kernel, 
                    verifier_messages_conv.as_slice()
                ]
                .concat()
            );
            let kernel_fingerprint_eval = layer_kernel_mle.fix_variables(
                kernel_randomness.as_slice()
            );

            
            let predicate_randomness = Vec::from(
                [
                    init_rand_input, 
                    rxy, 
                    verifier_messages_reshape.as_slice()
                ]
                .concat()
            );
            let predicate_fingerprint_eval = layer_predicate_mle.fix_variables(predicate_randomness.as_slice());

            assert_eq!(
                &kernel_fingerprint_eval.evaluations()[0], 
                kernel_fingerprint, 
                "Verification of the kernel fingerprint failed at {}.", layer_info.name
            );
            assert_eq!(
                &predicate_fingerprint_eval.evaluations()[0],
                predicate_fingerprint,
                "Verification of the predicate fingerprint failed at {}.", layer_info.name
            );

            layer_output_randomness = Vec::from([verifier_messages_reshape.as_slice(), rsigma,].concat());
        
        }

        let input_fingerprint_eval = input_mle.fix_variables(layer_output_randomness.as_slice());

        assert_eq!(
            &input_fingerprint_eval.evaluations[0], 
            &input_fingerprint[0],
            "Verification of the input fingerprint failed."
        );

        Ok(())
    }

    // verify commitments for the input, predicates and kernel MLEs
    pub fn verify_commitments(
        &self,
        vk: MultilinearVerifierParam<E>,
        prover_commits: HashMap<&'a str, Vec<Commitment<E>>>,
        prover_openings: HashMap<&'a str, Vec<(MultilinearKzgProof<E>, F)>>,
    ) -> Result<(), Error> {

        let kernel_commits = prover_commits.get("kernel_commits").unwrap();
        let predicate_commits = prover_commits.get("predicate_commits").unwrap();
        let input_commit = prover_commits.get("input_commit").unwrap()[0];

        let kernel_openings = prover_openings.get("kernel_openings").unwrap();
        let predicate_openings = prover_openings.get("predicate_openings").unwrap();


        let input_opening = &prover_openings.get("input_opening").unwrap()[0];

        let mut layer_output_randomness = self.output_randomness.clone().unwrap();

        for (
            layer_info, 
            layer_randomness,
            kernel_commit,
            kernel_opening,
            predicate_commit,
            predicate_opening,
        ) in izip!(
            self.layers_info.iter().rev(), 
            self.layer_randomness.iter(),
            kernel_commits,
            kernel_openings,
            predicate_commits,
            predicate_openings,
        ) {

            println!("VERIFYING COMMITS/OPENINGS LAYER: {:?}", layer_info.name);

            let (verifier_messages_conv, verifier_messages_reshape) = layer_randomness;

            let dim_input = layer_info.dim_input;
            let dim_kernel = layer_info.dim_kernel;
            let padding = layer_info.padding;
            let strides = layer_info.strides;
            let dim_input_reshape = ((dim_input.1 - dim_kernel.2 + padding.0)/strides.0 + 1) * 
                ((dim_input.2 - dim_kernel.3 + padding.1)/strides.1 + 1);

            let (init_rand_input, init_rand_kernel) = layer_output_randomness.split_at(log2i!(dim_input_reshape));

            let (rxy, rsigma) = verifier_messages_conv.split_at(log2i!(layer_info.dim_kernel.2 * layer_info.dim_kernel.3));
        
            
            let kernel_randomness = Vec::from(
                [
                    init_rand_kernel, 
                    verifier_messages_conv.as_slice()
                ]
                .concat()
            );

            let result_kernel = MultilinearKzgPCS::verify(
                &vk, kernel_commit, &kernel_randomness, &kernel_opening.1, &kernel_opening.0
            ).unwrap();

            assert!(result_kernel);

            let predicate_randomness = Vec::from(
                [
                    init_rand_input, 
                    rxy, 
                    verifier_messages_reshape.as_slice()
                ]
                .concat()
            );

            let result_predicate = MultilinearKzgPCS::verify(
                &vk, predicate_commit, &predicate_randomness, &predicate_opening.1, &predicate_opening.0
            ).unwrap();

            assert!(result_predicate);

            layer_output_randomness = Vec::from([verifier_messages_reshape.as_slice(), rsigma,].concat());
        
        }

        let result_input = MultilinearKzgPCS::verify(
            &vk, &input_commit, &layer_output_randomness, &input_opening.1, &input_opening.0
        ).unwrap();

        assert!(result_input);
        Ok(())
    }

}