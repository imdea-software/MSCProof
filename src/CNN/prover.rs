// //! Prover
use ark_ff::{
    Field, 
    PrimeField
};

use ark_linear_sumcheck::rng::Blake2s512Rng;

use ark_poly::DenseMultilinearExtension;

use ark_std::{
    rc::Rc, 
    vec::Vec, 
};

use ark_ec::{PairingEngine, AffineCurve};

use itertools::izip;

use subroutines::{
    MultilinearKzgPCS,
    MultilinearProverParam,
    PolynomialCommitmentScheme,
    Commitment,
    MultilinearKzgProof,
};

use derive_new::new;

use std::time::Instant;
use std::collections::HashMap;

use crate::ipformlsumcheck::prover::ProverState;


use crate::conv::{ProverConv2D, ProverConvOutput};

use crate::data_structures::DenseOrSparseMultilinearExtension;

use crate::utils::{
    kernel_processing,
    matrix_reshape_predicate,
    predicate_processing,
    input_to_mle, 
};

use crate::LayerInfoConv;

use crate::log2i;

/* The CNN prover currently inputs CNN layer parameters and produces a VE proof 
for the whole CNN. The proof is information-theoretic, and in particular:
1) Fingerprints are sent to the verifier in the clear
2) No commitments and opening proofs are involved.
Every CNN layer X_{i} = f_{i}(X_{i-1}, W_{i}) (for layer i) involves two sumcheck (VE) steps:
STEP 1: VE that takes a fingerprint H(X_{i}, .) as input, and outputs two fingerprints:
         H(Xrsh_{i-1}, .) of the reshaped input, and
         H(W_{i}, .) of the kernel(s) at layer i.
STEP 2: VE that takes a fingerprint H(Xrsh_{i-1}, .) and outputs two fingerprints:
         H(X_{i-1}, .) of the input in the standard form, and
         H(Pred(i), .) to the "reshaping predicate".

STEP 1 and STEP 2 are repeated sequentially for layers d, ... 1, starting from the output layer and finishing at the input layer.

Step 2 shall be implemented together with an activation function, a pooling predicate, or a quantization (re-scaling) operation.
This is not carried out in the current implementation.
*/

#[derive(new)]
pub struct ProverCNN<'a, E: PairingEngine<Fr=F>, F: Field> {
    pub layers_info: Vec<LayerInfoConv<F>>,
    pub output_randomness: Option<Vec<F>>,

    // State of the fs_rng
    // To be transfered to the provers
    // pub fs_rng: Option<&'a mut Blake2s512Rng>,

    #[new(default)]
    pub input_randomness: Option<Vec<F>>,

    #[new(default)]
    pub processing_state: bool,
    // For now ordered from first kernel to last
    #[new(default)]
    pub kernel_mles: Option<Vec<DenseOrSparseMultilinearExtension<F>>>,
    #[new(default)]
    pub predicate_mles: Option<Vec<DenseOrSparseMultilinearExtension<F>>>,
    #[new(default)]
    pub commits: HashMap<&'a str, Vec<Commitment<E>>>,
    #[new(default)]
    pub openings: HashMap<&'a str, Vec<(MultilinearKzgProof<E>, F)>>,
    #[new(default)]
    pub layer_outputs: Option<Vec<ProverConvOutput<F>>>,
    #[new(default)]
    pub layer_states: Option<Vec<(ProverState<F>, ProverState<F>)>>,

    #[new(default)]
    pub times: HashMap<String, u128>,
}

pub type ProverCNNOutput<F> = Vec<ProverConvOutput<F>>;

impl<'a, E: PairingEngine<Fr=F>, F: Field + PrimeField> ProverCNN<'a, E, F> {

    // Preprocessing of the MLE of the kernel and of the reshape predicate for sumchecks.
    pub fn preprocessing(&mut self) {

        let mut kernel_mles: Vec<DenseOrSparseMultilinearExtension<F>> = Vec::new();
        let mut predicate_mles: Vec<DenseOrSparseMultilinearExtension<F>> = Vec::new();

        for layer_info in self.layers_info.iter() {
            // kernel processing
            let layer_kernel = layer_info.kernel.clone();
            let dim_kernel = layer_info.dim_kernel;
            let kernel_mle = kernel_processing(layer_kernel, dim_kernel);
            kernel_mles.push(kernel_mle);

            // predicate processing
            let (predicate, dim_input_reshape) = matrix_reshape_predicate(
                layer_info.dim_input,
                layer_info.dim_kernel, 
                layer_info.padding,
                layer_info.strides, 
            );
            let predicate_mle = predicate_processing::<F>(
                predicate, 
                layer_info.dim_input, 
                dim_input_reshape
            );

            predicate_mles.push(predicate_mle);

        }

        self.kernel_mles = Some(kernel_mles);
        self.predicate_mles = Some(predicate_mles);

        self.processing_state = true;
        
    }

    // Does not consumes self when called 
    // uses the layers as order from network input to network output 
    // so creates the provers using reverse order i.e. from output to input
    pub fn prove_CNN(&mut self, fs_rng: &mut Blake2s512Rng) {

        let now = Instant::now();
        if !self.processing_state {
            self.preprocessing();
        }
        let elapsed_time = now.elapsed();
        self.times.insert("preprocessing_time".to_string(), elapsed_time.as_micros());

        let mut prover_outputs: Vec<ProverConvOutput<F>> = Vec::new();
        let mut prover_states: Vec<(ProverState<F>, ProverState<F>)> = Vec::new();
        let mut init_randomness = self.output_randomness.clone().expect("No output randomness given to the prover");

        let kernel_mles: Vec<DenseOrSparseMultilinearExtension<F>> = match self.kernel_mles.clone() {
            Some(kmles) => {kmles},
            None => panic!("The kernels were not processed before the proof.")
            
        };

        // Check that the memory increase due to the clone is not to great
        // else find a way to not consume the mles or clone them one at a time
        let predicate_mles = match self.predicate_mles.clone() {
            Some(pmles) => {pmles},
            None => panic!("The predicates were not processed before the proof.")
            
        };

        for (layer_info, kernel_mle, predicate_mle) in izip!(
            self.layers_info.iter(), 
            kernel_mles.into_iter(), 
            predicate_mles.into_iter()
            ).rev() 
        {
            let now_layer = Instant::now();

            println!("PROVING LAYER: {:?}", layer_info.name);

            let mut layer_prover = ProverConv2D::<E, F>::new(
                layer_info.input.clone(),
                layer_info.kernel.clone(),
                layer_info.strides,
                layer_info.padding,
                layer_info.dim_input,
                layer_info.dim_kernel,
                Vec::<F>::new(),
                init_randomness.clone(), 
            );

            layer_prover.name = layer_info.name.clone();

            layer_prover.mles = HashMap::from([("kernel_mle", kernel_mle), ("predicate_mle", predicate_mle)]);

            let now_prove = Instant::now();
            let (layer_output_conv, layer_output_reshape) = layer_prover.prove(fs_rng).unwrap();
            let elapsed_time_prove = now_prove.elapsed();
            self.times.insert(format!("proving_time_layer_{}", layer_info.name), elapsed_time_prove.as_micros());

            let conv_sc_rand = layer_prover.prover_state_conv.clone().unwrap().randomness;
            let reshape_sc_rand = layer_prover.prover_state_reshape.clone().unwrap().randomness;

            let dim_input_reshape = layer_prover.dim_input_reshape.expect("dim_input_reshape not found");
            // split randomness at log2i(dim_kernel.2 * dim_kernel.3)
            let (_, rsigma) = conv_sc_rand.split_at(log2i!(dim_input_reshape.2));

            // Concatenate the randomness from the reshape sumcheck to fix x and y
            // and the randomness from the conv sumcheck to fix sigma
            // Ordering is important as it is unpacked this way in the matrix processing
            init_randomness = Vec::from([reshape_sc_rand, rsigma.into()].concat());
            
            prover_outputs.push((layer_output_conv, layer_output_reshape));
            prover_states.push((layer_prover.prover_state_conv.unwrap(), layer_prover.prover_state_reshape.unwrap()));
            
            let elapsed_time_layer = now_layer.elapsed();
            self.times.insert(format!("time_for_proving_layer_{}", layer_info.name), elapsed_time_layer.as_micros());
            
            self.times.extend(layer_prover.times.into_iter());

            // self.fs_rng = Some(fs_rng);
        }

        self.layer_outputs = Some(prover_outputs);
        self.layer_states = Some(prover_states);
        self.input_randomness = Some(init_randomness);
    }

    // Should commit to the input of the CNN, to all the weights and to the predicates
    pub fn commit(&mut self, ck: &'a MultilinearProverParam<E>) {
        
        let mut kernel_commits: Vec<Commitment<E>> = Vec::new();
        let mut predicate_commits: Vec<Commitment<E>> = Vec::new();

        let kernel_mles: Vec<DenseOrSparseMultilinearExtension<F>> = match self.kernel_mles.clone() {
            Some(kmles) => {kmles},
            None => panic!("The kernels were not processed before the proof.")
            
        };

        let predicate_mles = match self.predicate_mles.clone() {
            Some(pmles) => {pmles},
            None => panic!("The predicates were not processed before the proof.")
            
        };

        for (kernel_mle, predicate_mle) in izip!(
            kernel_mles.into_iter(), 
            predicate_mles.into_iter()
            ).rev()
        {

            let rc_predicate_mle = Rc::new(DenseMultilinearExtension::from(predicate_mle.clone()));
            let rc_kernel_mle = Rc::new(DenseMultilinearExtension::from(kernel_mle.clone()));
            if rc_kernel_mle.num_vars == 0 {
                println!("Trying to commit to a kernel of size 1");
                println!("Using the value of the MLE in G1Affine as commitment");
                let kernel_commit = Commitment(ck.g.mul(rc_kernel_mle[0]).into());
                kernel_commits.push(kernel_commit);
            } else {
                let kernel_commit = MultilinearKzgPCS::<E>::commit(ck, &rc_kernel_mle).unwrap();
                kernel_commits.push(kernel_commit);
            }

            let predicate_commit = MultilinearKzgPCS::<E>::commit(ck, &rc_predicate_mle).unwrap();
            predicate_commits.push(predicate_commit);

        }

        // Can probably be done with the last base_input_mle 
        // if we started storing them but the rng needs to be changed a bit
        // to fit the different order of the variables
        let input = self.layers_info[0].input.clone();
        let input_mle = input_to_mle(&input, self.layers_info[0].dim_input);
        let rc_input_mle = Rc::new(input_mle.clone());
        let input_commit = MultilinearKzgPCS::<E>::commit(ck, &rc_input_mle).unwrap();

        self.commits.insert("kernel_commits", kernel_commits);
        self.commits.insert("predicate_commits", predicate_commits);
        self.commits.insert("input_commit", vec![input_commit]);


    }

    pub fn open_commitments(&mut self, ck: &'a MultilinearProverParam<E>) 
        {

        let mut kernel_openings = Vec::new();
        let mut predicate_openings = Vec::new();

        let kernel_mles: Vec<DenseOrSparseMultilinearExtension<F>> = match self.kernel_mles.clone() {
            Some(kmles) => {kmles},
            None => panic!("The kernels were not processed before the proof.")
            
        };

        let predicate_mles = match self.predicate_mles.clone() {
            Some(pmles) => {pmles},
            None => panic!("The predicates were not processed before the proof.")
            
        };

        let mut init_randomness = self.output_randomness.clone().unwrap();

        for (
            layer_info, 
            layer_state,
            kernel_mle,
            predicate_mle,
        ) in izip!(
            self.layers_info.iter().rev(), 
            self.layer_states.as_ref().unwrap().iter(),
            kernel_mles.iter().rev(),
            predicate_mles.iter().rev(),
        ) {

            let (layer_state_conv, layer_state_reshape) = layer_state;

            let rc_predicate_mle = Rc::new(DenseMultilinearExtension::from(predicate_mle.clone()));
            let rc_kernel_mle = Rc::new(DenseMultilinearExtension::from(kernel_mle.clone()));


            let conv_sc_rand = layer_state_conv.randomness.clone();
            let reshape_sc_rand = layer_state_reshape.randomness.clone();

            let dim_input = layer_info.dim_input;
            let dim_kernel = layer_info.dim_kernel;
            let padding = layer_info.padding;
            let strides = layer_info.strides;
            let dim_input_reshape = ((dim_input.1 - dim_kernel.2 + padding.0)/strides.0 + 1) * 
                ((dim_input.2 - dim_kernel.3 + padding.1)/strides.1 + 1);
            
            let (init_rand_input, init_rand_kernel) = init_randomness.split_at(log2i!(dim_input_reshape));

            // split randomness at log2i(dim_kernel.2 * dim_kernel.3)
            let (rxy, rsigma) = conv_sc_rand.split_at(log2i!(dim_kernel.2 * dim_kernel.3));

            let kernel_point = Vec::from(
                [
                    init_rand_kernel, 
                    conv_sc_rand.as_slice()
                ]
                .concat()
            );

            let kernel_opening: (MultilinearKzgProof<E>, F) = 
                MultilinearKzgPCS::<E>::open(ck, &rc_kernel_mle, &kernel_point).unwrap();

            let predicate_point = Vec::from(
                [
                    init_rand_input, 
                    rxy, 
                    reshape_sc_rand.as_slice()
                ]
                .concat()
            );

            let predicate_opening: (MultilinearKzgProof<E>, F) = 
                MultilinearKzgPCS::<E>::open(ck, &rc_predicate_mle, &predicate_point).unwrap();

            // Concatenate the randomness from the reshape sumcheck to fix x and y
            // and the randomness from the conv sumcheck to fix sigma
            // Ordering is important as it is unpacked this way in the matrix processing
            init_randomness = Vec::from([reshape_sc_rand, rsigma.into()].concat());

            kernel_openings.push(kernel_opening);
            predicate_openings.push(predicate_opening);

        }

        let input = self.layers_info[0].input.clone();
        let input_mle = input_to_mle(&input, self.layers_info[0].dim_input);
        let rc_input_mle = Rc::new(input_mle.clone());

        let input_opening: (MultilinearKzgProof<E>, F) = 
            MultilinearKzgPCS::<E>::open(ck, &rc_input_mle, &init_randomness).unwrap();

        self.openings.insert("kernel_openings", kernel_openings);
        self.openings.insert("predicate_openings", predicate_openings);
        self.openings.insert("input_opening", vec![input_opening]);
             
    }
}