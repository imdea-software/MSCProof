use ark_ec::PairingEngine;

use ark_linear_sumcheck::Error;
use ark_linear_sumcheck::rng::{Blake2s512Rng, FeedableRNG};

use ark_ff::{Field, PrimeField};
use ark_poly::DenseMultilinearExtension;

use ark_std::vec::Vec;
use ark_std::rc::Rc;

use subroutines::{
    MultilinearProverParam,
    MultilinearVerifierParam,
    Commitment,
};

use std::time::Instant;
use std::collections::HashMap;

use derive_new::new;

use crate::utils::{
    Matrix,
    matrix_reshape,
    interpolate_uni_poly, 
    matrix_reshape_predicate, 
    predicate_processing,
};

use crate::mlsumcheck::*;

use crate::data_structures::{
    ListOfProductsOfPolynomials, 
    DenseOrSparseMultilinearExtension
};

use crate::ipformlsumcheck::prover::{
        ProverMsg,
        ProverState,
};

use crate::LayerProverOutput;

#[derive(new)]
pub struct ProverConv2D<'a, E: PairingEngine<Fr=F>, F: Field> {
    
    pub input: Vec<Matrix<F>>, // input to the convolution layer
    pub kernel: Vec<Matrix<F>>, // kernels of the convolution layer

    pub strides: (usize, usize), // used as dimension (x,y)
    pub padding: (usize, usize), 

    pub dim_input: (usize, usize, usize), // Order: channels in (sigma), x, y
    pub dim_kernel: (usize, usize, usize, usize), // Order: channels in (sigma), channels out (tau), x, y

    pub prover_msg: Vec<F>,

    // Should have length dimx_vars (input part) + ch_out_nv (weights part).
    pub initial_randomness: Vec<F>, 

    // State of the fs_rng
    // To be transfered to the next prover
    // pub fs_rng: &'a mut Blake2s512Rng,

    // Commitment key if necessary
    #[new(default)]
    pub ck: Option<&'a MultilinearProverParam<E>>,

    #[new(default)]
    pub mles: HashMap<&'a str, DenseOrSparseMultilinearExtension<F>>,
    #[new(default)]
    pub times: HashMap<String, u128>,
    #[new(default)]
    pub commits: HashMap<&'a str, Commitment<E>>,

    #[new(default)]
    pub dim_input_reshape: Option<(usize, usize, usize)>,
    
    #[new(default)]
    pub prover_state_conv: Option<ProverState<F>>, 
    #[new(default)]
    pub prover_state_reshape: Option<ProverState<F>>, 

    #[new(default)]
    pub name: String, 

}

#[derive(new, Clone)]
pub struct ProverSCOutput<E: PairingEngine, F: Field> {
    // Proof of MLSC
    pub claimed_values: (F,F),
    pub prover_msgs: Vec<ProverMsg<F>>,
    pub polynomial: ListOfProductsOfPolynomials<F>,

    // Proofs of openings
    pub input_opening:  Option<(subroutines::MultilinearKzgProof<E>, F)>,
    pub kernel_opening:  Option<(subroutines::MultilinearKzgProof<E>, F)>,

}

pub type ProverConvOutput<F> = (LayerProverOutput<F>,LayerProverOutput<F>);


#[derive(new)]
pub struct VerifierConv2D<'a, E: PairingEngine<Fr=F>, F: Field> {
    
    pub conv_output_eval: F,
    pub prover_output: ProverSCOutput<E,F>,
    
    #[new(default)]
    pub vk: Option<&'a MultilinearVerifierParam<E>>,

    // Random challenges "sent" to the prover (using FS)
    #[new(default)]
    pub verifier_msgs: Option<Vec<F>>,
    // Final challenge (using FS)
    #[new(default)]
    pub final_challenge: Option<F>,

}

// Should take as input the two matrices and process them to be used by MLSC
// then do the variable fix and outputs the final values for the verifier 
impl<'a, E: PairingEngine<Fr=F>, F: Field + PrimeField> ProverConv2D<'a, E, F> {

    // Reshape the input for matrix mul, then write it into the MLE.
    // Take into account strides and padding
    pub fn input_processing(&mut self) -> (
        DenseOrSparseMultilinearExtension<F>, 
        (usize,usize,usize), 
        (usize,usize,usize)
    )
    {
        let input = self.input.clone();
        let ch_in = self.dim_kernel.0;
        let mx = self.dim_kernel.2;
        let my = self.dim_kernel.3;

        let nx = self.dim_input.1;
        let ny = self.dim_input.2;

        assert_eq!(ch_in, self.dim_input.0);

        let stridex = self.strides.0;
        let stridey = self.strides.1; 

        let paddingx = self.padding.0;
        let paddingy = self.padding.1;

        let now = Instant::now();
        let (input_reshaped, (dimin, dimx, dimy)) = matrix_reshape(
            &input, 
            (ch_in, nx, ny), 
            (mx, my),
            self.strides,
            self.padding,
        );
        let elapsed_time = now.elapsed();
        self.times.insert(format!("time_input_reshape_{}", self.name), elapsed_time.as_micros());

        assert_eq!(dimx, 
            ((nx - mx + paddingx) / stridex + 1) * 
            ((ny - my + paddingy) / stridey + 1)
        );
        assert_eq!(dimy, mx * my);
        
        let dimx_vars = log2i!(dimx);
        let dimy_vars = log2i!(dimy);
        let ch_in_nv = log2i!(ch_in);
        let total_nv = dimx_vars + dimy_vars + ch_in_nv;

        let mut input_reshaped_mle = vec![F::zero(); 1 << (total_nv)]; 

        // Write the polynomial X(x, y, sigma)
        for x in 0..dimx {
            let index_x = x * (1 << (ch_in_nv + dimy_vars));
            for y in 0..dimy {
                let index_y = y * (1 << ch_in_nv);
                for sigma in 0..ch_in {
                    let rev_index = rbin!(index_x + index_y + sigma, total_nv);
                    input_reshaped_mle[rev_index] = input_reshaped[sigma][x][y];
                }
            }
        }


        // Creates the MLE of the base input
        let base_dimxy_vars = log2i!(nx * ny);
        let base_ch_in_nv = log2i!(ch_in);
        let base_total_nv = base_dimxy_vars + base_ch_in_nv;
        let mut input_mle = vec![F::zero(); 1 << (base_total_nv)];

        // Compute the "transpose" of the base input mle
        // To be used with the predicate in the "reshape proof" sumcheck
        // X(sigma, x, y)
        for sigma in 0..ch_in {
            let index_s = sigma * (1 << (base_dimxy_vars));
            for x in 0..nx {
                let index_x = x * ny;
                for y in 0..ny {
                    let rev_index = rbin!(index_s + index_x + y, base_total_nv);
                    input_mle[rev_index] = input[sigma][x][y];
                }
            }
        }

        self.mles.insert(
            "base_input_mle", 
            DenseOrSparseMultilinearExtension::from(
                DenseMultilinearExtension::from_evaluations_vec(
                    base_total_nv, 
                    input_mle,
                )
            )
        );

        return (
            DenseOrSparseMultilinearExtension::from(
                DenseMultilinearExtension::from_evaluations_vec(total_nv, input_reshaped_mle)), 
            (ch_in_nv, dimx_vars, dimy_vars),
            (dimin, dimx, dimy)
            )

    }

    pub fn kernel_processing(&self) -> (DenseMultilinearExtension<F>, (usize,usize,usize))
    {
        let kernel = self.kernel.clone();

        let ch_in = self.dim_kernel.0;
        let ch_out = self.dim_kernel.1;
        let mx = self.dim_kernel.2;
        let my = self.dim_kernel.3;

        // Naming convention: x_nv indicates the number of variables (in boolean representation) required to express x

        let kernelsize = mx * my;
        let kernelsize_nv = log2i!(kernelsize); 
        let ch_in_nv = log2i!(ch_in);
        let ch_out_nv = log2i!(ch_out);

        let total_nv = kernelsize_nv + ch_in_nv + ch_out_nv;

        let mut kernel_reshaped = vec![F::zero(); 1 << (total_nv)];
        // Encodes W(tau, y, sigma)
        for tau in 0..ch_out {
            let index_tau = tau * (1 << ch_in_nv + kernelsize_nv);    
            for y in 0..kernelsize {
                let index_y = y * (1 << (ch_in_nv));
                for sigma in 0..ch_in {
                    let rev_index = rbin!(index_tau + index_y + sigma, total_nv);
                    kernel_reshaped[rev_index] = kernel[sigma][tau][y]; // transposed as we want the weights in columns for the multiplication
                }
            }
        }

        return (DenseMultilinearExtension::from_evaluations_vec(total_nv, kernel_reshaped),
                (ch_in_nv, ch_out_nv,kernelsize_nv))

    }


    pub fn matrices_processing(&mut self) -> ListOfProductsOfPolynomials<F> 
    {
        // input is in reshaped form
        let now = Instant::now();
        let (
            mut input_mle,
            input_reshape_dim_vars, 
            dim_input_reshape
        ) = self.input_processing();
        let elapsed_time = now.elapsed();
        self.times.insert(format!("time_input_process_{}", self.name), elapsed_time.as_micros());

        self.dim_input_reshape = Some(dim_input_reshape);
        self.mles.insert("input_mle", DenseOrSparseMultilinearExtension::from(input_mle.clone()));

        // Extract the multilinear extension of the kernel
        // it should have been computed in the preprocessing phase
        // Computes it if not
        let (
            mut kernel_mle,
            (ch_in_nv, _, kernelsize_nv)
        ) = match self.mles.get("kernel_mle") {
            Some(k) => {
                let kernelsize_nv = log2i!(self.dim_kernel.2 * self.dim_kernel.3); 
                let ch_in_nv = log2i!(self.dim_kernel.0);
                let ch_out_nv = log2i!(self.dim_kernel.1);
                
                (k.clone(), (ch_in_nv, ch_out_nv, kernelsize_nv))
            },
            None => {
                let (k, (ch_in_nv, ch_out_nv, kernelsize_nv)) = self.kernel_processing();
                self.mles.insert("kernel_mle", DenseOrSparseMultilinearExtension::from(k.clone()));
                (DenseOrSparseMultilinearExtension::from(k), (ch_in_nv, ch_out_nv, kernelsize_nv))
            }
        };

        // Splitting the initial randomness
        let (init_rand_input, init_rand_kernel) = self.initial_randomness.split_at(input_reshape_dim_vars.1);

        let now = Instant::now();
        input_mle = input_mle.fix_variables(init_rand_input);
        kernel_mle = kernel_mle.fix_variables(init_rand_kernel);
        let elapsed_time = now.elapsed();
        self.times.insert(format!("time_fix_var_conv_{}", self.name), elapsed_time.as_micros());

        // This corresponds to ch in and dimy for 
        let product_dim = input_reshape_dim_vars.0 + input_reshape_dim_vars.2;

        assert_eq!(input_mle.num_vars(), kernel_mle.num_vars());
        assert_eq!(product_dim, kernelsize_nv + ch_in_nv);

        let mut poly = ListOfProductsOfPolynomials::new(input_mle.num_vars());
        let mut prod = Vec::new();
        prod.push(Rc::new(input_mle));
        prod.push(Rc::new(kernel_mle));

        // coefficient
        poly.add_product(prod, F::from(1 as u32));

        poly

    }

    pub fn predicate_processing(&mut self, sumcheck_randomness: &[F]) -> ListOfProductsOfPolynomials<F> 
    {

        let mut predicate_mle = match self.mles.get("predicate_mle") {
            Some(p) => {
                p.clone()
            },
            None => {
                // predicate processing
                let (predicate, dim_input_reshape) = matrix_reshape_predicate(
                    self.dim_input,
                    self.dim_kernel, 
                    self.padding,
                    self.strides, 
                );
                let predicate_mle = predicate_processing::<F>(
                    predicate, 
                    self.dim_input, 
                    dim_input_reshape
                );
                predicate_mle
                // panic!("The predicate function was not processed or added to the local prover.")
            }
        };

        let mut base_input_mle = match self.mles.get("base_input_mle") {
            Some(i) => {
                i.clone()
            },
            None => {
                panic!("The base input mle 'transpose' was not processed.")
            }
        };

        let (_, input_reshape_x, input_reshape_y) = self.dim_input_reshape.unwrap();

        // Add the initial randomness maybe 
        let (init_rand_input, _) = self.initial_randomness.split_at(log2i!(input_reshape_x));

        // Splitting the sumcheck randomness
        // Check if the order is not the opposite!
        let (rxy, rsigma) = sumcheck_randomness.split_at(log2i!(input_reshape_y));

        let predicate_initial_randomness = [init_rand_input, rxy].concat();

        //We need to transpose the base_input mle before fixing the variables
        let now = Instant::now();
        base_input_mle = base_input_mle.fix_variables(rsigma);
        predicate_mle = predicate_mle.fix_variables(predicate_initial_randomness.as_slice());
        let elapsed_time = now.elapsed();
        self.times.insert(format!("time_fix_var_reshape_{}", self.name), elapsed_time.as_micros());

        assert_eq!(base_input_mle.num_vars(), predicate_mle.num_vars());

        let mut poly = ListOfProductsOfPolynomials::new(base_input_mle.num_vars());
        let mut prod = Vec::new();
        prod.push(Rc::new(base_input_mle));
        prod.push(Rc::new(predicate_mle));

        // coefficient
        poly.add_product(prod, F::from(1 as u32));

        poly

    }

    pub fn prove(&mut self, fs_rng: &mut Blake2s512Rng) -> Result<ProverConvOutput<F>, Error> 
    {

        /* ---- First sumcheck to prove the convolution --- */
        let polynomial_conv = self.matrices_processing();

        if polynomial_conv.num_variables == 0 {
            return Err(Error::OtherError("Polynomial is a constant. Aborting for now.".to_string()))
        }

        // self.fs_rng.feed(&polynomial_conv.info()).unwrap();
        fs_rng.feed(&polynomial_conv.info()).unwrap();

        let now = Instant::now();
        let (prover_msgs, mut prover_state) = MLSC::prove(
            &polynomial_conv, 
            // &mut self.fs_rng
            fs_rng
        ).unwrap();
        let elapsed_time = now.elapsed();
        self.times.insert(format!("time_mlsc_conv_{}", self.name), elapsed_time.as_micros());

        // let oracle_randomness = &[F::rand(&mut self.fs_rng)];
        let oracle_randomness = &[F::rand(fs_rng)];

        prover_state.randomness.push(oracle_randomness[0]);
        // self.fs_rng.feed(&oracle_randomness[0]).unwrap();
        fs_rng.feed(&oracle_randomness[0]).unwrap();

        let proof_conv = Self::final_oracle_access(&prover_state, oracle_randomness);

        /* ---- Second sumcheck to prove the reshape --- */

        let polynomial_reshape =  self.predicate_processing(
            prover_state.randomness.as_slice()
        );
        // self.fs_rng.feed(&polynomial_reshape.info()).unwrap();
        fs_rng.feed(&polynomial_reshape.info()).unwrap();

        let now = Instant::now();
        let (prover_msgs_reshape, mut prover_state_reshape) = MLSC::prove(
            &polynomial_reshape, 
            // &mut self.fs_rng
            fs_rng
        ).unwrap();
        let elapsed_time = now.elapsed();
        self.times.insert(format!("time_mlsc_reshape_{}", self.name), elapsed_time.as_micros());

        // let oracle_randomness = &[F::rand(self.fs_rng)];
        let oracle_randomness = &[F::rand(fs_rng)];

        prover_state_reshape.randomness.push(oracle_randomness[0]);
        // self.fs_rng.feed(&oracle_randomness[0]).unwrap();
        fs_rng.feed(&oracle_randomness[0]).unwrap();

        let proof_reshape = Self::final_oracle_access(&prover_state_reshape, oracle_randomness);

        self.prover_state_conv = Some(prover_state);
        self.prover_state_reshape = Some(prover_state_reshape);


        // let output_conv = ProverSCOutput {
        //     claimed_values: proof_conv,
        //     prover_msgs: prover_msgs,
        //     polynomial: polynomial_conv,
        //     input_opening: None,
        //     kernel_opening: None,
        // };

        // let output_reshape = ProverSCOutput {
        //     claimed_values: proof_reshape,
        //     prover_msgs: prover_msgs_reshape,
        //     polynomial: polynomial_reshape,
        //     input_opening: None,
        //     kernel_opening: None,
        // };
        let output_conv = LayerProverOutput {
            claimed_values: proof_conv,
            prover_msgs: prover_msgs,
            polynomial: polynomial_conv,
        };

        let output_reshape = LayerProverOutput {
            claimed_values: proof_reshape,
            prover_msgs: prover_msgs_reshape,
            polynomial: polynomial_reshape,
        };
        
        Ok((output_conv, output_reshape))

    }

    pub fn final_oracle_access(prover_state: &ProverState<F>, final_randomness: &[F]) -> (F,F) 
    {
    
        let mut input_mle = prover_state.flattened_ml_extensions[0].clone();
        let mut kernel_mle = prover_state.flattened_ml_extensions[1].clone();

        input_mle = input_mle.fix_variables(final_randomness);
        kernel_mle = kernel_mle.fix_variables(final_randomness);

        // Checking all variables were fixed
        assert_eq!(input_mle.num_vars(), 0);
        assert_eq!(kernel_mle.num_vars(), 0);

        let input_mle_eval = input_mle.evaluations()[0];
        let kernel_mle_eval = kernel_mle.evaluations()[0];

        (input_mle_eval, kernel_mle_eval)

    }

}


impl<'a, F: Field + PrimeField, E: PairingEngine<Fr=F>> VerifierConv2D<'a,E,F> {

    // verify the claimed sum using the proof
    pub fn verify(
        &mut self,
        fs_rng: &mut Blake2s512Rng
        ) -> Result<Vec<F>, Error> 
    {
        let polynomial_info = self.prover_output.polynomial.info();
        let prover_msgs = &self.prover_output.prover_msgs;
        let claimed_values = self.prover_output.claimed_values;

        fs_rng.feed(&polynomial_info)?;

        let prover_first_msg1 = prover_msgs[0].evaluations[0];
        let prover_first_msg2 = prover_msgs[0].evaluations[1];    
        let first_claimed_sum = prover_first_msg1 + prover_first_msg2;

        if self.conv_output_eval != first_claimed_sum {
            return Err(Error::Reject(Some("Prover first message is inconsistent with the output value.".into())));
        }

        let mut challenge_vec = Vec::new();        
        for i in 0..(polynomial_info.num_variables - 1) {
            let round_prover_msg = &prover_msgs[i].evaluations;
            fs_rng.feed(round_prover_msg)?;

            let round_challenge = F::rand(fs_rng);
            challenge_vec.push(round_challenge);
            let round_interpolation = interpolate_uni_poly::<F>(round_prover_msg, round_challenge);
            let round_claimed_sum: F = prover_msgs[i+1].evaluations[0] + prover_msgs[i+1].evaluations[1];

            if round_interpolation != round_claimed_sum {
                return Err(Error::Reject(Some("Prover message is inconsistent with the claim.".into())));
            }
        }

        // Last round
        let last_prover_msg = &prover_msgs[polynomial_info.num_variables-1].evaluations;
        fs_rng.feed(last_prover_msg)?;

        // Same as oracle randomness
        let last_challenge = F::rand(fs_rng);
        fs_rng.feed(&last_challenge)?;
        challenge_vec.push(last_challenge);

        let last_interpolation = interpolate_uni_poly::<F>(last_prover_msg, last_challenge);
        let prover_claimed_value = claimed_values.0 * claimed_values.1;
        // Last check
        if last_interpolation != prover_claimed_value {
            return Err(Error::Reject(Some("Prover claim is inconsistent with the claim.".into())));
        }

        self.verifier_msgs = Some(challenge_vec.clone());

        Ok(challenge_vec)
    }
}