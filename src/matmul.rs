use crate::data_structures::{ListOfProductsOfPolynomials, DenseOrSparseMultilinearExtension};

use ark_linear_sumcheck::rng::{Blake2s512Rng, FeedableRNG};
use ark_linear_sumcheck::Error;

use ark_ff::{Field, PrimeField};
use ark_std::rc::Rc;
use ark_std::vec::Vec;

use derive_new::new;
use std::collections::HashMap;
use std::time::Instant;

use crate::{mlsumcheck::*, LayerInfoDense};
use crate::utils::{
    Matrix,
    interpolate_uni_poly,
    matrix_to_mle,
};

use crate::ipformlsumcheck::prover::{
        ProverMsg,
        ProverState,
};

#[derive(new)]
pub struct ProverMatMul<'a, F: Field> {
    pub mat_L: Matrix<F>,
    pub mat_R: Matrix<F>,

    pub dim_L: (usize, usize),
    pub dim_R: (usize, usize),

    pub initial_randomness: Option<Vec<F>>,

    // State of the fs_rng
    // To be transfered to the next prover
    pub fs_rng: Option<&'a mut Blake2s512Rng>,

    #[new(default)]
    pub times: HashMap<String, u128>,
    #[new(default)]
    pub input_randomness: Option<Vec<F>>,
    #[new(default)]
    pub name: String, 
}

#[derive(new, Clone)]
pub struct ProverMatMulOutput<F: Field> {
    // Proof of MLSC
    pub claimed_values: (F,F),
    pub prover_msgs: Vec<ProverMsg<F>>,
    pub polynomial: ListOfProductsOfPolynomials<F>,

}


#[derive(new)]
pub struct VerifierMatMul<'a, F: Field> {
    pub layers_info: LayerInfoDense<F>,
    pub prover_output: ProverMatMulOutput<F>,

    pub output_eval: Option<F>,
    pub output_randomness: Option<Vec<F>>,

    // State of the fs_rng
    // To be transfered to the next prover
    pub fs_rng: Option<&'a mut Blake2s512Rng>,

    #[new(default)]
    pub input_randomness: Option<Vec<F>>,
    #[new(default)]
    pub input_fingerprint: Option<F>,

    #[new(default)]
    pub name: String, 
}

// Should take as input the two matrices and process them to be used by MLSC
// then do the variable fix and outputs the final values for the verifier
impl<'a, F: Field + PrimeField> ProverMatMul<'a, F> {

    pub fn matrix_processing(&mut self) -> ListOfProductsOfPolynomials<F> {

        let mat_L = self.mat_L.clone();
        let (mle_mat_L, mat_L_x_nv, _) = matrix_to_mle(mat_L, self.dim_L, false);        
        
        let mut mle_mat_L = DenseOrSparseMultilinearExtension::from(mle_mat_L);
        
        let initial_randomness = self.initial_randomness.clone().unwrap();
        let (init_rand_a, init_rand_b) = initial_randomness.split_at(mat_L_x_nv);

        let now = Instant::now();
        mle_mat_L = mle_mat_L.fix_variables(init_rand_a);
        let elapsed_time = now.elapsed();

        let mat_R = self.mat_R.clone();

        // flag for transpose is true for R matrix to setup the indices properly
        // and facilitate the computations
        let (mle_mat_R, _, _) = matrix_to_mle(mat_R, self.dim_R, true);
        let mut mle_mat_R = DenseOrSparseMultilinearExtension::from(mle_mat_R);

        let now = Instant::now();
        mle_mat_R = mle_mat_R.fix_variables(init_rand_b);
        let elapsed_time = elapsed_time + now.elapsed();
        self.times.insert(format!("time_fix_var_{}", self.name), elapsed_time.as_micros());
        
        let mut poly = ListOfProductsOfPolynomials::new(mle_mat_L.num_vars());
        let mut prod = Vec::new();
        prod.push(Rc::new(mle_mat_L));
        prod.push(Rc::new(mle_mat_R));

        // coefficient
        poly.add_product(prod, F::from(1 as u32));

        poly
    }

    pub fn prove(&mut self) -> Result<
            ProverMatMulOutput<F>,
            Error
            > 
        {

        let polynomial = self.matrix_processing();
        let mut local_rng = self.fs_rng.take().unwrap();
        local_rng.feed(&polynomial.info()).unwrap();

        let now = Instant::now();
        let (prover_msgs, mut prover_state) = MLSC::prove(
            &polynomial, 
            &mut local_rng
        ).unwrap();
        let elapsed_time = now.elapsed();
        self.times.insert(format!("time_SC_{}", self.name), elapsed_time.as_micros());
        
        let oracle_randomness = &[F::rand(&mut local_rng)];
        prover_state.randomness.push(oracle_randomness[0]);
        local_rng.feed(&oracle_randomness[0]).unwrap();

        self.fs_rng = Some(local_rng);
        self.input_randomness = Some(prover_state.randomness.clone());
        
        Ok(
            ProverMatMulOutput { 
                claimed_values: Self::final_oracle_access(&prover_state, oracle_randomness), 
                prover_msgs: prover_msgs, 
                polynomial: polynomial,
            }
        )
    }

    pub fn final_oracle_access(prover_state: &ProverState<F>, final_randomness: &[F]) -> (F, F) {

        let mut mle_mat_L = prover_state.flattened_ml_extensions[0].clone();
        let mut mle_mat_R = prover_state.flattened_ml_extensions[1].clone();

        mle_mat_L = mle_mat_L.fix_variables(final_randomness);
        mle_mat_R = mle_mat_R.fix_variables(final_randomness);

        let eval_L = mle_mat_L.evaluations()[0];
        let eval_R = mle_mat_R.evaluations()[0];

        (eval_L, eval_R)
    }

}


impl<'a, F: Field> VerifierMatMul<'a, F> {

    // verify the claimed sum using the proof
    pub fn verify(&mut self) -> Result<Vec<F>, Error> {

        let poly_info = self.prover_output.polynomial.info();
        let proof = &self.prover_output.prover_msgs;
        let claimed_values = self.prover_output.claimed_values;
        
        let mut local_rng = self.fs_rng.take().unwrap();
        local_rng.feed(&poly_info)?;

        let prover_first_msg1 = proof[0].evaluations[0];
        let prover_first_msg2 = proof[0].evaluations[1];

        let first_claimed_sum = prover_first_msg1 + prover_first_msg2;
        if self.output_eval.unwrap() != first_claimed_sum {
            return Err(Error::Reject(Some(
                "Prover first message is inconsistent with the output value.".into(),
            )));
        }

        let mut challenge_vec = Vec::new();

        // loop over the number of variables - 1 as the last test is done between the last
        // messages and the product of the claimed values
        for i in 0..(poly_info.num_variables - 1) {
            let round_prover_msg = &proof[i].evaluations;
            local_rng.feed(round_prover_msg)?;
            let round_challenge = F::rand(&mut local_rng);
            challenge_vec.push(round_challenge);
            let round_interpolation = interpolate_uni_poly::<F>(round_prover_msg, round_challenge);
            let round_claimed_sum: F =
                proof[i + 1].evaluations[0] + proof[i + 1].evaluations[1];
            if round_interpolation != round_claimed_sum {
                return Err(Error::Reject(Some(
                    "Prover message is inconsistent with the claim.".into(),
                )));
            }
        }

        // Last round
        let last_prover_msg = &proof[poly_info.num_variables - 1].evaluations;
        local_rng.feed(last_prover_msg)?;
        // Same as oracle randomness
        let last_challenge = F::rand(&mut local_rng);
        local_rng.feed(&last_challenge)?;
        challenge_vec.push(last_challenge);
        let last_interpolation = interpolate_uni_poly::<F>(last_prover_msg, last_challenge);
        let prover_claimed_value = claimed_values.0 * claimed_values.1;

        // Last check
        if last_interpolation != prover_claimed_value {
            println!("Prover claim is inconsistent with the claim.");
            return Err(Error::Reject(Some(
                "Prover claim is inconsistent with the claim.".into(),
            )));
        }

        self.fs_rng = Some(local_rng);
        self.input_randomness = Some(challenge_vec.clone());
        self.input_fingerprint = Some(claimed_values.0);

        Ok(challenge_vec)
    }
}
