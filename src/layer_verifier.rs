
use ark_linear_sumcheck::Error;
use ark_linear_sumcheck::rng::{Blake2s512Rng, FeedableRNG};

use ark_ff::Field;

use ark_std::vec::Vec;


use derive_new::new;

use crate::utils::interpolate_uni_poly;

use crate::LayerProverOutput;


#[derive(new)]
pub struct LayerVerifier<F: Field> {
    
    pub prover_output: LayerProverOutput<F>,
    pub output_eval: F,

    #[new(default)]
    pub input_randomness: Option<Vec<F>>,
    #[new(default)]
    pub input_fingerprint: Option<F>,

    #[new(default)]
    pub name: String, 

}


impl<F: Field> LayerVerifier<F> {

    // verify the claimed sum using the proof
    pub fn verify_SC(&mut self, fs_rng: &mut Blake2s512Rng) -> Result<Vec<F>, Error> {

        let poly_info = self.prover_output.polynomial.info();
        let proof = &self.prover_output.prover_msgs;
        let claimed_values = self.prover_output.claimed_values;
        
        fs_rng.feed(&poly_info)?;

        let prover_first_msg1 = proof[0].evaluations[0];
        let prover_first_msg2 = proof[0].evaluations[1];

        let first_claimed_sum = prover_first_msg1 + prover_first_msg2;
        if self.output_eval != first_claimed_sum {
            return Err(Error::Reject(Some(
                "Prover first message is inconsistent with the output value.".into(),
            )));
        }

        let mut challenge_vec = Vec::new();

        // loop over the number of variables - 1 as the last test is done between the last
        // messages and the product of the claimed values
        for i in 0..(poly_info.num_variables - 1) {
            let round_prover_msg = &proof[i].evaluations;
            fs_rng.feed(round_prover_msg)?;
            let round_challenge = F::rand(fs_rng);
            challenge_vec.push(round_challenge);
            let round_interpolation = interpolate_uni_poly::<F>(round_prover_msg, round_challenge);
            let round_claimed_sum: F =
                proof[i + 1].evaluations[0] + proof[i + 1].evaluations[1];
            if round_interpolation != round_claimed_sum {
                return Err(Error::Reject(Some(
                    format!("Prover message {} is inconsistent with the claim.", i).into(),
                )));
            }
        }

        // Last round
        let last_prover_msg = &proof[poly_info.num_variables - 1].evaluations;
        fs_rng.feed(last_prover_msg)?;
        // Same as oracle randomness
        let last_challenge = F::rand(fs_rng);
        fs_rng.feed(&last_challenge)?;
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

        Ok(challenge_vec)
    }
}