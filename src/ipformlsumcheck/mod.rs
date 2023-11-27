//! Interactive Proof Protocol used for Multilinear Sumcheck
//! File taken from arkworks mlsumcheck library and modified to fit our needs


use ark_ff::Field;
use ark_std::marker::PhantomData;

pub mod prover;
pub mod verifier;
pub use ark_linear_sumcheck::ml_sumcheck::data_structures::{ListOfProductsOfPolynomials, PolynomialInfo};
/// Interactive Proof for Multilinear Sumcheck
pub struct IPForMLSumcheck<F: Field> {
    #[doc(hidden)]
    _marker: PhantomData<F>,
}
