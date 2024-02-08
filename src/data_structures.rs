//! Defines the data structures used by the `MLSumcheck` protocol.
//! File taken from arkworks ark_linear_sumcheck library and modified to fit our needs
//! 
use ark_ff::Field;
use ark_poly::{SparseMultilinearExtension,DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Read, SerializationError, Write};
use ark_std::cmp::max;
use ark_std::rc::Rc;
use ark_std::vec::Vec;
use ark_std::ops::{Add, AddAssign, Index, Neg, Sub, SubAssign};
use ark_std::fmt;
use ark_std::fmt::Formatter;


use hashbrown::HashMap;

use DenseOrSparseMultilinearExtension::*;

use crate::utils::dense_to_sparse;

/// Stores a list of products of `DenseMultilinearExtension` that is meant to be added together.
///
/// The polynomial is represented by a list of products of polynomials along with its coefficient that is meant to be added together.
///
/// This data structure of the polynomial is a list of list of `(coefficient, DenseMultilinearExtension)`.
/// * Number of products n = `self.products.len()`,
/// * Number of multiplicands of ith product m_i = `self.products[i].1.len()`,
/// * Coefficient of ith product c_i = `self.products[i].0`
///
/// The resulting polynomial is
///
/// $$\sum_{i=0}^{n}C_i\cdot\prod_{j=0}^{m_i}P_{ij}$$
///
/// The result polynomial is used as the prover key.
#[derive(Clone, PartialEq, Debug)]
pub struct ListOfProductsOfPolynomials<F: Field> {
    /// max number of multiplicands in each product
    pub max_multiplicands: usize,
    /// number of variables of the polynomial
    pub num_variables: usize,
    /// list of reference to products (as usize) of multilinear extension
    pub products: Vec<(F, Vec<usize>)>,
    /// Stores multilinear extensions in which product multiplicand can refer to.
    pub flattened_ml_extensions: Vec<Rc<DenseOrSparseMultilinearExtension<F>>>,
    raw_pointers_lookup_table: HashMap<*const DenseOrSparseMultilinearExtension<F>, usize>,
}

impl<F: Field> ListOfProductsOfPolynomials<F> {
    /// Extract the max number of multiplicands and number of variables of the list of products.
    pub fn info(&self) -> PolynomialInfo {
        PolynomialInfo {
            max_multiplicands: self.max_multiplicands,
            num_variables: self.num_variables,
        }
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
/// Stores the number of variables and max number of multiplicands of the added polynomial used by the prover.
/// This data structures will is used as the verifier key.
pub struct PolynomialInfo {
    /// max number of multiplicands in each product
    pub max_multiplicands: usize,
    /// number of variables of the polynomial
    pub num_variables: usize,
}

impl<F: Field> ListOfProductsOfPolynomials<F> {
    /// Returns an empty polynomial
    pub fn new(num_variables: usize) -> Self {
        ListOfProductsOfPolynomials {
            max_multiplicands: 0,
            num_variables,
            products: Vec::new(),
            flattened_ml_extensions: Vec::new(),
            raw_pointers_lookup_table: HashMap::new(),
        }
    }

    /// Add a list of multilinear extensions that is meant to be multiplied together.
    /// The resulting polynomial will be multiplied by the scalar `coefficient`.
    pub fn add_product(
        &mut self,
        product: impl IntoIterator<Item = Rc<DenseOrSparseMultilinearExtension<F>>>,
        coefficient: F,
    ) {
        let product: Vec<Rc<DenseOrSparseMultilinearExtension<F>>> = product.into_iter().collect();
        let mut indexed_product = Vec::with_capacity(product.len());
        assert!(product.len() > 0);
        self.max_multiplicands = max(self.max_multiplicands, product.len());
        for m in product {
            assert_eq!(
                m.num_vars(), self.num_variables,
                "product has a multiplicand with wrong number of variables"
            );
            let m_ptr: *const DenseOrSparseMultilinearExtension<F> = Rc::as_ptr(&m);
            if let Some(index) = self.raw_pointers_lookup_table.get(&m_ptr) {
                indexed_product.push(*index)
            } else {
                let curr_index = self.flattened_ml_extensions.len();
                self.flattened_ml_extensions.push(m.clone());
                self.raw_pointers_lookup_table.insert(m_ptr, curr_index);
                indexed_product.push(curr_index);
            }
        }
        self.products.push((coefficient, indexed_product));
    }

    /// Evaluate the polynomial at point `point`
    pub fn evaluate(&self, point: &[F]) -> F {
        self.products
            .iter()
            .map(|(c, p)| {
                *c * p
                    .iter()
                    .map(|&i| self.flattened_ml_extensions[i].evaluate(point).unwrap())
                    .product::<F>()
            })
            .sum()
    }
}

#[derive(Clone, PartialEq)]
pub enum DenseOrSparseMultilinearExtension<F: Field> {
    SMultilinearExtension(SparseMultilinearExtension<F>),
    DMultilinearExtension(DenseMultilinearExtension<F>),
    
}

impl<F: Field> From<DenseMultilinearExtension<F>> for DenseOrSparseMultilinearExtension<F> {
    fn from(other: DenseMultilinearExtension<F>) -> Self {
        DMultilinearExtension(other)
    }
    
}

impl<F: Field> From<SparseMultilinearExtension<F>> for DenseOrSparseMultilinearExtension<F> {
    fn from(other: SparseMultilinearExtension<F>) -> Self {
        SMultilinearExtension(other)
    }
    
}

impl<F: Field> From<DenseOrSparseMultilinearExtension<F>> for DenseMultilinearExtension<F> {
    fn from(other: DenseOrSparseMultilinearExtension<F>) -> DenseMultilinearExtension<F> {
        match other {
            DMultilinearExtension(mle) => mle,
            SMultilinearExtension(mle) => mle.to_dense_multilinear_extension(),
        }
    }
}

impl<F: Field> TryInto<SparseMultilinearExtension<F>> for DenseOrSparseMultilinearExtension<F> {
    type Error = ();

    fn try_into(self) -> Result<SparseMultilinearExtension<F>, ()> {
        match self {
            SMultilinearExtension(p) => Ok(p),
            _ => Err(()),
        }
    }
}


impl<F: Field> DenseOrSparseMultilinearExtension<F> {

    pub fn evaluations(&self) -> Vec<F> {
        match self {
            DMultilinearExtension(mle) => mle.evaluations.clone(),
            SMultilinearExtension(mle) => mle.to_evaluations(),
        } 
    }

    pub fn num_vars(&self) -> usize {
        match self {
            DMultilinearExtension(mle) => mle.num_vars(),
            SMultilinearExtension(mle) => mle.num_vars(),
        }
    }

    pub fn evaluate(&self, point: &[F]) -> Option<F> {
        match self {
            DMultilinearExtension(mle) => mle.evaluate(point),
            SMultilinearExtension(mle) => mle.evaluate(point),
        }
    }


    pub fn relabel(&self, a: usize, b: usize, k: usize) -> Self {
        match self {
            DMultilinearExtension(mle) => Self::from(mle.relabel(a, b, k)),
            SMultilinearExtension(mle) => Self::from(mle.relabel(a, b, k)),
        }
    }

    pub fn fix_variables(&self, partial_point: &[F]) -> Self {
        match self {
            DMultilinearExtension(mle) => Self::from(mle.fix_variables(partial_point)),
            SMultilinearExtension(mle) => Self::from(mle.fix_variables(partial_point)),
        }
    }

    pub fn to_evaluations(&self) -> Vec<F> {
        match self {
            DMultilinearExtension(mle) => mle.to_evaluations(),
            SMultilinearExtension(mle) => mle.to_evaluations(),
        }
    }
   
}


impl<F: Field> Index<usize> for DenseOrSparseMultilinearExtension<F> {
    type Output = F;

    /// Returns the evaluation of the polynomial at a point represented by index.
    ///
    /// Index represents a vector in {0,1}^`num_vars` in little endian form. For example, `0b1011` represents `P(1,1,0,1)`
    ///
    /// For dense multilinear polynomial, `index` takes constant time.
    fn index(&self, index: usize) -> &Self::Output {
        match self {
            DMultilinearExtension(mle) => mle.index(index),
            SMultilinearExtension(mle) => mle.index(index),
        }
    }
}

impl<F: Field> Add for DenseOrSparseMultilinearExtension<F> {
    type Output = DenseOrSparseMultilinearExtension<F>;

    fn add(self, other: DenseOrSparseMultilinearExtension<F>) -> Self {
        &self + &other
    }
}

impl<'a, 'b, F: Field> Add<&'a DenseOrSparseMultilinearExtension<F>> for &'b DenseOrSparseMultilinearExtension<F> {
    type Output = DenseOrSparseMultilinearExtension<F>;

    fn add(self, rhs: &'a DenseOrSparseMultilinearExtension<F>) -> Self::Output {
        match self {
            DMultilinearExtension(mle_lhs) => {
                match rhs {
                    DMultilinearExtension(mle_rhs) => {
                        let result = mle_lhs.add(mle_rhs);
                        let output = DenseOrSparseMultilinearExtension::from(result);
                        output
                      
                    },
                    SMultilinearExtension(mle_rhs) => {
                        let result = mle_lhs.add(&mle_rhs.to_dense_multilinear_extension());
                        let output = DenseOrSparseMultilinearExtension::from(result);
                        output
                    }
                }
                
            },

            SMultilinearExtension(mle_lhs) => {
                match rhs {
                    DMultilinearExtension(mle_rhs) => {
                        let result = mle_lhs.add(&dense_to_sparse(mle_rhs));
                        let output = DenseOrSparseMultilinearExtension::from(result);
                        output
                      
                    },
                    SMultilinearExtension(mle_rhs) => {
                        let result = mle_lhs.add(&mle_rhs);
                        let output = DenseOrSparseMultilinearExtension::from(result);
                        output
                    }
                }
                
            },
        }
    }
}


impl<F: Field> AddAssign for DenseOrSparseMultilinearExtension<F> {
    fn add_assign(&mut self, other: Self) {
        *self = &*self + &other;
    }
}

impl<'a, 'b, F: Field> AddAssign<&'a DenseOrSparseMultilinearExtension<F>>
    for DenseOrSparseMultilinearExtension<F>
{
    fn add_assign(&mut self, other: &'a DenseOrSparseMultilinearExtension<F>) {
        *self = &*self + other;
    }
}

impl<F: Field> Neg for DenseOrSparseMultilinearExtension<F> {
    type Output = DenseOrSparseMultilinearExtension<F>;

    fn neg(self) -> Self::Output {
        match self {
            DMultilinearExtension(mle) => Self::from(mle.neg()),
            SMultilinearExtension(mle) => Self::from(mle.neg()),
        }
    }
}

impl<F: Field> Sub for DenseOrSparseMultilinearExtension<F> {
    type Output = DenseOrSparseMultilinearExtension<F>;

    fn sub(self, other: DenseOrSparseMultilinearExtension<F>) -> Self {
        &self - &other
    }
}

impl<'a, 'b, F: Field> Sub<&'a DenseOrSparseMultilinearExtension<F>> for &'b DenseOrSparseMultilinearExtension<F> {
    type Output = DenseOrSparseMultilinearExtension<F>;

    fn sub(self, rhs: &'a DenseOrSparseMultilinearExtension<F>) -> Self::Output {
        self + &rhs.clone().neg()
    }
}

impl<F: Field> SubAssign for DenseOrSparseMultilinearExtension<F> {
    fn sub_assign(&mut self, other: Self) {
        *self = &*self - &other;
    }
}

impl<'a, 'b, F: Field> SubAssign<&'a DenseOrSparseMultilinearExtension<F>>
    for DenseOrSparseMultilinearExtension<F>
{
    fn sub_assign(&mut self, other: &'a DenseOrSparseMultilinearExtension<F>) {
        *self = &*self - other;
    }
}

impl<F: Field> fmt::Debug for DenseOrSparseMultilinearExtension<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            DMultilinearExtension(mle) => mle.fmt(f),
            SMultilinearExtension(mle) => mle.fmt(f),
        }
    }
}
