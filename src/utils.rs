use ark_std::fmt::Formatter;
use ark_std::vec::Vec;

use ark_ff::{Field, PrimeField, Zero};

use ark_poly::{DenseMultilinearExtension, SparseMultilinearExtension};

// use ark_bls12_381::Fr;

use ark_serialize::CanonicalSerialize;

use ndarray::*;

use std::fmt;
use std::io::{BufWriter, Write, Error};
use std::fs::File;
use std::ops::AddAssign;
use std::vec;

use std::i64;

use std::cmp;

use rand::Rng;

use crate::LayerInfoConv;
use crate::LayerInfoDense;
use crate::data_structures::DenseOrSparseMultilinearExtension;

use crate::LayerInfo;



// the lifetime is here so we can use a reference and not move the value when we print it
pub struct DMLE<'a, T: Field>(pub &'a DenseMultilinearExtension<T>);

pub struct DMLElarge<'a, T: Field>(pub &'a DenseMultilinearExtension<T>);

// implementation of a wrapper around DenseMultilinearExtension to implement a Debug trait
// and allow somewhat readable printing
impl<F: Field + PrimeField> fmt::Debug for DMLE<'_, F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "DenseML(nv = {}, evaluations = [", self.0.num_vars)?;
        for i in 0..ark_std::cmp::min(256, self.0.evaluations.len()) {
            if let Ok(num) =
                i128::from_str_radix(&F::into_repr(&self.0.evaluations[i]).to_string(), 16)
            {
                write!(f, "{:?} ", num)?;
            } else {
                let num =
                    i128::from_str_radix(&F::into_repr(&-self.0.evaluations[i]).to_string(), 16)
                        .unwrap();
                write!(f, "-{:?} ", num)?;
            }
        }
        if self.0.evaluations.len() < 4 {
            write!(f, "])")?;
        } else {
            write!(f, "...])")?;
        }
        Ok(())
    }
}

// same as above but for large field numbers
impl<F: Field + PrimeField> fmt::Debug for DMLElarge<'_, F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "DenseML(nv = {}, evaluations = [", self.0.num_vars)?;
        for i in 0..ark_std::cmp::min(16, self.0.evaluations.len()) {
            write!(
                f,
                "{:?} ",
                &F::into_repr(&self.0.evaluations[i]).to_string()
            )?;
        }
        if self.0.evaluations.len() < 4 {
            write!(f, "])")?;
        } else {
            write!(f, "...])")?;
        }
        Ok(())
    }
}

// Computes the ceiling of the log of a dimension
// So we get the number of variables needed to represent its MLE
#[macro_export]
macro_rules! log2i {
    ($i: expr) => {
        ($i as f32).log2().ceil() as usize
    };
}

// Computes the binary reverse number of n (uint)
// with zeros padding to the left to N bits i.e.
// if n = 7 and N = 4 then bin(n) = 111, padded(bin(n),N) = 0111
// and reversed(padded(bin(n))) = 1110 = 14
// Used to determine the index of matrix values inside the MLE
#[macro_export]
macro_rules! rbin {
    ($n: expr, $nb_bits: expr) => {
        if $nb_bits == 0{
            0
        } else {
        ($n).reverse_bits() >> ((usize::BITS as usize) - $nb_bits)
        }
    };
}

// Easy way to create field elements (mostly for testing)
#[macro_export]
macro_rules! F {
    ($n: expr) => {
        Fr::from($n as i32)
    };
}

// Given the dimension of the convolution input, kernel and output
// computes the padding and the strides to make the computation work
#[macro_export]
macro_rules! get_padding_and_strides {
    ($di: expr, $dk: expr, $do: expr) => {
        if $do == 1{
            (0,1)
        } else {
            if ($di - $dk) / ($do - 1) < 1 {
                ($do + $dk - $di - 1, 1)
            } else {
                let s = ((($di - $dk) / ($do - 1)) as f32)
                        .ceil() as usize;
                (0, s)
            }
        }
    };
}

// Given the size of the convolution input, kernel, padding and strides
// computes output size of one dimension
#[macro_export]
macro_rules! get_output_size {
    ($di: expr, $dk: expr, $p: expr, $s: expr) => {
        ($di - $dk + $p) / $s + 1
    }
}

pub type Matrix<T> = Vec<Vec<T>>;

//Dimension of the input of a convolution (in, x, y)
pub type DimIC = (usize, usize, usize); 
//Dimension of the kernel of a convolution (in, out, x, y)
pub type DimKC = (usize, usize, usize, usize);
//Dimension of the output of a convolution (out, x, y)
pub type DimOC = (usize, usize, usize);

//Dimension of the input of a dense layer (1, x)
pub type DimID = (usize, usize); 
//Dimension of the kernel of a dense layer (x, y)
pub type DimKD = (usize, usize); 
//Dimension of the output of a dense layer (y, 1)
pub type DimOD = (usize, usize);

// Converts a matrix to a MLE representation
pub fn matrix_to_mle<F: Field>(
    matrix: Matrix<F>,
    size: (usize, usize),
    transposed: bool,
) -> (DenseMultilinearExtension<F>, usize, usize) {
    /*
    Take an input matrix and evaluate it as an MLE.
    Rows are concatenated in the MLE. Namely, the ordering of the elements is:
    [(0,0), (0,1), ..., (0, n_cols - 1), (1,0), ..., (n_rows - 1, n_cols - 1)]
    For n_cols and n_rows being a power of 2. Otherwise, we pad with zeroes.
    */
    let num_vars_0 = log2i!(size.0);
    let num_vars_1 = log2i!(size.1);

    let n = num_vars_0 + num_vars_1;

    // Initialize an MLE of zeros (field elements)
    let mut poly_evals = vec![F::zero(); 1 << n];
    if transposed {
        for j in 0..size.1 {
            let index = j * (1 << num_vars_0);
            for i in 0..size.0 {
                let rev_index = rbin!(index + i, n);
                poly_evals[rev_index] = matrix[i][j];
            }
        }
    } else {
        for i in 0..size.0 {
            let index = i * (1 << num_vars_1);
            for j in 0..size.1 {
                let rev_index = rbin!(index + j, n);
                poly_evals[rev_index] = matrix[i][j];
            }
        }
    }

    let poly = DenseMultilinearExtension::from_evaluations_vec(
        num_vars_0 + num_vars_1,
        poly_evals
    );
    // drop(matrix);
    return (poly, num_vars_0, num_vars_1);
}


// Reshape the input matrix of a convolution to perform the convolution
// as a matrix multiplication given a padding and strides
pub fn matrix_reshape<T: Clone + Zero + std::fmt::Debug>(
    input_matrix: &Vec<Matrix<T>>,
    input_dim: (usize, usize, usize),
    kernel_dim: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> (Vec<Matrix<T>>, (usize, usize, usize)) {
    let mut matrix_reshaped: Vec<Matrix<T>> = Vec::new();
    let (num_channels, n, m) = input_dim;
    let (np, mp) = kernel_dim;
    let (sx, sy) = stride;
    let (px, py) = padding;

    for channel in 0..num_channels {
        let pmatrix: Matrix<T>;
        let size: (usize, usize);
        if padding != (0, 0) {
            // Generates a padded matrix if needed
            (pmatrix, size) = matrix_padding(&input_matrix[channel], input_dim, padding);
        } else {
            (pmatrix, size) = (input_matrix[channel].clone(), (input_dim.1, input_dim.2));
        }

        let (n, m) = size;

        let mut matrix_reshaped_temp: Matrix<T> = Vec::new();
        // superblock loop
        for i in (0..(n - np + 1)).step_by(sx) {
            let mut temp_superblock: Vec<Vec<T>> = Vec::new();
            // block height loop
            for j in (0..(m - mp + 1)).step_by(sy) {
                let mut temp_block = Vec::new();
                // block length loop
                for k in 0..np {
                    // n*k => row of the OG matrix for the row of the block
                    // j => shift in the cols due to the block row
                    // n*i => shift of the rows due to the superblock
                    let row_index = k + i;
                    let col_index = j;
                    let mut matrix_slice = pmatrix[row_index][col_index..(col_index + mp)].to_vec();
                    temp_block.append(&mut matrix_slice);
                }
                temp_superblock.push(temp_block);
            }
            matrix_reshaped_temp.append(&mut temp_superblock);
        }
        matrix_reshaped.push(matrix_reshaped_temp);
    }

    (
        matrix_reshaped,
        (
            num_channels,
            ((n - np + px ) / sx + 1) * ((m - mp + py) / sy + 1),
            np * mp,
        ),
    )
}


pub fn matrix_padding<T: Clone + Zero>(
    input_matrix: &Matrix<T>,
    input_dim: (usize, usize, usize),
    padding: (usize, usize),
) -> (Matrix<T>, (usize, usize)) {
    let (_, n, m) = input_dim;
    let (px, py) = padding;

    let pb = px / 2; //bottom
    let pt = px - pb; //top
    let pr = py / 2; //right
    let pl = py - pr; //left

    let mut padded_matrix = Vec::<Vec<T>>::with_capacity(n + pt + pb);
    for _ in 0..pt {
        padded_matrix.push(vec![T::zero(); m + pl + pr])
    }
    for r in 0..n {
        let mut padded_row = vec![T::zero(); pl];
        padded_row.append(&mut input_matrix[r].to_vec());
        padded_row.append(&mut vec![T::zero(); pr]);

        padded_matrix.push(padded_row);
    }
    for _ in 0..pb {
        padded_matrix.push(vec![T::zero(); m + pl + pr])
    }

    (padded_matrix, (n + pt + pb, m + pl + pr))
}


pub fn matrix_reshape_predicate(
    dim_input: DimIC,
    dim_kernel: DimKC,
    padding: (usize, usize),
    strides: (usize,usize)
) -> (Vec<(usize, usize)>, DimIC){
    let (_, n, m) = dim_input;
    let (_, _, np, mp) = dim_kernel;
    let full_kernel = (np * mp).next_power_of_two(); 

    let (sx, sy) = strides;

    let (px, py) = padding;
    let pb = px / 2; //bottom
    let pt = px - pb; //top
    let pr = py / 2; //right
    let pl = py - pr; //left

    let mut index_change = Vec::new();

    // idx in reshape input
    let mut new_idx = 0;
    for i in (0..(n - np + px + 1)).step_by(sx) {
        for j in (0..(m - mp + py + 1)).step_by(sy) {
                for k in 0..np {
                for l in 0..mp {
                    let idx_x = i + k;
                    let idx_y = j + l;
                    if (idx_x >= pt) & (idx_x <= n + pt -1) & (idx_y >= pl) & (idx_y <= m + pl - 1) {
                        // idx in base input
                        let idx = (idx_x - pt) * m + (idx_y - pl);
                        index_change.push((
                            new_idx,
                            idx, 
                        ));
                    }
                    new_idx += 1;
                }
            }
            new_idx += full_kernel - np * mp;
        }
    }
    (
        index_change,
        (
            dim_input.0,
            ((n - np + px)/sx + 1) * ((m - mp + py)/sy + 1),
            np * mp,
        )
    )
}


// Performs a convolution operation between an  multichannel input matrix and
// a multichannel kernel matrix as matrix multiplications using Arrays
pub fn convolution_as_matmul<T>(
    input: Vec<Vec<Vec<T>>>,
    input_dim: (usize,usize,usize),
    kernel: Vec<Vec<Vec<Vec<T>>>>,
    output_dim: (usize,usize,usize),

) -> Array3<T>
    where
        T: Clone + Zero + LinalgScalar + AddAssign + std::fmt::Debug,
{

    let mut output = Array3::<T>::zeros(output_dim);

    for (_, (input_chin,kernel_chin)) in input.iter().zip(kernel.iter()).enumerate() {

        let input_array = from_matrix_to_arr2(input_chin.to_vec(), (input_dim.1, input_dim.2));

        for (chout, kernel_chout) in kernel_chin.iter().enumerate() {
            
            let kernel_vec = kernel_chout
                .clone()
                .into_iter()
                .flatten()
                .collect::<Vec<T>>();

            let kernel_array = Array::from_vec(kernel_vec.clone());
            let conv_result = input_array.dot(&kernel_array);

            let mut slice = output.slice_mut(s![chout,..,..]).into_shape(output_dim.1 * output_dim.2).unwrap();
            slice += &conv_result.clone();
        }
    }

    output
}


// Convert a slice of u32 into a 3D matrix of field element
pub fn matrix_encoder<F: Field>(values: &[u32], dim: (usize, usize, usize)) -> Vec<Matrix<F>> {
    let (channel, rows, cols) = dim;
    assert_eq!(rows * cols * channel, values.len());

    let mut mat = Vec::with_capacity(channel);
    for c in 0..channel {
        let mut row = Vec::with_capacity(rows);
        for i in 0..rows {
            let mut v = Vec::with_capacity(cols);
            for j in 0..cols {
                v.push(F::from(values[c * rows * cols + i * cols + j] as u32));
            }
            row.push(v);
        }
        mat.push(row)
    }
    mat
}


// Convert a slice of field element into a 2D Matrix
pub fn field_matrix_encoder<T: Copy>(values: &[T], rows: usize, cols: usize) -> Matrix<T> {
    assert_eq!(rows * cols, values.len());
    let mut mat = Vec::with_capacity(rows);
    for i in 0..rows {
        let mut v = Vec::with_capacity(cols);
        for j in 0..cols {
            v.push(values[i * cols + j]);
        }
        mat.push(v);
    }
    mat
}

// Convert a slice of field element into a 3D Matrix with multiple channels
pub fn field_matrix_encoder_multichannel<T: Copy>(
    values: &[T],
    dim: (usize, usize, usize),
) -> Vec<Matrix<T>> {
    let (channel, rows, cols) = dim;
    assert_eq!(rows * cols * channel, values.len());
    let mut mat = Vec::with_capacity(channel);
    for c in 0..channel {
        let mut row = Vec::with_capacity(rows);
        for i in 0..rows {
            let mut v = Vec::with_capacity(cols);
            for j in 0..cols {
                v.push(values[c * rows * cols + i * cols + j]);
            }
            row.push(v);
        }
        mat.push(row)
    }
    mat
}

// Convert a slice of field element into a 3D matrix with multiple channels
// following the dimension order: dim_in, dim_out, xy
pub fn field_kernel_encoder_multichannel<T: Copy>(
    values: &[T],
    dim: (usize, usize, usize, usize),
) -> Vec<Vec<Vec<T>>> {
    let (channel_in, channel_out, rows, cols) = dim;
    assert_eq!(rows * cols * channel_in * channel_out, values.len());
    let mut mat = Vec::with_capacity(channel_in);
    for cin in 0..channel_in {
        let mut chanout = Vec::with_capacity(channel_out);
        for cout in 0..channel_out {
            let mut row = Vec::with_capacity(rows * cols);
            for i in 0..rows {
                for j in 0..cols {
                    row.push(
                        values[cin * channel_out * rows * cols + cout * rows * cols + i * cols + j],
                    );
                }
            }
            chanout.push(row);
        }
        mat.push(chanout);
    }
    mat
}

// Convert a slice of field element into a 4D Matrix with multiple channels
// following the dimension order: dim_in, dim_out, x, y
pub fn field_kernel_encoder_multichannel_matrix_for_conv<T: Copy>(
    values: &[T],
    dim: (usize, usize, usize, usize),
) -> Vec<Vec<Vec<Vec<T>>>> {
    let (channel_in, channel_out, rows, cols) = dim;
    assert_eq!(rows * cols * channel_in * channel_out, values.len());
    let mut mat = Vec::with_capacity(channel_in);
    for cin in 0..channel_in {
        let mut chanout = Vec::with_capacity(channel_out);
        for cout in 0..channel_out {
            let mut row = Vec::with_capacity(rows);
            for i in 0..rows {
                let mut col = Vec::with_capacity(cols);
                for j in 0..cols {
                    col.push(
                        values[cin * channel_out * rows * cols + cout * rows * cols + i * cols + j],
                    );
                }
                row.push(col);
            }
            chanout.push(row);
        }
        mat.push(chanout);
    }
    mat
}



// Creates an MLE of the output of the convolution with the correct variable order
pub fn layer_output_to_mle<F: PrimeField + Field>(
    output: &Vec<Matrix<F>>,
    dim: (usize, usize, usize),
    sparse: bool,
) -> DenseOrSparseMultilinearExtension<F> {
    let (dimch, dimx, dimy) = dim;
    let dimxy = dimx * dimy;
    let dimxy_vars = log2i!(dimxy);
    let ch_vars = log2i!(dimch);

    let num_vars = dimxy_vars + ch_vars;

    let mut output_mle_dense = vec![F::zero(); 1 << (num_vars)];
    let mut output_mle_sparse = Vec::<(usize, F)>::new();

    // Y(x,y,tau)
    for x in 0..dimx {
        let index_x = x * dimy * (1 << ch_vars); 
        for y in 0..dimy {
            let index_y = y * (1 << ch_vars);
            for tau in 0..dimch {
                let rev_index = rbin!(index_x + index_y + tau, num_vars);
                if sparse {
                    output_mle_sparse.push((rev_index, output[tau][x][y]));
                }
                else {
                    output_mle_dense[rev_index] = output[tau][x][y];
                }
            }
        }
    }

    if sparse {
        return DenseOrSparseMultilinearExtension::from(
        SparseMultilinearExtension::from_evaluations(num_vars, &output_mle_sparse)
    );
    }
    return DenseOrSparseMultilinearExtension::from(
        DenseMultilinearExtension::from_evaluations_vec(num_vars, output_mle_dense)
    );
}


// Creates an MLE of the flattened output of the convolution with the correct variable order
pub fn conv_layer_output_to_flatten<F: PrimeField + Field>(
    output: &Vec<Matrix<F>>,
    dim: (usize, usize, usize),
) -> Vec<F> {
    let (dimch, dimx, dimy) = dim;
    let dimxy = dimx * dimy;
    let dimxy_vars = log2i!(dimxy);
    let ch_vars = log2i!(dimch);

    let num_vars = dimxy_vars + ch_vars;


    let mut flatten_output = vec![F::zero(); 1 << num_vars];

    // Y(x,y,tau)
    for x in 0..dimx {
        let index_x = x * dimy * (1 << ch_vars); 
        for y in 0..dimy {
            let index_y = y * (1 << ch_vars);
            for tau in 0..dimch {
                flatten_output[index_x + index_y + tau] = output[tau][x][y];
            }
        }
    }

    flatten_output
}


// Creates an MLE of the input of the convolution
pub fn input_to_mle<F: PrimeField + Field>(
    input: &Vec<Matrix<F>>,
    dim: (usize, usize, usize),
) -> DenseMultilinearExtension<F> {
    let (dimch, dimx, dimy) = dim;
    let dimxy = dimx * dimy;
    let dimxy_vars = log2i!(dimxy);
    let ch_vars = log2i!(dimch);

    let num_vars = dimxy_vars + ch_vars;

    let mut input_mle = vec![F::zero(); 1 << (num_vars)];

    // Write the polynomial X(x, y, sigma)
    for x in 0..dimx {
        let index_x = x * dimy * (1 << (ch_vars));
        for y in 0..dimy {
            let index_y = y * (1 << ch_vars);
            for sigma in 0..dimch {
                let rev_index = rbin!(index_x + index_y + sigma, num_vars);
                input_mle[rev_index] = input[sigma][x][y];
            }
        }
    }

    return DenseMultilinearExtension::from_evaluations_vec(num_vars, input_mle);
}


// Creates an MLE of the multichannel kernel of the convolution 
// where the inside kernels are flattened
pub fn kernel_processing<F: Field>(kernel: Vec<Vec<Vec<F>>>, dim_kernel: DimKC) -> DenseOrSparseMultilinearExtension<F>{
    let ch_in = dim_kernel.0;
    let ch_out = dim_kernel.1;
    let mx = dim_kernel.2;
    let my = dim_kernel.3;

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
                kernel_reshaped[rev_index] = kernel[sigma][tau][y];
            }
        }
    }
    let kernel_mle = DenseMultilinearExtension::from_evaluations_vec(total_nv, kernel_reshaped);
    DenseOrSparseMultilinearExtension::from(kernel_mle)
}


// Creates a Sparse MLE of the predicate function given the predicate and 
// the input and reshaped input dimensions
pub fn predicate_processing<F: Field>(
    predicate: Vec<(usize,usize)>, 
    dim_input: DimIC, 
    dim_input_reshape: DimIC,
) -> DenseOrSparseMultilinearExtension<F> {

    let x = dim_input.1;
    let y = dim_input.2;
    let input_size = x * y;
    let input_size_nv = log2i!(input_size);

    let xr = dim_input_reshape.1;
    let yr = dim_input_reshape.2;
    let input_reshape_size_nv = log2i!(xr) + log2i!(yr); 

    let total_nv = input_size_nv + input_reshape_size_nv;

    let mut predicate_mle: Vec<(usize, F)> = Vec::new();

    // Encodes P((x, y), z)
    // xy = new_idx
    // z = idx
    for (xy, z) in predicate.into_iter() {
        let rev_index = rbin!(xy * (1 << (input_size_nv)) + z, total_nv);
        predicate_mle.push((rev_index, F::one()))
    }

    DenseOrSparseMultilinearExtension::from(SparseMultilinearExtension::from_evaluations(total_nv, &predicate_mle))
}


// A naive way to convert DenseMLE to SparseMLE
// To be replace by direct sparse construction in the future
pub fn dense_to_sparse<F: Field>(dmle: &DenseMultilinearExtension<F>) -> SparseMultilinearExtension<F> {
    let mut sparse_vec = Vec::new();
    for (i, e) in dmle.evaluations.iter().enumerate().filter(|(_, &e)| e != F::zero()) {
        sparse_vec.push((i,e.clone()));
    }
    let sparse_mle = SparseMultilinearExtension::from_evaluations(dmle.num_vars, &sparse_vec);
    sparse_mle
}


// FUNCTION TAKEN FROM CRATE IPforMLSC
// interpolate a uni-variate degree-`p_i.len()-1` polynomial and evaluate this
// polynomial at `eval_at`:
//   \sum_{i=0}^len p_i * (\prod_{j!=i} (eval_at - j)/(i-j))
pub fn interpolate_uni_poly<F: Field>(p_i: &[F], eval_at: F) -> F {
    let len = p_i.len();

    let mut evals = vec![];

    let mut prod = eval_at;
    evals.push(eval_at);

    // `prod = \prod_{j} (eval_at - j)`
    for e in 1..len {
        let tmp = eval_at - F::from(e as u64);
        evals.push(tmp);
        prod *= tmp;
    }
    let mut res = F::zero();
    // we want to compute \prod (j!=i) (i-j) for a given i
    //
    // we start from the last step, which is
    //  denom[len-1] = (len-1) * (len-2) * ... * 2 * 1
    // the step before that is
    //  denom[len-2] = (len-2) * (len-3) * ... * 2 * 1 * -1
    // and the step before that is
    //  denom[len-3] = (len-3) * (len-4) * ... * 2 * 1 * -1 * -2
    //
    // i.e., for any i, the one before this will be derived from
    //  denom[i-1] = - denom[i] * (len-i) / i
    //
    // that is, we only need to store
    // - the last denom for i = len-1, and
    // - the ratio between the current step and the last step, which is the
    //   product of -(len-i) / i from all previous steps and we store
    //   this product as a fraction number to reduce field divisions.

    // We know
    //  - 2^61 < factorial(20) < 2^62
    //  - 2^122 < factorial(33) < 2^123
    // so we will be able to compute the ratio
    //  - for len <= 20 with i64
    //  - for len <= 33 with i128
    //  - for len >  33 with BigInt
    if p_i.len() <= 20 {
        let last_denom = F::from(u64_factorial(len - 1));
        let mut ratio_numerator = 1i64;
        let mut ratio_enumerator = 1u64;

        for i in (0..len).rev() {
            let ratio_numerator_f = if ratio_numerator < 0 {
                -F::from((-ratio_numerator) as u64)
            } else {
                F::from(ratio_numerator as u64)
            };

            res += p_i[i] * prod * F::from(ratio_enumerator)
                / (last_denom * ratio_numerator_f * evals[i]);

            // compute ratio for the next step which is current_ratio * -(len-i)/i
            if i != 0 {
                ratio_numerator *= -(len as i64 - i as i64);
                ratio_enumerator *= i as u64;
            }
        }
    } else if p_i.len() <= 33 {
        let last_denom = F::from(u128_factorial(len - 1));
        let mut ratio_numerator = 1i128;
        let mut ratio_enumerator = 1u128;

        for i in (0..len).rev() {
            let ratio_numerator_f = if ratio_numerator < 0 {
                -F::from((-ratio_numerator) as u128)
            } else {
                F::from(ratio_numerator as u128)
            };

            res += p_i[i] * prod * F::from(ratio_enumerator)
                / (last_denom * ratio_numerator_f * evals[i]);

            // compute ratio for the next step which is current_ratio * -(len-i)/i
            if i != 0 {
                ratio_numerator *= -(len as i128 - i as i128);
                ratio_enumerator *= i as u128;
            }
        }
    } else {
        // since we are using field operations, we can merge
        // `last_denom` and `ratio_numerator` into a single field element.
        let mut denom_up = field_factorial::<F>(len - 1);
        let mut denom_down = F::one();

        for i in (0..len).rev() {
            res += p_i[i] * prod * denom_down / (denom_up * evals[i]);

            // compute denom for the next step is -current_denom * (len-i)/i
            if i != 0 {
                denom_up *= -F::from((len - i) as u64);
                denom_down *= F::from(i as u64);
            }
        }
    }

    res
}

// compute the factorial(a) = 1 * 2 * ... * a
#[inline]
fn field_factorial<F: Field>(a: usize) -> F {
    let mut res = F::one();
    for i in 1..=a {
        res *= F::from(i as u64);
    }
    res
}

// compute the factorial(a) = 1 * 2 * ... * a
#[inline]
fn u128_factorial(a: usize) -> u128 {
    let mut res = 1u128;
    for i in 1..=a {
        res *= i as u128;
    }
    res
}

// compute the factorial(a) = 1 * 2 * ... * a
#[inline]
fn u64_factorial(a: usize) -> u64 {
    let mut res = 1u64;
    for i in 1..=a {
        res *= i as u64;
    }
    res
}

// Function for printing a small field element
// if a value exceed the limit of u128 prints 1234567890
pub fn display_small_field_elem<F: Field + PrimeField>(elem: F) {
    if let Ok(num) = i128::from_str_radix(&F::into_repr(&elem).to_string(), 16) {
        println!("{:?}", num);
    } else {
        let num = i128::from_str_radix(&F::into_repr(&-elem).to_string(), 16).unwrap();
        println!("-{:?}", num);
    }
}

// Function for printing a vectore of small field element
// if a value exceed the limit of u128 prints 1234567890
pub fn display_vec_small_field_elem<F: Field + PrimeField>(vec_elem: &Vec<F>) {
    println!(
        "{:?}",
        vec_elem
            .iter()
            .map(
                |&e| if let Ok(num) = i128::from_str_radix(&F::into_repr(&e).to_string(), 16) {
                    num
                } else {
                    let num = i128::from_str_radix(
                        &("-".to_owned() + &F::into_repr(&-e).to_string()),
                        16,
                    )
                    .unwrap();
                    num
                } 
            )
            .collect::<Vec<i128>>()
    );
}

// Function for printing any field element
pub fn display_field_elem<F: Field + PrimeField>(elem: &F) {
    println!("{:?}", F::into_repr(elem).to_string());
}

// Function for printing a vector of any field element
pub fn display_vec_field_elem<F: Field + PrimeField>(vec_elem: &Vec<F>) {
    println!(
        "{:?}",
        vec_elem
            .iter()
            .map(|e| F::into_repr(e).to_string())
            .collect::<Vec<String>>()
    );
}

// Displays a matrix of elements having the Debug trait
pub fn display_matrix<T: fmt::Debug>(matrix: &Vec<Vec<T>>) {
    println!();
    for i in matrix.into_iter() {
        println!("{:?}", i);
    }
}

// Displays a matrix of potentially large field elements
pub fn display_matrix_field<F: Field + PrimeField + fmt::Debug>(matrix: &Vec<Vec<F>>) {
    for i in matrix.into_iter() {
        display_vec_field_elem(i);
    }
    println!();
}

// Displays a matrix of small field elements or crashes
pub fn display_matrix_small_field<F: Field + PrimeField + fmt::Debug>(matrix: &Vec<Vec<F>>) {
    for i in matrix.into_iter() {
        display_vec_small_field_elem(i);
    }
    println!();
}

// Displays a vector of matrices of elements having the Debug trait
// Used for multichannel inputs
pub fn display_vec_matrix<T: fmt::Debug>(matrix: &Vec<Vec<Vec<T>>>) {
    println!();
    for i in matrix.into_iter() {
        display_matrix(i);
    }
}

// Displays a vector of matrices of potentially large field elements
// Used for multichannel inputs
pub fn display_vec_matrix_field<F: Field + PrimeField + fmt::Debug>(matrix: &Vec<Vec<Vec<F>>>) {
    for i in matrix.into_iter() {
        display_matrix_field(i);
    }
    println!();
}

// Displays a vector of matrices of small field elements
// Used for multichannel inputs
pub fn display_vec_matrix_small_field<F: Field + PrimeField + fmt::Debug>(
    matrix: &Vec<Vec<Vec<F>>>,
) {
    for i in matrix.into_iter() {
        display_matrix_small_field(i);
    }
    println!();
}

// Displays a vector of vector of matrices of small field elements
// Used for multichannel inputs and outputs
pub fn display_vec_vec_matrix_small_field<F: Field + PrimeField + fmt::Debug>(
    vec_vec_matrix: &Vec<Vec<Vec<Vec<F>>>>,
) {
    for i in vec_vec_matrix.into_iter() {
        for j in i.into_iter() {
            display_matrix_small_field(j);
        }
    }
    println!();
}

// Convert from a matrix to an Array2
pub fn from_matrix_to_arr2<T: Zero + Clone>(
    matrix: Matrix<T>,
    input_dim: (usize, usize),
) -> Array2<T> {
    let (n, m) = input_dim;
    let mut arr = Array2::<T>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            arr[[i, j]] = matrix[i][j].clone();
        }
    }
    arr
}


// Convert from an Array2 to a matrix
pub fn from_arr2_to_matrix<T: Zero + Clone>(
    arr: Array2<T>,
    input_dim: (usize, usize),
) -> Matrix<T> {
    let (n, m) = input_dim;
    let mut mat = Vec::new();
    for i in 0..n {
        let mut v = Vec::new();
        for j in 0..m {
            v.push(arr[[i,j]].clone());
        }
        mat.push(v);
    }
    mat
}


// Convert from a 3D matrix to an Array3
pub fn input_to_arr3<T: Zero + Clone>(
    input: Vec<Matrix<T>>,
    input_dim: (usize, usize, usize),
) -> Array3<T> {
    let (channel, n, m) = input_dim;
    let mut arr = Array3::<T>::zeros((channel, n, m));
    for c in 0..channel {
        for i in 0..n {
            for j in 0..m {
                arr[[c, i, j]] = input[c][i][j].clone();
            }
        }
    }
    arr
}

// Creates an array to compute the convolution
// Reverse the channel in and out
pub fn kernel_to_array4<T: Zero + Clone>(
    kernel: Vec<Vec<Vec<Vec<T>>>>,
    dim: (usize, usize, usize, usize),
    permuted_in_out: bool,
) -> Array4<T> {
    let (dcin, dcout, dx, dy) = dim;
    let mut arr; 
    if permuted_in_out {
        arr = Array4::<T>::zeros((dcout, dcin, dx, dy));
    } else {
        arr = Array4::<T>::zeros((dcin, dcout, dx, dy));
    }
    for i in 0..dcin {
        for o in 0..dcout {
            for x in 0..dx {
                for y in 0..dy {
                    if permuted_in_out {
                        arr[[o, i, x, y]] = kernel[i][o][x][y].clone();
                    } else {
                        arr[[i, o, x, y]] = kernel[i][o][x][y].clone();
                    }
                }
            }
        }
    }
    arr
}

// Vector to Array1
pub fn bias_to_arr1<T: Zero + Clone>(bias: Vec<T>) -> Array1<T> {
    let dim = bias.len();
    let mut arr = Array1::<T>::zeros(dim);
    for d in 0..dim {
        arr[[d]] = bias[d].clone();
    }
    arr
}

// Convert a convolution result as Array3 to a 3D matrix
pub fn conv_result_to_matrix<T: Zero + Clone>(
    conv: Array3<T>,
    conv_size: (usize, usize, usize),
) -> Vec<Matrix<T>> {
    let (channel_out, n, m) = conv_size;
    let mut matrix = vec![vec![vec![T::zero(); m]; n]; channel_out];
    for c in 0..channel_out {
        for i in 0..n {
            for j in 0..m {
                matrix[c][i][j] = conv[[c, i, j]].clone();
            }
        }
    }
    matrix
}


// Generate a description of a convolutional neural network evaluation
// creating a random input and kernels and adapting the padding and strides
// to fit the random output dimensions generated
// Used mostly for easier testing 
pub fn generate_random_CNN_execution<F: Field + PrimeField, R: Rng>(
    dim_first_input: DimIC, 
    dim_first_kernel: DimKC, 
    depth: usize,
    mut rng: R
) -> Result<Vec<LayerInfoConv<F>>, Error>{

    let mut layers = Vec::new();

    // Generating the values
    let mut dim_input: (usize, usize, usize) = dim_first_input;
    let mut dim_kernel: (usize, usize, usize, usize) = dim_first_kernel;

    assert_eq!(dim_input.0, dim_kernel.0, "number of input channels not consistant");
    assert!(dim_input.1 >= dim_kernel.2, "kernel larger than the input");

    // generate random output size
    let dim_outputx = rng.gen_range((1+dim_input.1-dim_kernel.2)..=dim_input.1);
    let dim_outputy = rng.gen_range((1+dim_input.2-dim_kernel.3)..=dim_input.2);

    let mut dim_output = (
        dim_kernel.1,
        dim_outputx,
        dim_outputy,
    );

    // get the padding and strides to match the output size
    let (paddingx,stridex) = get_padding_and_strides!(dim_input.1, dim_kernel.2, dim_output.1);
    let (paddingy,stridey) = get_padding_and_strides!(dim_input.2, dim_kernel.3, dim_output.2);
    let strides: (usize, usize) = (stridex, stridey);
    let padding: (usize, usize) = (paddingx,paddingy);


    let mut input: Vec<F> = Vec::with_capacity(
        dim_input.0 * dim_input.1 * dim_input.2);
    let mut kernel_vec = Vec::with_capacity(
        dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3);

    for _ in 0..(dim_input.0 * dim_input.1 * dim_input.2) {
        input.push(F::rand(&mut rng));
    }
    for _ in 0..(dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3) {
        kernel_vec.push(F::rand(&mut rng));
    }

    let input_for_prover = field_matrix_encoder_multichannel(input.as_slice(), dim_input);
    let kernel_for_prover = field_kernel_encoder_multichannel(kernel_vec.as_slice(), dim_kernel);

    let mut conv_result_matrix = full_convolution_computation::<F>(
        input, 
        kernel_vec, 
        dim_input, 
        dim_kernel, 
        dim_output, 
        strides, 
        padding);

    layers.push(LayerInfoConv {
        name: "Conv1".to_string(),
        input: input_for_prover,
        kernel: kernel_for_prover,
        output: conv_result_matrix.clone(),
        dim_input: dim_input,
        dim_kernel: dim_kernel,
        dim_output: dim_output,
        padding: padding,
        strides: strides
    });

    println!("Description of the network:");

    println!("input size = {:?}", dim_input);
    println!("input reshape size = {:?}", (dim_kernel.0, 
        dim_output.1 * dim_output.2, 
        dim_kernel.2 * dim_kernel.3));
    println!("kernel size = {:?}", dim_kernel);
    println!("output size = {:?}", dim_output);
    println!("padding = {:?}", padding);
    println!("strides = {:?}\n", strides);


    for d in 1..depth {

        dim_input = dim_output;


        let dim_chout = rng.gen_range(1..=dim_input.0);
        let mut mink = 1;
        let dim_wx = cmp::min(8, rng.gen_range(mink..dim_input.1));
        if dim_wx == mink {
            mink = 2;
        }
        let dim_wy = cmp::min(8, rng.gen_range(mink..dim_input.2));

        dim_kernel = (dim_input.0, dim_chout, dim_wx, dim_wy);

        let conv_input= conv_result_matrix.clone();

        let input = conv_input.clone().into_iter().flatten().flatten().collect();

        let mut conv_kernel_vec =
            Vec::with_capacity(dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3);
    
        for _ in 0..(dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3) {
            conv_kernel_vec.push(F::rand(&mut rng));
        }

        let dim_outputx = rng.gen_range((1 + dim_input.1 - dim_kernel.2)..=dim_input.1);
        let dim_outputy = rng.gen_range((1 + dim_input.2 - dim_kernel.3)..=dim_input.2);

        dim_output = (
            dim_kernel.1,
            dim_outputx,
            dim_outputy
        );

        println!("input size = {:?}", dim_input);
        println!("input reshape size = {:?}", (dim_kernel.0, 
            dim_output.1 * dim_output.2, 
            dim_kernel.2 * dim_kernel.3));
        println!("kernel size = {:?}", dim_kernel);
        println!("output size = {:?}", dim_output);
        
        let (paddingx,stridex) = get_padding_and_strides!(dim_input.1, dim_kernel.2, dim_output.1);
        let (paddingy,stridey) = get_padding_and_strides!(dim_input.2, dim_kernel.3, dim_output.2);
        let strides: (usize, usize) = (stridex, stridey);
        let padding: (usize, usize) = (paddingx,paddingy);
        
        println!("padding = {:?}", padding);
        println!("strides = {:?}\n", strides);

        let kernel_for_prover = field_kernel_encoder_multichannel(conv_kernel_vec.as_slice(), dim_kernel);
    
        conv_result_matrix = full_convolution_computation::<F>(
            input, 
            conv_kernel_vec, 
            dim_input, 
            dim_kernel, 
            dim_output, 
            strides, 
            padding);
    
        layers.push(LayerInfoConv {
            name: format!("Conv{}", d+1),
            input: conv_input,
            kernel: kernel_for_prover,
            output: conv_result_matrix.clone(),
            dim_input: dim_input,
            dim_kernel: dim_kernel,
            dim_output: dim_output,
            padding: padding,
            strides: strides
        });
    }

    Ok(layers)
}


// Generate a description of a neural network evaluation with dense and conv layers
// creating a random input and kernels and adapting the padding and strides
// to fit the random output dimensions generated
// Used mostly for easier testing 
pub fn generate_random_CNN_dense_execution<F: Field + PrimeField, R: Rng>(
    dim_first_input: DimIC, 
    dim_first_kernel: DimKC, 
    depth: usize,
    nb_dense_layers: usize,
    mut rng: R
) -> Result<Vec<LayerInfo<F>>, Error>{

    let mut layers = Vec::new();

    // Generating the values
    let mut dim_input: (usize, usize, usize) = dim_first_input;
    let mut dim_kernel: (usize, usize, usize, usize) = dim_first_kernel;

    assert_eq!(dim_input.0, dim_kernel.0, "number of input channels not consistant");
    assert!(dim_input.1 >= dim_kernel.2, "kernel larger than the input");

    // generate random output size
    let dim_outputx = rng.gen_range((1+dim_input.1-dim_kernel.2)..=dim_input.1);
    let dim_outputy = rng.gen_range((1+dim_input.2-dim_kernel.3)..=dim_input.2);

    let mut dim_output = (
        dim_kernel.1,
        dim_outputx,
        dim_outputy,
    );

    // get the padding and strides to match the output size
    let (paddingx,stridex) = get_padding_and_strides!(dim_input.1, dim_kernel.2, dim_output.1);
    let (paddingy,stridey) = get_padding_and_strides!(dim_input.2, dim_kernel.3, dim_output.2);
    let strides: (usize, usize) = (stridex, stridey);
    let padding: (usize, usize) = (paddingx,paddingy);


    let mut input: Vec<F> = Vec::with_capacity(
        dim_input.0 * dim_input.1 * dim_input.2);
    let mut kernel_vec = Vec::with_capacity(
        dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3);

    for _ in 0..(dim_input.0 * dim_input.1 * dim_input.2) {
        input.push(F::rand(&mut rng));
    }
    for _ in 0..(dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3) {
        kernel_vec.push(F::rand(&mut rng));
    }

    let input_for_prover = field_matrix_encoder_multichannel(input.as_slice(), dim_input);
    let kernel_for_prover = field_kernel_encoder_multichannel(kernel_vec.as_slice(), dim_kernel);

    let mut conv_result_matrix = full_convolution_computation::<F>(
        input, 
        kernel_vec, 
        dim_input, 
        dim_kernel, 
        dim_output, 
        strides, 
        padding);

    layers.push(LayerInfo::LIC(LayerInfoConv {
        name: "Conv1".to_string(),
        input: input_for_prover,
        kernel: kernel_for_prover,
        output: conv_result_matrix.clone(),
        dim_input: dim_input,
        dim_kernel: dim_kernel,
        dim_output: dim_output,
        padding: padding,
        strides: strides
    }));

    println!("Description of the network:");

    println!("input size = {:?}", dim_input);
    println!("input reshape size = {:?}", (dim_kernel.0, 
        dim_output.1 * dim_output.2, 
        dim_kernel.2 * dim_kernel.3));
    println!("kernel size = {:?}", dim_kernel);
    println!("output size = {:?}", dim_output);
    println!("padding = {:?}", padding);
    println!("strides = {:?}\n", strides);


    for d in 1..depth {

        dim_input = dim_output;


        let dim_chout = rng.gen_range(1..=dim_input.0);
        let mut mink = 1;
        let dim_wx = cmp::min(8, rng.gen_range(mink..dim_input.1));
        if dim_wx == mink {
            mink = 2;
        }
        let dim_wy = cmp::min(8, rng.gen_range(mink..dim_input.2));

        dim_kernel = (dim_input.0, dim_chout, dim_wx, dim_wy);

        let conv_input= conv_result_matrix.clone();

        let input = conv_input.clone().into_iter().flatten().flatten().collect();

        let mut conv_kernel_vec =
            Vec::with_capacity(dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3);
    
        for _ in 0..(dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3) {
            conv_kernel_vec.push(F::rand(&mut rng));
        }

        let dim_outputx = rng.gen_range((1 + dim_input.1 - dim_kernel.2)..=dim_input.1);
        let dim_outputy = rng.gen_range((1 + dim_input.2 - dim_kernel.3)..=dim_input.2);

        dim_output = (
            dim_kernel.1,
            dim_outputx,
            dim_outputy
        );

        println!("input size = {:?}", dim_input);
        println!("input reshape size = {:?}", (dim_kernel.0, 
            dim_output.1 * dim_output.2, 
            dim_kernel.2 * dim_kernel.3));
        println!("kernel size = {:?}", dim_kernel);
        println!("output size = {:?}", dim_output);
        
        let (paddingx,stridex) = get_padding_and_strides!(dim_input.1, dim_kernel.2, dim_output.1);
        let (paddingy,stridey) = get_padding_and_strides!(dim_input.2, dim_kernel.3, dim_output.2);
        let strides: (usize, usize) = (stridex, stridey);
        let padding: (usize, usize) = (paddingx,paddingy);
        
        println!("padding = {:?}", padding);
        println!("strides = {:?}\n", strides);

        let kernel_for_prover = field_kernel_encoder_multichannel(conv_kernel_vec.as_slice(), dim_kernel);
    
        conv_result_matrix = full_convolution_computation::<F>(
            input, 
            conv_kernel_vec, 
            dim_input, 
            dim_kernel, 
            dim_output, 
            strides, 
            padding);
    
        layers.push(LayerInfo::LIC(LayerInfoConv {
            name: format!("Conv{}", d+1),
            input: conv_input,
            kernel: kernel_for_prover,
            output: conv_result_matrix.clone(),
            dim_input: dim_input,
            dim_kernel: dim_kernel,
            dim_output: dim_output,
            padding: padding,
            strides: strides
     
        }));
    }
    
    let flatten_cnn_output = vec![conv_result_matrix
        .clone()
        .into_iter()
        .flatten()
        .flatten()
        .collect::<Vec<F>>()];

    let mut dense_input = flatten_cnn_output.clone();

    for d in 0..nb_dense_layers {


        let flatten_dim = dense_input[0].len();


        let dim_wx = flatten_dim;
        let dim_wy = rng.gen_range(4..256);

        let mut W_dense = Vec::new();

        for _ in 0..dim_wy*dim_wx {
            W_dense.push(F::rand(&mut rng));
        }

        let W_dense = field_matrix_encoder(W_dense.as_slice(), dim_wx, dim_wy);

        let W_dense_arr = from_matrix_to_arr2(W_dense.clone(), (dim_wx, dim_wy));

        let fco = from_matrix_to_arr2(dense_input.clone(), (1, flatten_dim));
    
        let output_arr = fco.dot(&W_dense_arr);
    
        let dense_output = from_arr2_to_matrix(output_arr, (1,dim_wy));
    

        layers.push(
            LayerInfo::<_>::LID( LayerInfoDense { 
                name: format!("Dense{}", d+1), 
                input: vec![dense_input.clone()], 
                kernel: W_dense, 
                output: dense_output.clone(), 
                dim_input: (1, dim_wx), 
                dim_kernel: (dim_wx, dim_wy), 
                dim_output: (1, dim_wy) 
            })
        );

        dense_input = dense_output;

    }

    Ok(layers)
}


// Computes the result of a convolution as a vec of matrices
// Used mostly for easier testing 
pub fn full_convolution_computation<T: Field>(
    input: Vec<T>, 
    kernel: Vec<T>, 
    dim_input: DimIC, 
    dim_kernel: DimKC, 
    dim_output: DimOC,
    strides: (usize,usize),
    padding: (usize,usize)
) -> Vec<Vec<Vec<T>>> {
    
    let input_mat = field_matrix_encoder_multichannel(input.as_slice(), dim_input);

    // kernel version to use to compute the output
    let kernel_for_conv = field_kernel_encoder_multichannel_matrix_for_conv(
            kernel.as_slice(), 
            dim_kernel
    );

    // Computing the convolution
    let (reshaped_mat, dim_reshape) =
        matrix_reshape(
            &input_mat, 
            dim_input, 
            (dim_kernel.2, dim_kernel.3), 
            strides, 
            padding);


    let conv_result = convolution_as_matmul(
        reshaped_mat, 
        dim_reshape, 
        kernel_for_conv, 
        dim_output
    );
    conv_result_to_matrix(conv_result.clone(), dim_output).clone()

    
}



// Generate a description of a VGG11 evaluation using random field elements
// The dimension of the dense layers is reduced to 2048 to avoid memory issues
// creating a random input and kernels and adapting the padding and strides
// to fit the random output dimensions generated
// Used mostly for easier testing 
pub fn generate_random_VGG11_execution<F: Field + PrimeField, R: Rng>(
    mut rng: R
) -> Result<Vec<LayerInfoDense<F>>, Error>{

    let mut layers = Vec::new();

    // Generating the values
    let dim_input: (usize, usize, usize) = (3,224,224);
    let dim_kernel: (usize, usize, usize, usize) = (3,64,3,3);

    assert_eq!(dim_input.0, dim_kernel.0, "number of input channels not consistant");
    assert!(dim_input.1 >= dim_kernel.2, "kernel larger than the input");


    let strides: (usize, usize) = (2, 2);
    let padding: (usize, usize) = (1, 1);
    // let padding: (usize, usize) = (0,0);

    // generate random output size
    let dim_outputx = (dim_input.1 - dim_kernel.2 + padding.0) / strides.0 + 1;
    let dim_outputy = (dim_input.2 - dim_kernel.3 + padding.1) / strides.1 + 1;

    let dim_output = (
        dim_kernel.1,
        dim_outputx,
        dim_outputy,
    );

    let mut input: Vec<F> = Vec::with_capacity(
        dim_input.0 * dim_input.1 * dim_input.2);
    let mut kernel_vec = Vec::with_capacity(
        dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3);

    for _ in 0..(dim_input.0 * dim_input.1 * dim_input.2) {
        input.push(F::rand(&mut rng));
    }
    for _ in 0..(dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3) {
        kernel_vec.push(F::rand(&mut rng));
    }

    let input_for_prover = field_matrix_encoder_multichannel(input.as_slice(), dim_input);
    let kernel_for_prover = field_kernel_encoder_multichannel(kernel_vec.as_slice(), dim_kernel);

    let conv_result_matrix = full_convolution_computation::<F>(
        input, 
        kernel_vec, 
        dim_input, 
        dim_kernel, 
        dim_output, 
        strides, 
        padding);

    layers.push(LayerInfo::<_>::LIC(LayerInfoConv {
        name: "Conv1_64".to_string(),
        input: input_for_prover,
        kernel: kernel_for_prover,
        output: conv_result_matrix.clone(),
        dim_input: dim_input,
        dim_kernel: dim_kernel,
        dim_output: dim_output,
        padding: padding,
        strides: strides
    }));

    println!("Description of the Layer:");

    println!("input size = {:?}", dim_input);
    println!("input reshape size = {:?}", (dim_kernel.0, 
        dim_output.1 * dim_output.2, 
        dim_kernel.2 * dim_kernel.3));
    println!("kernel size = {:?}", dim_kernel);
    println!("output size = {:?}", dim_output);
    println!("padding = {:?}", padding);
    println!("strides = {:?}\n", strides);

    /* ------------------------------------------------------------------------ */

    // Generating the values
    // (64, 112, 112)
    let dim_input: (usize, usize, usize) = dim_output;
    let dim_kernel: (usize, usize, usize, usize) = (64,128,3,3);

    assert_eq!(dim_input.0, dim_kernel.0, "number of input channels not consistant");
    assert!(dim_input.1 >= dim_kernel.2, "kernel larger than the input");


    let strides: (usize, usize) = (2, 2);
    let padding: (usize, usize) = (1, 1);
    // let padding: (usize, usize) = (0,0);

    // generate random output size
    let dim_outputx = (dim_input.1 - dim_kernel.2 + padding.0) / strides.0 + 1;
    let dim_outputy = (dim_input.2 - dim_kernel.3 + padding.1) / strides.1 + 1;

    let dim_output = (
        dim_kernel.1,
        dim_outputx,
        dim_outputy,
    );


    let mut kernel_vec = Vec::with_capacity(
        dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3);
    for _ in 0..(dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3) {
        kernel_vec.push(F::rand(&mut rng));
    }

    let conv_input= conv_result_matrix.clone();
    let input: Vec<F> = conv_input.clone().into_iter().flatten().flatten().collect();

    let input_for_prover = field_matrix_encoder_multichannel(input.as_slice(), dim_input);
    let kernel_for_prover = field_kernel_encoder_multichannel(kernel_vec.as_slice(), dim_kernel);

    let conv_result_matrix = full_convolution_computation::<F>(
        input, 
        kernel_vec, 
        dim_input, 
        dim_kernel, 
        dim_output, 
        strides, 
        padding);

    layers.push(LayerInfo::<_>::LIC(LayerInfoConv {
        name: "Conv2_128".to_string(),
        input: input_for_prover,
        kernel: kernel_for_prover,
        output: conv_result_matrix.clone(),
        dim_input: dim_input,
        dim_kernel: dim_kernel,
        dim_output: dim_output,
        padding: padding,
        strides: strides
    }));

    println!("Description of the Layer:");

    println!("input size = {:?}", dim_input);
    println!("input reshape size = {:?}", (dim_kernel.0, 
        dim_output.1 * dim_output.2, 
        dim_kernel.2 * dim_kernel.3));
    println!("kernel size = {:?}", dim_kernel);
    println!("output size = {:?}", dim_output);
    println!("padding = {:?}", padding);
    println!("strides = {:?}\n", strides);

/* ------------------------------------------------------------------------ */

    // Generating the values
    // (128, 56, 56)
    let dim_input: (usize, usize, usize) = dim_output;
    let dim_kernel: (usize, usize, usize, usize) = (128,256,3,3);

    assert_eq!(dim_input.0, dim_kernel.0, "number of input channels not consistant");
    assert!(dim_input.1 >= dim_kernel.2, "kernel larger than the input");


    let strides: (usize, usize) = (1, 1);
    let padding: (usize, usize) = (2, 2);

    // generate random output size
    let dim_outputx = (dim_input.1 - dim_kernel.2 + padding.0) / strides.0 + 1;
    let dim_outputy = (dim_input.2 - dim_kernel.3 + padding.1) / strides.1 + 1;

    let dim_output = (
        dim_kernel.1,
        dim_outputx,
        dim_outputy,
    );


    let mut kernel_vec = Vec::with_capacity(
        dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3);
    for _ in 0..(dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3) {
        kernel_vec.push(F::rand(&mut rng));
    }

    let conv_input= conv_result_matrix.clone();
    let input: Vec<F> = conv_input.clone().into_iter().flatten().flatten().collect();

    let input_for_prover = field_matrix_encoder_multichannel(input.as_slice(), dim_input);
    let kernel_for_prover = field_kernel_encoder_multichannel(kernel_vec.as_slice(), dim_kernel);

    let conv_result_matrix = full_convolution_computation::<F>(
        input, 
        kernel_vec, 
        dim_input, 
        dim_kernel, 
        dim_output, 
        strides, 
        padding);

    layers.push(LayerInfo::<_>::LIC(LayerInfoConv {
        name: "Conv3_256".to_string(),
        input: input_for_prover,
        kernel: kernel_for_prover,
        output: conv_result_matrix.clone(),
        dim_input: dim_input,
        dim_kernel: dim_kernel,
        dim_output: dim_output,
        padding: padding,
        strides: strides
    }));

    println!("Description of the Layer:");

    println!("input size = {:?}", dim_input);
    println!("input reshape size = {:?}", (dim_kernel.0, 
        dim_output.1 * dim_output.2, 
        dim_kernel.2 * dim_kernel.3));
    println!("kernel size = {:?}", dim_kernel);
    println!("output size = {:?}", dim_output);
    println!("padding = {:?}", padding);
    println!("strides = {:?}\n", strides);

    /* ------------------------------------------------------------------------ */

    // Generating the values
    // (256, 56, 56)
    let dim_input: (usize, usize, usize) = dim_output;
    let dim_kernel: (usize, usize, usize, usize) = (256,256,3,3);

    assert_eq!(dim_input.0, dim_kernel.0, "number of input channels not consistant");
    assert!(dim_input.1 >= dim_kernel.2, "kernel larger than the input");


    let strides: (usize, usize) = (2, 2);
    let padding: (usize, usize) = (1, 1);

    // generate random output size
    let dim_outputx = (dim_input.1 - dim_kernel.2 + padding.0) / strides.0 + 1;
    let dim_outputy = (dim_input.2 - dim_kernel.3 + padding.1) / strides.1 + 1;

    let dim_output = (
        dim_kernel.1,
        dim_outputx,
        dim_outputy,
    );

    let mut kernel_vec = Vec::with_capacity(
        dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3);
    for _ in 0..(dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3) {
        kernel_vec.push(F::rand(&mut rng));
    }

    let conv_input= conv_result_matrix.clone();
    let input: Vec<F> = conv_input.clone().into_iter().flatten().flatten().collect();

    let input_for_prover = field_matrix_encoder_multichannel(input.as_slice(), dim_input);
    let kernel_for_prover = field_kernel_encoder_multichannel(kernel_vec.as_slice(), dim_kernel);

    let conv_result_matrix = full_convolution_computation::<F>(
        input, 
        kernel_vec, 
        dim_input, 
        dim_kernel, 
        dim_output, 
        strides, 
        padding);

    layers.push(LayerInfo::<_>::LIC(LayerInfoConv {
        name: "Conv4_256".to_string(),
        input: input_for_prover,
        kernel: kernel_for_prover,
        output: conv_result_matrix.clone(),
        dim_input: dim_input,
        dim_kernel: dim_kernel,
        dim_output: dim_output,
        padding: padding,
        strides: strides
    }));

    println!("Description of the Layer:");

    println!("input size = {:?}", dim_input);
    println!("input reshape size = {:?}", (dim_kernel.0, 
        dim_output.1 * dim_output.2, 
        dim_kernel.2 * dim_kernel.3));
    println!("kernel size = {:?}", dim_kernel);
    println!("output size = {:?}", dim_output);
    println!("padding = {:?}", padding);
    println!("strides = {:?}\n", strides);


    /* ------------------------------------------------------------------------ */

    // Generating the values
    // (256, 28, 28)
    let dim_input: (usize, usize, usize) = dim_output;
    let dim_kernel: (usize, usize, usize, usize) = (256,512,3,3);

    assert_eq!(dim_input.0, dim_kernel.0, "number of input channels not consistant");
    assert!(dim_input.1 >= dim_kernel.2, "kernel larger than the input");


    let strides: (usize, usize) = (1, 1);
    let padding: (usize, usize) = (2, 2);

    // generate random output size
    let dim_outputx = (dim_input.1 - dim_kernel.2 + padding.0) / strides.0 + 1;
    let dim_outputy = (dim_input.2 - dim_kernel.3 + padding.1) / strides.1 + 1;

    let dim_output = (
        dim_kernel.1,
        dim_outputx,
        dim_outputy,
    );


    let mut kernel_vec = Vec::with_capacity(
        dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3);

    for _ in 0..(dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3) {
        kernel_vec.push(F::rand(&mut rng));
    }

    let conv_input= conv_result_matrix.clone();
    let input: Vec<F> = conv_input.clone().into_iter().flatten().flatten().collect();

    let input_for_prover = field_matrix_encoder_multichannel(input.as_slice(), dim_input);
    let kernel_for_prover = field_kernel_encoder_multichannel(kernel_vec.as_slice(), dim_kernel);

    let conv_result_matrix = full_convolution_computation::<F>(
        input, 
        kernel_vec, 
        dim_input, 
        dim_kernel, 
        dim_output, 
        strides, 
        padding);

    layers.push(LayerInfo::<_>::LIC(LayerInfoConv {
        name: "Conv5_512".to_string(),
        input: input_for_prover,
        kernel: kernel_for_prover,
        output: conv_result_matrix.clone(),
        dim_input: dim_input,
        dim_kernel: dim_kernel,
        dim_output: dim_output,
        padding: padding,
        strides: strides
    }));

    println!("Description of the Layer:");

    println!("input size = {:?}", dim_input);
    println!("input reshape size = {:?}", (dim_kernel.0, 
        dim_output.1 * dim_output.2, 
        dim_kernel.2 * dim_kernel.3));
    println!("kernel size = {:?}", dim_kernel);
    println!("output size = {:?}", dim_output);
    println!("padding = {:?}", padding);
    println!("strides = {:?}\n", strides);

    /* ------------------------------------------------------------------------ */

    // Generating the values
    // (512, 28, 28)
    let dim_input: (usize, usize, usize) = dim_output;
    let dim_kernel: (usize, usize, usize, usize) = (512,512,3,3);

    assert_eq!(dim_input.0, dim_kernel.0, "number of input channels not consistant");
    assert!(dim_input.1 >= dim_kernel.2, "kernel larger than the input");


    let strides: (usize, usize) = (2, 2);
    let padding: (usize, usize) = (1, 1);

    // generate random output size
    let dim_outputx = (dim_input.1 - dim_kernel.2 + padding.0) / strides.0 + 1;
    let dim_outputy = (dim_input.2 - dim_kernel.3 + padding.1) / strides.1 + 1;

    let dim_output = (
        dim_kernel.1,
        dim_outputx,
        dim_outputy,
    );


    let mut kernel_vec = Vec::with_capacity(
        dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3);

    for _ in 0..(dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3) {
        kernel_vec.push(F::rand(&mut rng));
    }

    let conv_input= conv_result_matrix.clone();
    let input: Vec<F> = conv_input.clone().into_iter().flatten().flatten().collect();

    let input_for_prover = field_matrix_encoder_multichannel(input.as_slice(), dim_input);
    let kernel_for_prover = field_kernel_encoder_multichannel(kernel_vec.as_slice(), dim_kernel);

    let conv_result_matrix = full_convolution_computation::<F>(
        input, 
        kernel_vec, 
        dim_input, 
        dim_kernel, 
        dim_output, 
        strides, 
        padding);

    layers.push(LayerInfo::<_>::LIC(LayerInfoConv {
        name: "Conv6_512".to_string(),
        input: input_for_prover,
        kernel: kernel_for_prover,
        output: conv_result_matrix.clone(),
        dim_input: dim_input,
        dim_kernel: dim_kernel,
        dim_output: dim_output,
        padding: padding,
        strides: strides
    }));

    println!("Description of the Layer:");

    println!("input size = {:?}", dim_input);
    println!("input reshape size = {:?}", (dim_kernel.0, 
        dim_output.1 * dim_output.2, 
        dim_kernel.2 * dim_kernel.3));
    println!("kernel size = {:?}", dim_kernel);
    println!("output size = {:?}", dim_output);
    println!("padding = {:?}", padding);
    println!("strides = {:?}\n", strides);

    /* ------------------------------------------------------------------------ */

    // Generating the values
    // (512, 14, 14)
    let dim_input: (usize, usize, usize) = dim_output;
    let dim_kernel: (usize, usize, usize, usize) = (512,512,3,3);

    assert_eq!(dim_input.0, dim_kernel.0, "number of input channels not consistant");
    assert!(dim_input.1 >= dim_kernel.2, "kernel larger than the input");


    let strides: (usize, usize) = (1, 1);
    let padding: (usize, usize) = (2, 2);
    // let padding: (usize, usize) = (0,0);

    // generate random output size
    let dim_outputx = (dim_input.1 - dim_kernel.2 + padding.0) / strides.0 + 1;
    let dim_outputy = (dim_input.2 - dim_kernel.3 + padding.1) / strides.1 + 1;

    let dim_output = (
        dim_kernel.1,
        dim_outputx,
        dim_outputy,
    );

    let mut kernel_vec = Vec::with_capacity(
        dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3);

    for _ in 0..(dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3) {
        kernel_vec.push(F::rand(&mut rng));
    }

    let conv_input= conv_result_matrix.clone();
    let input: Vec<F> = conv_input.clone().into_iter().flatten().flatten().collect();

    let input_for_prover = field_matrix_encoder_multichannel(input.as_slice(), dim_input);
    let kernel_for_prover = field_kernel_encoder_multichannel(kernel_vec.as_slice(), dim_kernel);

    let conv_result_matrix = full_convolution_computation::<F>(
        input, 
        kernel_vec, 
        dim_input, 
        dim_kernel, 
        dim_output, 
        strides, 
        padding);

    layers.push(LayerInfo::<_>::LIC(LayerInfoConv {
        name: "Conv7_512".to_string(),
        input: input_for_prover,
        kernel: kernel_for_prover,
        output: conv_result_matrix.clone(),
        dim_input: dim_input,
        dim_kernel: dim_kernel,
        dim_output: dim_output,
        padding: padding,
        strides: strides
    }));

    println!("Description of the Layer:");

    println!("input size = {:?}", dim_input);
    println!("input reshape size = {:?}", (dim_kernel.0, 
        dim_output.1 * dim_output.2, 
        dim_kernel.2 * dim_kernel.3));
    println!("kernel size = {:?}", dim_kernel);
    println!("output size = {:?}", dim_output);
    println!("padding = {:?}", padding);
    println!("strides = {:?}\n", strides);

    /* ------------------------------------------------------------------------ */

    // Generating the values
    // (512, 14, 14)
    let dim_input: (usize, usize, usize) = dim_output;
    let dim_kernel: (usize, usize, usize, usize) = (512,512,3,3);

    assert_eq!(dim_input.0, dim_kernel.0, "number of input channels not consistant");
    assert!(dim_input.1 >= dim_kernel.2, "kernel larger than the input");


    let strides: (usize, usize) = (2, 2);
    let padding: (usize, usize) = (1, 1);

    // generate random output size
    let dim_outputx = (dim_input.1 - dim_kernel.2 + padding.0) / strides.0 + 1;
    let dim_outputy = (dim_input.2 - dim_kernel.3 + padding.1) / strides.1 + 1;

    let dim_output = (
        dim_kernel.1,
        dim_outputx,
        dim_outputy,
    );


    let mut kernel_vec = Vec::with_capacity(
        dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3);

    for _ in 0..(dim_kernel.0 * dim_kernel.1 * dim_kernel.2 * dim_kernel.3) {
        kernel_vec.push(F::rand(&mut rng));
    }

    let conv_input= conv_result_matrix.clone();
    let input: Vec<F> = conv_input.clone().into_iter().flatten().flatten().collect();

    let input_for_prover = field_matrix_encoder_multichannel(input.as_slice(), dim_input);
    let kernel_for_prover = field_kernel_encoder_multichannel(kernel_vec.as_slice(), dim_kernel);

    let conv_result_matrix = full_convolution_computation::<F>(
        input, 
        kernel_vec, 
        dim_input, 
        dim_kernel, 
        dim_output, 
        strides, 
        padding);

    layers.push(
        LayerInfo::<_>::LIC(LayerInfoConv {
        name: "Conv8_512".to_string(),
        input: input_for_prover,
        kernel: kernel_for_prover,
        output: conv_result_matrix.clone(),
        dim_input: dim_input,
        dim_kernel: dim_kernel,
        dim_output: dim_output,
        padding: padding,
        strides: strides
    }));

    println!("Description of the Layer:");

    println!("input size = {:?}", dim_input);
    println!("input reshape size = {:?}", (dim_kernel.0, 
        dim_output.1 * dim_output.2, 
        dim_kernel.2 * dim_kernel.3));
    println!("kernel size = {:?}", dim_kernel);
    println!("output size = {:?}", dim_output);
    println!("padding = {:?}", padding);
    println!("strides = {:?}\n", strides);

    println!("Dense1");

    let flatten_cnn_output = conv_layer_output_to_flatten(&conv_result_matrix, dim_output);


    let mut dense_input = vec![flatten_cnn_output.clone()];

    let mut layers_dense = Vec::new();

    let flatten_dim = dense_input[0].len();

    let dim_wx = flatten_dim;
    let dim_wy = 2048;

    let mut W_dense = Vec::new();

    for _ in 0..dim_wx {
        let mut W_dense_row = Vec::new();
        for _ in 0..dim_wy {
            W_dense_row.push(F::rand(&mut rng));
        }
        W_dense.push(W_dense_row);
    }

    let W_dense_arr = from_matrix_to_arr2(W_dense, (dim_wx, dim_wy));
    let fco = from_matrix_to_arr2(dense_input.clone(), (1, flatten_dim));
    let output_arr = fco.dot(&W_dense_arr);

    let dense_output = from_arr2_to_matrix(output_arr, (1,dim_wy));

    let W_dense = from_arr2_to_matrix(W_dense_arr, (dim_wx, dim_wy));

    layers_dense.push(
        LayerInfoDense { 
            name: "Dense1_2048".to_string(), 
            input: vec![dense_input.clone()], 
            kernel: W_dense, 
            output: dense_output.clone(), 
            dim_input: (1, dim_wx), 
            dim_kernel: (dim_wx, dim_wy), 
            dim_output: (1, dim_wy) 
        }
    );

    println!("Dense2");


    dense_input = dense_output;

    let flatten_dim = dense_input[0].len();


    let dim_wx = flatten_dim;
    let dim_wy = 2048;

    let mut W_dense = Vec::new();

    for _ in 0..dim_wy*dim_wx {
        W_dense.push(F::rand(&mut rng));
    }

    let W_dense = field_matrix_encoder(W_dense.as_slice(), dim_wx, dim_wy);
    let W_dense_arr = from_matrix_to_arr2(W_dense.clone(), (dim_wx, dim_wy));
    let fco = from_matrix_to_arr2(dense_input.clone(), (1, flatten_dim));
    let output_arr = fco.dot(&W_dense_arr);
    let dense_output = from_arr2_to_matrix(output_arr, (1,dim_wy));


    layers_dense.push(
        LayerInfoDense { 
            name: "Dense2_2048".to_string(), 
            input: vec![dense_input.clone()], 
            kernel: W_dense, 
            output: dense_output.clone(), 
            dim_input: (1, dim_wx), 
            dim_kernel: (dim_wx, dim_wy), 
            dim_output: (1, dim_wy) 
        }
    );

    println!("Dense3");


    dense_input = dense_output;

    let flatten_dim = dense_input[0].len();


    let dim_wx = flatten_dim;
    let dim_wy = 1000;

    let mut W_dense = Vec::new();

    for _ in 0..dim_wy*dim_wx {
        W_dense.push(F::rand(&mut rng));
    }

    let W_dense = field_matrix_encoder(W_dense.as_slice(), dim_wx, dim_wy);
    let W_dense_arr = from_matrix_to_arr2(W_dense.clone(), (dim_wx, dim_wy));
    let fco = from_matrix_to_arr2(dense_input.clone(), (1, flatten_dim));
    let output_arr = fco.dot(&W_dense_arr);
    let dense_output = from_arr2_to_matrix(output_arr, (1,dim_wy));


    layers_dense.push(
        LayerInfoDense { 
            name: "Dense3_1000".to_string(), 
            input: vec![dense_input.clone()], 
            kernel: W_dense, 
            output: dense_output.clone(), 
            dim_input: (1, dim_wx), 
            dim_kernel: (dim_wx, dim_wy), 
            dim_output: (1, dim_wy) 
        }
    );

    // Can be used to save the executes
    // /!\ For a regular VGG11 evaluation the file will be around 2.5GB
    let mut compressed_struct = Vec::<u8>::new();
    layers_dense.serialize(&mut compressed_struct).unwrap();
    let mut buffer_struct = BufWriter::new(File::create("VGG11_dense2048_execution_description.txt").unwrap());
    buffer_struct.write_all(&compressed_struct).unwrap();


    Ok(layers_dense)
}


pub fn extract_dense_layers() {
    use ark_bls12_381::Bls12_381;
    use ark_ec::PairingEngine;

    use std::io::Read;
    use ark_serialize::CanonicalDeserialize;

    type E = Bls12_381;
    type Fr = <E as PairingEngine>::Fr;

    let mut save_file = File::open("VGG11_dense4096_execution_description.txt").unwrap();
    let mut file_content = Vec::new();
    save_file
        .read_to_end(&mut file_content)
        .unwrap();


    let mut layers_dense = Vec::<LayerInfoDense<Fr>>::deserialize(file_content.as_slice()).unwrap();

    let l = layers_dense.remove(0);
    drop(layers_dense);

    let mut compressed_struct = Vec::<u8>::new();
    l.serialize(&mut compressed_struct).unwrap();
    let mut buffer_struct = BufWriter::new(File::create("VGG11_dense4096_1_execution_description.txt").unwrap());
    buffer_struct.write_all(&compressed_struct).unwrap();
    
}
