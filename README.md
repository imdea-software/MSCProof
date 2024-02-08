# MSCProof

Implementation of the paper [1], published in the proceedings of ACM CCS 2023. The full version of the paper is available in https://eprint.iacr.org/2023/1342.

[1] David Balbás, Dario Fiore, Maria Isabel González Vasco, Damien Robissout and Claudio Soriente. *Modular Sumcheck Proofs with Applications to Machine Learning and Image Processing*. ACM CCS 2023, Copenhagen, Denmark.

**Disclaimer**: this code is made for academic purposes. It is not safe to be deployed in practice.

## Project Description:

This crate defines several modules that implement efficient sumcheck-based proofs for Machine Learning-related computations, following the theoretical framework introduced in [1]. The implementation, entirely in Rust, includes:

- Sumcheck proofs for matrix multiplication.
- Sumcheck proofs for multi-channel convolution operations, supporting variable strides and padding.
- A general prover for convolutional neural networks, which works by sequentially composing the convolution provers at each layer (i.e. following [1]).
- Support for committing to inputs via multilinear polynomial commitments. We use the PC from [HyperPlonk](https://eprint.iacr.org/2022/1355.pdf).
- A benchmarking module.

Our current implementation does not include sumcheck proofs for activation and pooling layers. Our protocols are non-interactive via the Fiat-Shamir transform.

In all our implementations, we use the 256-bit prime field from the BLS12-381 curve. We also utilize the [arkworks](https://github.com/arkworks-rs) library for elliptic curve, field, and sumcheck operations.

## Module Description:

This crate contains the following modules:

- `conv.rs`: Implements sumcheck proofs for convolution operations. It is based on arkworks mlsumcheck library and manipulates MultilinearExtensions (MLE) of polynomials representing the data. The computation is composed of two parts: (1) a reshape sumcheck and (2) a matrix multiplication-like sumcheck with efficient channel batching. The reshape operation consists of generating a rearrangment of the original input to the convolution, by defining an adequate indexing. This allows to then perform a proof for convolution itself, while avoiding the overhead of padding the convolution kernels.

- `matmul.rs`: Implements sumcheck proofs for matrix multiplication operations. It is based on the arkworks mlsumcheck library and manipulates MultilinearExtensions of polynomials representing the data. 

- `data_structures.rs`: A modified version of the ark_linear_sumcheck file. It implements DenseOrSparseMultilinearExtension that allows easier computations on both Dense and Sparse MLEs.

- `mlsumcheck.rs`: A modified version of the ark_linear_sumcheck file. The main modification consists of allowing the prove function of MultilinearSumCheck to return the internal state of the prover.

- `utils.rs`: Implements utility functions, mainly for manipulating vectors of field elements and for converting them into MLEs using the correct ordering of variables.

- `CNN/`:
	- `prover.rs`: Module based on the conv.rs module. Implements a prover for multiple sequential convolution operations building upon the convolution prover.
	- `verifier.rs`: A verifier for `CNN/verifier.rs`.

- `ipformlsumcheck`: Crate that modifies ark_linear_sumcheck to be able to handle DenseOrSparseMultilinearExtension.

- `generalprover.rs`: Implements a general prover structure. This structure takes the description of a model and creates provers, running sequentially (in reverse order to the computation, following [1]), and specialized for each layer. The current version only supports convolution and dense layer. In particular, no activation or pooling is implemented.

- `generalverifier.rs`: A verifier for generalprover.rs.

- `lib.rs`: Implements several structures and enum that describe the data of the different types of layers.

- `test.rs`: Implements a battery of tests for different model architectures.

- `benches`: Implements benchmarking code to evaluate the performance of different uses of this crate.

## Running our benchmarks:

Our benchmarks can be run simply via `cargo bench`. We provide a benchmarking module for the convolutional and dense layers in `vgg11_bench.rs`.

The running time measured by these benchmarks does not include the generation and preparation of the computation inputs. It does include the time required for fixing all the required variables in the multilinear extensions in preparation for the proof.

Coming soon: Extended benchmark for all VGG11 layers.

## Acknowledgements
This work has received funding by: the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation program under project PICOCRYPT (grant agreement No. 101001283), by projects PRODIGY (TED2021-132464B-I00) and ESPADA (PID2022-142290OB-I00) funded by MCIN/AEI/10.13039/501100011033/ and the European Union NextGenerationEU / PRTR, and by Ministerio de Universidades (FPU21/00600).

