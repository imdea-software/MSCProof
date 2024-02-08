use std::rc::Rc;

use ark_bls12_381::{Bls12_381, Fr};

use ark_ff::BigInteger256;
use ark_linear_sumcheck::rng::{Blake2s512Rng, FeedableRNG};
use ark_poly::{DenseMultilinearExtension, SparseMultilinearExtension};
use ark_std::{UniformRand, test_rng};

use crate::{
    utils::{
        matrix_encoder,
        field_matrix_encoder,
        field_matrix_encoder_multichannel,
        field_kernel_encoder_multichannel,
        field_kernel_encoder_multichannel_matrix_for_conv,
        from_matrix_to_arr2, 
        matrix_reshape, 
        convolution_as_matmul,
        conv_result_to_matrix, 
        input_to_mle, 
        DMLE, 
        kernel_processing, 
        layer_output_to_mle, 
        matrix_reshape_predicate, 
        predicate_processing, 
        display_vec_field_elem, 
        full_convolution_computation,
    }, 
    data_structures::{
        DenseOrSparseMultilinearExtension, 
        ListOfProductsOfPolynomials
    }, 
    matmul::ProverMatMul, 
    LayerProverOutput, 
    conv::ProverConv2D, 
    layer_verifier::LayerVerifier, 
    ipformlsumcheck::prover::ProverMsg, 
    general_prover::GeneralProver, 
    LayerInfoConv, CNN::{prover::ProverCNN, verifier::VerifierCNN}, 
    LayerInfo
};

use ndarray::Array2;
/*
Description of what we should test:

- reverse binary [X]
- matrix/kernel encoder [X]
- Matrix multiplication with field elements [X]
- Reshaping matrix for convolution [X]
- Convolution as matrix multiplication [X]
- MLE generation for output/input/kernel [X]
- Predicate function [X]
- Matmul prover [X]
    - functions of prover []
- Conv prover [X]
    - input processing [X]
    - kernel processing [X]
    - matrices processing [X]
    - predicate processing [X]
    - prove [X]
- Layer verifier [X]
- CNN prover/verifier [X]
    - prover_CNN [X]
    - verifier CNN []
- General prover/verifier []
- 

*/


#[test]
fn test_matrix_encoder() {
    let matA_values = &[0,2,5,1,4,0];
    let matA_dim = (1, 2, 3);
    let matA = matrix_encoder::<Fr>(matA_values, matA_dim);

    assert_eq!(
        matA, 
        vec![
            vec![
                vec![F!(0), F!(2), F!(5)],
                vec![F!(1), F!(4), F!(0)],
            ],
        ]
    );

    let matA_values = vec![F!(0), F!(2), F!(5), F!(1), F!(4), F!(0)];
    let matA_dim = (2, 3);
    let matA = field_matrix_encoder(&matA_values, matA_dim.0, matA_dim.1);

    assert_eq!(
        matA, 
        vec![
            vec![F!(0), F!(2), F!(5)],
            vec![F!(1), F!(4), F!(0)],
        ]
    );

    let matA_values = vec![
        F!(0), F!(2), F!(5), 
        F!(1), F!(4), F!(2), 

        F!(4), F!(3), F!(1), 
        F!(6), F!(0), F!(6)
    ];
    let matA_dim = (2, 2, 3);
    let matA = field_matrix_encoder_multichannel(&matA_values, matA_dim);

    assert_eq!(
        matA, 
        vec![
            vec![
                vec![F!(0), F!(2),F!(5)],
                vec![F!(1),F!(4), F!(2)],
            ],
            vec![
                vec![F!(4), F!(3), F!(1)],
                vec![F!(6), F!(0), F!(6)],
            ]
        ]
    );
}


#[test]
fn test_kernel_encoders() {
    let kernel_values = &[
        F!(1),F!(2),

        F!(3),F!(0),

        F!(6),F!(2),

        
        F!(5),F!(4),

        F!(1),F!(1),

        F!(3),F!(4),
        ];
    let dim_kernel = (2, 3, 1, 2);
    let kernel = field_kernel_encoder_multichannel::<Fr>(kernel_values, dim_kernel);

    assert_eq!(
        kernel, 
        vec![
            vec![
                vec![F!(1),F!(2)],
                vec![F!(3),F!(0)],
                vec![F!(6),F!(2)],


            ],
            vec![
                vec![F!(5),F!(4)],
                vec![F!(1),F!(1)],
                vec![F!(3),F!(4)],
            ]
        ]
    );


    let kernel_values = &[
        F!(1),F!(2),
        F!(0),F!(6),
        
        F!(3),F!(3),
        F!(2),F!(5),
        

        F!(6),F!(1),
        F!(3),F!(0),
        
        F!(5),F!(4),
        F!(2),F!(2),

        
        F!(0),F!(1),
        F!(9),F!(7),

        F!(2),F!(0),
        F!(3),F!(6),
    ];
    let dim_kernel = (3, 2, 2, 2);
    let kernel = field_kernel_encoder_multichannel_matrix_for_conv(kernel_values, dim_kernel);

    assert_eq!(
        kernel, 
        vec![
            vec![
                vec![
                    vec![F!(1),F!(2)],
                    vec![F!(0),F!(6)]
                    ],
                vec![
                    vec![F!(3),F!(3)],
                    vec![F!(2),F!(5)],
                    ],
            ],
            vec![
                vec![
                    vec![F!(6),F!(1)],
                    vec![F!(3),F!(0)],
                    ],
                vec![
                    vec![F!(5),F!(4)],
                    vec![F!(2),F!(2)],
                    ],
            ],
            vec![
                vec![
                    vec![F!(0),F!(1)],
                    vec![F!(9),F!(7)],
                    ],
                vec![
                    vec![F!(2),F!(0)],
                    vec![F!(3),F!(6)],
                    ],
                ]
            ]
    );

}


#[test]
fn test_rbin_macro() {
    let number = 7usize;
    let target_nb_bits = 4;

    assert_eq!(rbin!(number, target_nb_bits), 14);

    let number =128usize;
    let target_nb_bits = 8;

    assert_eq!(rbin!(number, target_nb_bits), 1);
}


#[test]
fn test_matmul_in_field() {
    let matA = vec![
        vec![F!(0), F!(2), F!(5)],
        vec![F!(1), F!(4), F!(0)],
    ];
    let dim_matA = (2, 3);
        
    let matB = vec![
        vec![F!(1), F!(1)],
        vec![F!(2), F!(0)],
        vec![F!(3), F!(1)],
    ];
    let dim_matB = (3, 2);

    let matA_array = from_matrix_to_arr2(matA.to_vec(), (dim_matA.0, dim_matA.1));
    let matB_array = from_matrix_to_arr2(matB.to_vec(), (dim_matB.0, dim_matB.1));

    let matmul_result = matA_array.dot(&matB_array);

    let expected_result = Array2::from_shape_vec(
        (2, 2), 
        vec![F!(19), F!(5), F!(9), F!(1)]
    ).unwrap();

    assert_eq!(
        matmul_result,
        expected_result
    );
}


#[test]
fn test_reshape_matrix() {


    let mat = vec![
        vec![
            vec![
                F!(1), F!(2), F!(3)
            ],
            vec![
                F!(4), F!(5), F!(6)
            ],
            vec![
                F!(7), F!(8), F!(9)
            ]
        ]
    ];

    let mat_dim = (1,3,3);
    let kernel_dim = (2,2);

    let (reshaped_mat, _) = matrix_reshape(
        &mat, 
        mat_dim, 
        kernel_dim, 
        (1,1), 
        (0,0)
    );

    assert_eq!(reshaped_mat,
        vec![
            vec![
                vec![
                    F!(1), F!(2), F!(4), F!(5)
                ],
                vec![
                    F!(2), F!(3), F!(5), F!(6)
                ],
                vec![
                    F!(4), F!(5), F!(7), F!(8)
                ],
                vec![
                    F!(5), F!(6), F!(8), F!(9)
                ]
            ]
        ]
    );


    let mat = vec![
        vec![
            vec![
                F!(1), F!(2), F!(3), F!(4)
            ],
            vec![
                F!(5), F!(6), F!(7), F!(8),
            ],
            vec![
                F!(9), F!(10), F!(11), F!(12)
            ]
        ],
        vec![
            vec![
                F!(2), F!(3), F!(4), F!(5)
            ],
            vec![
                F!(6), F!(7), F!(8), F!(9)
            ],
            vec![
                F!(10), F!(11), F!(12), F!(13)
            ]
        ]
    ];

    let mat_dim = (2,3,4);
    let kernel_dim = (2,3);

    let (reshaped_mat, _) = matrix_reshape(
        &mat, 
        mat_dim, 
        kernel_dim, 
        (2,3), 
        (1,2)
    );


    assert_eq!(reshaped_mat,
        vec![
            vec![
                vec![
                    F!(0), F!(0), F!(0), F!(0), F!(1), F!(2), 
                ],
                vec![
                    F!(0), F!(0), F!(0), F!(3), F!(4), F!(0), 
                ],
                vec![
                    F!(0), F!(5), F!(6), F!(0), F!(9), F!(10)
                ],
                vec![
                    F!(7), F!(8), F!(0), F!(11), F!(12), F!(0)
                ]
            ],
            vec![
                vec![
                    F!(0), F!(0), F!(0), F!(0), F!(2), F!(3), 
                ],
                vec![
                    F!(0), F!(0), F!(0), F!(4), F!(5), F!(0), 
                ],
                vec![
                    F!(0), F!(6), F!(7), F!(0), F!(10), F!(11)
                ],
                vec![
                    F!(8), F!(9), F!(0), F!(12), F!(13), F!(0)
                ]
            ]
        ]
    );

}


#[test]
fn test_conv_as_matrixmul() {
    let input = vec![
        F!(0), F!(2), F!(0), 
        F!(1), F!(3), F!(2), 
        F!(1), F!(1), F!(2), 

        F!(1), F!(0), F!(1), 
        F!(3), F!(1), F!(2),
        F!(1), F!(0), F!(2)
    ];

    let dim_input = (2, 3, 3);
    let input_mat = field_matrix_encoder_multichannel(input.as_slice(), dim_input);

    let kernel = &[
        F!(1),F!(2),
        F!(1),F!(0),

        F!(3),F!(0),
        F!(1),F!(2),

        F!(1),F!(1),
        F!(2),F!(1),


        F!(1),F!(2),
        F!(0),F!(1),

        F!(0),F!(3),
        F!(3),F!(1),
        
        F!(3),F!(1),
        F!(3),F!(2),
        ];
    let dim_kernel = (2, 3, 2, 2);

    // kernel version to use to compute the output
    let kernel_for_conv = field_kernel_encoder_multichannel_matrix_for_conv(
            kernel.as_slice(), 
            dim_kernel
    );

    let strides = (1,1);
    let padding = (0,0);

    // Computing the convolution
    let (reshaped_mat, dim_reshape) =
        matrix_reshape(
            &input_mat, 
            dim_input, 
            (dim_kernel.2, dim_kernel.3), 
            strides, 
            padding);

    let dim_output = (3,2,2);

    let conv_result = convolution_as_matmul(
        reshaped_mat, 
        dim_reshape, 
        kernel_for_conv, 
        dim_output
    );
    let result = conv_result_to_matrix(conv_result.clone(), dim_output).clone();

    assert_eq!(
        result,
        vec![
            vec![
                vec![F!(7), F!(9)],
                vec![F!(13), F!(15)]
            ],
            vec![
                vec![F!(17), F!(21)],
                vec![F!(12), F!(22)]
            ],
            vec![
                vec![F!(21), F!(18)],
                vec![F!(20), F!(18)]
            ]
        ]
    )
}


#[test]
fn test_MLE_generation() {

    // INPUT
    let input = vec![
        vec![
            vec![F!(1), F!(2)],
            vec![F!(3), F!(4)],
            vec![F!(5), F!(6)],
        ]
    ];
    let dim_input = (1, 3, 2);

    let MLE_input = input_to_mle(&input, dim_input);

    assert_eq!(
        MLE_input,
        DenseMultilinearExtension::from_evaluations_vec(3, vec![
            F!(1),F!(5),F!(3),F!(0),F!(2),F!(6),F!(4),F!(0)])
    );


    // KERNEL
    let kernel = vec![
        vec![
            vec![F!(1),F!(2),F!(3),F!(4)],
            vec![F!(5),F!(6),F!(7),F!(8)],
        ],
        vec![
            vec![F!(9),F!(10),F!(11),F!(12)],
            vec![F!(13),F!(14),F!(15),F!(16)],
        ],
        vec![
            vec![F!(17),F!(18),F!(19),F!(20)],
            vec![F!(21),F!(22),F!(23),F!(24)],
            ]
    ];
    let dim_kernel = (3, 2, 2, 2);

    let MLE_kernel = kernel_processing(kernel, dim_kernel);

    // println!("{:?}", DMLE(&DenseMultilinearExtension::from(MLE_kernel.clone())));

    assert_eq!(
        DenseMultilinearExtension::from(MLE_kernel),
        DenseMultilinearExtension::from_evaluations_vec(5, vec![
            F!(1),F!(5),F!(3),F!(7),
            F!(2),F!(6),F!(4),F!(8),

            F!(17),F!(21),F!(19),F!(23),
            F!(18),F!(22),F!(20),F!(24),

            F!(9),F!(13),F!(11),F!(15),
            F!(10),F!(14),F!(12),F!(16),

            F!(0),F!(0),F!(0),F!(0),
            F!(0),F!(0),F!(0),F!(0),
            ])
    );

    // OUTPUT
    let output = vec![
        vec![
            vec![F!(1), F!(2), F!(3)], 
            vec![F!(4), F!(5), F!(6)], 
            vec![F!(7), F!(8), F!(9)], 
        ],
        vec![
            vec![F!(10), F!(11), F!(12)], 
            vec![F!(13), F!(14), F!(15)],
            vec![F!(16), F!(17), F!(18)],
        ],
        vec![
            vec![F!(19), F!(20), F!(21)], 
            vec![F!(22), F!(23), F!(24)],
            vec![F!(25), F!(26), F!(27)]
        ],
    ];

    let dim_output = (3, 3, 3);
    let MLE_output = layer_output_to_mle(&output, dim_output, false);
    println!("{:?}", DMLE(&DenseMultilinearExtension::from(MLE_output.clone())));

    assert_eq!(
        DenseMultilinearExtension::from(MLE_output),
        DenseMultilinearExtension::from_evaluations_vec(6, vec![
            F!(1),F!(9),F!(5),F!(0),
            F!(3),F!(0),F!(7),F!(0),
            F!(2),F!(0),F!(6),F!(0),
            F!(4),F!(0),F!(8),F!(0),

            F!(19),F!(27),F!(23),F!(0),
            F!(21),F!(0),F!(25),F!(0),
            F!(20),F!(0),F!(24),F!(0),
            F!(22),F!(0),F!(26),F!(0),

            F!(10),F!(18),F!(14),F!(0),
            F!(12),F!(0),F!(16),F!(0),
            F!(11),F!(0),F!(15),F!(0),
            F!(13),F!(0),F!(17),F!(0),

            F!(0),F!(0),F!(0),F!(0),
            F!(0),F!(0),F!(0),F!(0),
            F!(0),F!(0),F!(0),F!(0),
            F!(0),F!(0),F!(0),F!(0),
            ])
    );
}


#[test]
fn test_reshape_predicate_generation() {


    let dim_input = (2, 3, 3);
    let dim_kernel = (2, 3, 2, 2);

    let strides = (1,1);
    let padding = (0,0);

    // predicate processing
    let (predicate, dim_input_reshape) = matrix_reshape_predicate(
        dim_input,
        dim_kernel,
        padding,
        strides,
    );


    assert_eq!(predicate, vec![
        (0,0), (1,1), (2,3), (3,4), 
        (4,1), (5,2), (6,4), (7,5), 
        (8,3), (9,4), (10,6), (11,7), 
        (12,4), (13,5), (14,7), (15,8),
    ]);

    let predicate_mle = predicate_processing::<Fr>(
        predicate, 
        dim_input, 
        dim_input_reshape
    );

    assert_eq!(
        <DenseOrSparseMultilinearExtension<_> as TryInto<SparseMultilinearExtension<_>>>::try_into(predicate_mle).unwrap(), 
        SparseMultilinearExtension::from_evaluations(8, &vec![
            (0,F!(1)), (136,F!(1)), (196,F!(1)), (44,F!(1)), 
            (130,F!(1)), (74,F!(1)), (38,F!(1)), (174,F!(1)), 
            (193,F!(1)), (41,F!(1)), (101,F!(1)), (237,F!(1)), 
            (35,F!(1)), (171,F!(1)), (231,F!(1)), (31,F!(1)),
        ])
    );


}


#[test]
fn test_matmul_prover() {

    let mut rng: rand::prelude::StdRng = test_rng();

    let matA = vec![
        vec![F!(0), F!(2), F!(5)],
        vec![F!(1), F!(4), F!(0)],
    ];
    let dim_matA = (2, 3);
        
    let matB = vec![
        vec![F!(1), F!(1)],
        vec![F!(2), F!(0)],
        vec![F!(3), F!(1)],
    ];
    let dim_matB = (3, 2);

    let dim_matC = (2, 2);
    let mut initial_randomness = Vec::new();
    for _ in 0..(log2i!(dim_matC.0) + log2i!(dim_matC.1)) {
        initial_randomness.push(Fr::rand(&mut rng));
    }

    let mut matmul_prover = ProverMatMul::new(
        matA,
        matB,
        dim_matA,
        dim_matB,
        initial_randomness,
    );

    let mut fs_rng = Blake2s512Rng::setup();

    let proof = matmul_prover.prove(&mut fs_rng).unwrap();

    let claimed_value0 = BigInteger256([11573792320801518808, 8574574629511446019, 8266107609857501412, 423563442966955163]);
    let claimed_value1 = BigInteger256([14353799177417571384, 6372118770988075464, 13325372458890885239, 4440589244794704515]);
    let claimed_value = (Fr::from(claimed_value0), Fr::from(claimed_value1));

    assert_eq!(proof.claimed_values, claimed_value);

    let message0 = vec![
        Fr::from(BigInteger256([3988430839458495630, 7968002951805763855, 13725789482913870971, 4685530991393339072])),
        Fr::from(BigInteger256([11715651818195423519, 13384750612906784990, 13210129111641160599, 7171198446706218080])),
        Fr::from(BigInteger256([6159610743117329470, 12014040609811510703, 5682059069656408979, 1266129674669521425])),
    ];
    let message1 = vec![
        Fr::from(BigInteger256([8463701993950410751, 15167548089702808688, 14971559120052097171, 5019165809070136714])),
        Fr::from(BigInteger256([12024380035381120935, 16546595105469020973, 6100673688823537158, 3563505224403440312])),
        Fr::from(BigInteger256([14593948338297125221, 10546755944841611817, 14744308807205948158, 3334513889134307660])),
    ];

    assert_eq!(proof.prover_msgs[0].evaluations, message0);
    assert_eq!(proof.prover_msgs[1].evaluations, message1);

    let evaluations0 = vec![
        Fr::from(BigInteger256([12346421629811869064, 10832332258257352915, 17999185152888039383, 7443919619818212425])),
        Fr::from(BigInteger256([12054868124304024674, 12902623898822750168, 2247145023596867812, 4547986198231184631])),
        Fr::from(BigInteger256([6246099190209153809, 15630505107976623528, 13860407333426755497, 6534322380171975499])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
    ];
    let evaluations1 = vec![
        Fr::from(BigInteger256([1, 0, 0, 0])),
        Fr::from(BigInteger256([16323324743792361225, 3590468515581694299, 9805471883936094064, 2914130811527064747])),
        Fr::from(BigInteger256([16323324743792361224, 3590468515581694299, 9805471883936094064, 2914130811527064747])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
    ];

    assert_eq!(proof.polynomial.flattened_ml_extensions[0].evaluations(), evaluations0);
    assert_eq!(proof.polynomial.flattened_ml_extensions[1].evaluations(), evaluations1);
    
}


#[test]
fn test_conv_prover() {

    type E = Bls12_381;
    let mut rng: rand::prelude::StdRng = test_rng();

    let input = vec![
        F!(0), F!(2), F!(0), 
        F!(1), F!(3), F!(2), 
        F!(1), F!(1), F!(2), 

        F!(1), F!(0), F!(1), 
        F!(3), F!(1), F!(2),
        F!(1), F!(0), F!(2)
    ];

    let dim_input = (2, 3, 3);
    let input_mat = field_matrix_encoder_multichannel(input.as_slice(), dim_input);

    let kernel = &[
        F!(1),F!(2),
        F!(1),F!(0),

        F!(3),F!(0),
        F!(1),F!(2),

        F!(1),F!(1),
        F!(2),F!(1),


        F!(1),F!(2),
        F!(0),F!(1),

        F!(0),F!(3),
        F!(3),F!(1),
        
        F!(3),F!(1),
        F!(3),F!(2),
        ];
    let dim_kernel = (2, 3, 2, 2);
    // kernel version to use to compute the output
    let kernel_for_conv = field_kernel_encoder_multichannel(
        kernel.as_slice(), 
        dim_kernel
    );
    
    let strides = (1,1);
    let padding = (0,0);

    let dim_output = (
        dim_kernel.1,
        get_output_size!(dim_input.1, dim_kernel.2, padding.0, strides.0),
        get_output_size!(dim_input.2, dim_kernel.3, padding.1, strides.1),
    );
    let mut initial_randomness = Vec::new();
    for _ in 0..(log2i!(dim_output.0) + log2i!(dim_output.1 * dim_output.2)) {
        initial_randomness.push(Fr::rand(&mut rng));
    }

    display_vec_field_elem(&initial_randomness);

    let mut prover = ProverConv2D::<E, Fr>::new(
        input_mat,
        kernel_for_conv,
        strides,
        padding,
        dim_input,
        dim_kernel,
        Vec::new(),
        initial_randomness
    );

    // BASE INPUT MLE FOR PREDICATE SC
    let (_, _, _) = prover.input_processing();

    let MLE_base_input = prover.mles.get("base_input_mle").unwrap();

    let eval_vec_base_input = vec![
        Fr::from(BigInteger256([0, 0, 0, 0])), Fr::from(BigInteger256([1, 0, 0, 0])),
        Fr::from(BigInteger256([2, 0, 0, 0])), Fr::from(BigInteger256([2, 0, 0, 0])),
        Fr::from(BigInteger256([3, 0, 0, 0])), Fr::from(BigInteger256([1, 0, 0, 0])),
        Fr::from(BigInteger256([0, 0, 0, 0])), Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([0, 0, 0, 0])), Fr::from(BigInteger256([1, 0, 0, 0])),
        Fr::from(BigInteger256([0, 0, 0, 0])), Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([1, 0, 0, 0])), Fr::from(BigInteger256([1, 0, 0, 0])),
        Fr::from(BigInteger256([0, 0, 0, 0])), Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([2, 0, 0, 0])), Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([0, 0, 0, 0])), Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([2, 0, 0, 0])), Fr::from(BigInteger256([2, 0, 0, 0])),
        Fr::from(BigInteger256([0, 0, 0, 0])), Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([1, 0, 0, 0])), Fr::from(BigInteger256([3, 0, 0, 0])),
        Fr::from(BigInteger256([0, 0, 0, 0])), Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([1, 0, 0, 0])), Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([0, 0, 0, 0])), Fr::from(BigInteger256([0, 0, 0, 0])),
    ];

    let mle_bi = DenseOrSparseMultilinearExtension::from(
        DenseMultilinearExtension::from_evaluations_vec(5, eval_vec_base_input)
    );

    assert_eq!(MLE_base_input, &mle_bi);

    // MATRICES PROCESSING
    let poly = prover.matrices_processing();

    // for e in poly.flattened_ml_extensions.iter() {
    //     for v in e.evaluations().iter() {
    //         println!("v: {:?}", v.into_repr());
    //     }
    //     println!();
    // }

    let evaluations0 = vec![
        Fr::from(BigInteger256([14469840959729059458, 7241863742675658615, 8193713268951945319, 4529788808291147678])),
        Fr::from(BigInteger256([12268488139568157289, 6271456098419909563, 17891249045267394514, 2965255401490357032])),
        Fr::from(BigInteger256([14373839931923971033, 9491838466838245283, 4732924535367613350, 7418357324836566283])),
        Fr::from(BigInteger256([993866400300901131, 6817867426598335980, 16844905660891225302, 5750390531503033797])),
        Fr::from(BigInteger256([1033551897516233545, 16322542264230190307, 10473540767647223549, 2577564032979769411])),
        Fr::from(BigInteger256([14228063181317532484, 7509904582851902758, 13458039095111693354, 1793632184310827709])),
        Fr::from(BigInteger256([13408131294770464261, 9037098000466505765, 3873077174065216543, 5986854214054680052])),
        Fr::from(BigInteger256([11312869732295635521, 12956534067736714224, 7525644385240815833, 4866355586838443014])),
    ];
    let evaluations1 = vec![
        Fr::from(BigInteger256([9890643471239944486, 15455253112483188315, 783863192337820097, 6986117074087810066])),
        Fr::from(BigInteger256([4431792157819506087, 16922992651569175518, 18138566871476890010, 781673380728451842])),
        Fr::from(BigInteger256([13045268801453209672, 2816716209640087216, 6802344446386711405, 6270149232327880186])),
        Fr::from(BigInteger256([638583411655636803, 8574472444620403652, 7905652258033885178, 4925579040976415537])),
        Fr::from(BigInteger256([14332357578802766553, 126065029170868464, 2694815015667834239, 1536493800685261736])),
        Fr::from(BigInteger256([11177732248589501367, 12764601932013969563, 15123077835328494547, 2252461642445191615])),
        Fr::from(BigInteger256([3783286796420553304, 3707212794411164003, 8013785587880329086, 2087390993883036478])),
        Fr::from(BigInteger256([4431792157819506087, 16922992651569175518, 18138566871476890010, 781673380728451842])),
    ];

    assert_eq!(poly.flattened_ml_extensions[0].evaluations(), evaluations0);
    assert_eq!(poly.flattened_ml_extensions[1].evaluations(), evaluations1);

    let mut fs_rng = Blake2s512Rng::setup();

    let (proof_conv, proof_reshape) = prover.prove(&mut fs_rng).unwrap();

    // println!("{:?}", proof_conv.claimed_values.0.into_repr());
    // println!("{:?}", proof_conv.claimed_values.1.into_repr());

    let claimed_value_conv0 = BigInteger256([4222704482314361549, 13097999571724729586, 16992038626017478546, 5846558625670992528]);
    let claimed_value_conv1 = BigInteger256([4496201493315759591, 14890889785943852132, 17405260869610600127, 1981489524293078325]);
    let claimed_value_conv = (Fr::from(claimed_value_conv0), Fr::from(claimed_value_conv1));

    assert_eq!(proof_conv.claimed_values, claimed_value_conv);

    // for e in proof_conv.prover_msgs[0].evaluations.iter() {
    //     println!("{:?}", e.into_repr());
    // }

    // for e in proof_conv.prover_msgs[1].evaluations.iter() {
    //     println!("{:?}", e.into_repr());
    // }

    // for e in proof_conv.prover_msgs[2].evaluations.iter() {
    //     println!("{:?}", e.into_repr());
    // }
    


    let message_conv0 = vec![
        Fr::from(BigInteger256([9307159495985589929, 13310593442351049215, 12611582900133162415, 5735560879319044168])),
        Fr::from(BigInteger256([10621704069662826415, 2155862642730168244, 14773984245853373064, 2224019686403990685])),
        Fr::from(BigInteger256([789733431844260389, 18092311615752391416, 2759369993740374160, 6314332777158688233])),
    ];
    let message_conv1 = vec![
        Fr::from(BigInteger256([14444036183588308746, 17966371647798439628, 4097412604319220006, 7642092353783754387])),
        Fr::from(BigInteger256([10631865310493652880, 3816522776275426092, 4902606355625231322, 6998739771071700032])),
        Fr::from(BigInteger256([10042450528458616398, 13732877461643151851, 1517936242629224869, 7759311639636051325])),
    ];
    let message_conv2 = vec![
        Fr::from(BigInteger256([9269021908200268131, 3468817394188150549, 5339725192807209708, 8063601020870512906])),
        Fr::from(BigInteger256([17777828392327568627, 16627717095858580698, 4681589546529737349, 3246193870656461876])),
        Fr::from(BigInteger256([11380190865577589152, 13564113464235123429, 8513501830494655808, 7326066161964088070])),
    ];

    assert_eq!(proof_conv.prover_msgs[0].evaluations, message_conv0);
    assert_eq!(proof_conv.prover_msgs[1].evaluations, message_conv1);
    assert_eq!(proof_conv.prover_msgs[2].evaluations, message_conv2);

    // for e in proof_conv.polynomial.flattened_ml_extensions[0].evaluations().iter() {
    //     println!("{:?}", e.into_repr());
    // }    
    // println!();

    // for e in proof_conv.polynomial.flattened_ml_extensions[1].evaluations().iter() {
    //     println!("{:?}", e.into_repr());
    // }

    let evaluations_conv0 = vec![
        Fr::from(BigInteger256([14469840959729059458, 7241863742675658615, 8193713268951945319, 4529788808291147678])),
        Fr::from(BigInteger256([12268488139568157289, 6271456098419909563, 17891249045267394514, 2965255401490357032])),
        Fr::from(BigInteger256([14373839931923971033, 9491838466838245283, 4732924535367613350, 7418357324836566283])),
        Fr::from(BigInteger256([993866400300901131, 6817867426598335980, 16844905660891225302, 5750390531503033797])),
        Fr::from(BigInteger256([1033551897516233545, 16322542264230190307, 10473540767647223549, 2577564032979769411])),
        Fr::from(BigInteger256([14228063181317532484, 7509904582851902758, 13458039095111693354, 1793632184310827709])),
        Fr::from(BigInteger256([13408131294770464261, 9037098000466505765, 3873077174065216543, 5986854214054680052])),
        Fr::from(BigInteger256([11312869732295635521, 12956534067736714224, 7525644385240815833, 4866355586838443014])),
    ];
    let evaluations_conv1 = vec![
        Fr::from(BigInteger256([9890643471239944486, 15455253112483188315, 783863192337820097, 6986117074087810066])),
        Fr::from(BigInteger256([4431792157819506087, 16922992651569175518, 18138566871476890010, 781673380728451842])),
        Fr::from(BigInteger256([13045268801453209672, 2816716209640087216, 6802344446386711405, 6270149232327880186])),
        Fr::from(BigInteger256([638583411655636803, 8574472444620403652, 7905652258033885178, 4925579040976415537])),
        Fr::from(BigInteger256([14332357578802766553, 126065029170868464, 2694815015667834239, 1536493800685261736])),
        Fr::from(BigInteger256([11177732248589501367, 12764601932013969563, 15123077835328494547, 2252461642445191615])),
        Fr::from(BigInteger256([3783286796420553304, 3707212794411164003, 8013785587880329086, 2087390993883036478])),
        Fr::from(BigInteger256([4431792157819506087, 16922992651569175518, 18138566871476890010, 781673380728451842])),
    ];

    assert_eq!(proof_conv.polynomial.flattened_ml_extensions[0].evaluations(), evaluations_conv0);
    assert_eq!(proof_conv.polynomial.flattened_ml_extensions[1].evaluations(), evaluations_conv1);

    // println!("{:?}", proof_reshape.claimed_values.0.into_repr());
    // println!("{:?}", proof_reshape.claimed_values.1.into_repr());

    let claimed_value_reshape0 = BigInteger256([13640885558736527476, 527015822347850356, 1578806250728299763, 52923505022772399]);
    let claimed_value_reshape1 = BigInteger256([4593376755172602178, 12844788526791639584, 14681116261624787548, 7797257325484639128]);
    let claimed_value_reshape = (Fr::from(claimed_value_reshape0), Fr::from(claimed_value_reshape1));

    assert_eq!(proof_reshape.claimed_values, claimed_value_reshape);

    // for e in proof_reshape.prover_msgs[0].evaluations.iter() {
    //     println!("{:?}", e.into_repr());
    // }
    // println!();

    // for e in proof_reshape.prover_msgs[1].evaluations.iter() {
    //     println!("{:?}", e.into_repr());
    // }

    // for e in proof_reshape.prover_msgs[2].evaluations.iter() {
    //     println!("{:?}", e.into_repr());
    // }
    


    let message_reshape0 = vec![
        Fr::from(BigInteger256([15510251042167080539, 17259706178357994425, 17807128333907376069, 1852860568972225937])),
        Fr::from(BigInteger256([7159197513856832626, 14285037467076286776, 17631654365819654092, 3993698056698766590])),
        Fr::from(BigInteger256([981055235955364552, 12900319323372718651, 5513964945885031608, 6492278143514451643])),
    ];
    let message_reshape1 = vec![
        Fr::from(BigInteger256([8988415057333434029, 413256151761580299, 12836120477000495598, 2247524514544459364])),
        Fr::from(BigInteger256([14290863506303894387, 8330653883825129961, 8610424764542290451, 6009018091186111137])),
        Fr::from(BigInteger256([13189558537710759661, 16224761623710773353, 4245457238763653965, 4765387255375512305])),
    ];
    let message_reshape2 = vec![
        Fr::from(BigInteger256([13757706626873918034, 16606881646712172498, 11089884107393469784, 5954676274591900423])),
        Fr::from(BigInteger256([7988838005359038417, 8621037793081466141, 4307921523736267207, 3751496460886794913])),
        Fr::from(BigInteger256([15794848882832606366, 3390211588870696394, 5139194638935092430, 5933457365928714417])),
    ];

    assert_eq!(proof_reshape.prover_msgs[0].evaluations, message_reshape0);
    assert_eq!(proof_reshape.prover_msgs[1].evaluations, message_reshape1);
    assert_eq!(proof_reshape.prover_msgs[2].evaluations, message_reshape2);

    // for e in proof_reshape.polynomial.flattened_ml_extensions[0].evaluations().iter() {
    //     println!("{:?}", e.into_repr());
    // }    
    // println!();

    // for e in proof_reshape.polynomial.flattened_ml_extensions[1].evaluations().iter() {
    //     println!("{:?}", e.into_repr());
    // }

    let evaluations_reshape0 = vec![
        Fr::from(BigInteger256([3533910651351147025, 1054023330368126159, 11177754015695111571, 3466790071249388076])),
        Fr::from(BigInteger256([2, 0, 0, 0])),
        Fr::from(BigInteger256([11378922766712290274, 3926112747801829984, 18229199014668651743, 1419936716965673198])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([3533910651351147025, 1054023330368126159, 11177754015695111571, 3466790071249388076])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([1, 0, 0, 0])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([11378922766712290273, 3926112747801829984, 18229199014668651743, 1419936716965673198])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([2, 0, 0, 0])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([7067821302702294051, 2108046660736252318, 3908763957680671526, 6933580142498776153])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([14912833418063437297, 4980136078169956143, 10960208956654211698, 4886726788215061275])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
    ];
    let evaluations_reshape1 = vec![
        Fr::from(BigInteger256([13667729356518893408, 3581049743161457693, 14408768161668087829, 3718485686056845514])),
        Fr::from(BigInteger256([3579598756928416313, 7142518733538143388, 8815827182909827046, 1996849028349383295])),
        Fr::from(BigInteger256([185181596168927668, 4144540518342487568, 9763535503106311965, 4975012262876865527])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([6722672438720491267, 11787724167423488220, 7472541566764540473, 7294359734941121072])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([6000183756026674844, 3024523722994883918, 17763214028807963269, 4032214975516324204])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([5327132866826661708, 3240272597342969980, 13847061172759409195, 2948380405829119596])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([18165504139473462399, 16194322395058322478, 14257306583706019731, 7179829807070754798])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([4448841619473210907, 17949868898770878462, 16050631595623808485, 3251649039284983165])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([15690131743226631476, 18446208487186434652, 8310929066400648349, 6370803357396849584])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
    ];

    assert_eq!(proof_reshape.polynomial.flattened_ml_extensions[0].evaluations(), evaluations_reshape0);
    assert_eq!(proof_reshape.polynomial.flattened_ml_extensions[1].evaluations(), evaluations_reshape1);
    
}


#[test]
fn test_layer_verifier() {

    let mut rng: rand::prelude::StdRng = test_rng();

    let input = vec![
        F!(0), F!(2), F!(0), 
        F!(1), F!(3), F!(2), 
        F!(1), F!(1), F!(2), 

        F!(1), F!(0), F!(1), 
        F!(3), F!(1), F!(2),
        F!(1), F!(0), F!(2)
    ];

    let dim_input = (2, 3, 3);
    let input_mat = field_matrix_encoder_multichannel(input.as_slice(), dim_input);

    let kernel = &[
        F!(1),F!(2),
        F!(1),F!(0),

        F!(3),F!(0),
        F!(1),F!(2),

        F!(1),F!(1),
        F!(2),F!(1),


        F!(1),F!(2),
        F!(0),F!(1),

        F!(0),F!(3),
        F!(3),F!(1),
        
        F!(3),F!(1),
        F!(3),F!(2),
        ];
    let dim_kernel = (2, 3, 2, 2);

    // kernel version to use to compute the output
    let kernel_for_conv = field_kernel_encoder_multichannel_matrix_for_conv(
            kernel.as_slice(), 
            dim_kernel
    );

    let strides = (1,1);
    let padding = (0,0);

    // Computing the convolution
    let (reshaped_mat, dim_reshape) =
        matrix_reshape(
            &input_mat, 
            dim_input, 
            (dim_kernel.2, dim_kernel.3), 
            strides, 
            padding);

    let dim_output = (3,2,2);

    let conv_output = convolution_as_matmul(
        reshaped_mat, 
        dim_reshape, 
        kernel_for_conv, 
        dim_output
    );
    let output = conv_result_to_matrix(conv_output.clone(), dim_output).clone();
    let MLE_output = layer_output_to_mle(&output, dim_output, false);

    let mut initial_randomness = Vec::new();
    for _ in 0..(log2i!(dim_output.0) + log2i!(dim_output.1 * dim_output.2)) {
        initial_randomness.push(Fr::rand(&mut rng));
    }

    let MLE_output_eval = MLE_output.fix_variables(&initial_randomness.as_slice())[0];

    let claimed_value_conv0 = BigInteger256([4222704482314361549, 13097999571724729586, 16992038626017478546, 5846558625670992528]);
    let wrong_claimed_value_conv0 = BigInteger256([422270448314361549, 13097999571724729586, 16992038626017478546, 5846558625670992528]);
    let claimed_value_conv1 = BigInteger256([4496201493315759591, 14890889785943852132, 17405260869610600127, 1981489524293078325]);
    let claimed_value_conv = (Fr::from(claimed_value_conv0), Fr::from(claimed_value_conv1));
    let wrong_claimed_value_conv = (Fr::from(wrong_claimed_value_conv0), Fr::from(claimed_value_conv1));

    let message_conv0 = vec![
        Fr::from(BigInteger256([9307159495985589929, 13310593442351049215, 12611582900133162415, 5735560879319044168])),
        Fr::from(BigInteger256([10621704069662826415, 2155862642730168244, 14773984245853373064, 2224019686403990685])),
        Fr::from(BigInteger256([789733431844260389, 18092311615752391416, 2759369993740374160, 6314332777158688233])),
    ];
    let message_conv1 = vec![
        Fr::from(BigInteger256([14444036183588308746, 17966371647798439628, 4097412604319220006, 7642092353783754387])),
        Fr::from(BigInteger256([10631865310493652880, 3816522776275426092, 4902606355625231322, 6998739771071700032])),
        Fr::from(BigInteger256([10042450528458616398, 13732877461643151851, 1517936242629224869, 7759311639636051325])),
    ];
    let message_conv2 = vec![
        Fr::from(BigInteger256([9269021908200268131, 3468817394188150549, 5339725192807209708, 8063601020870512906])),
        Fr::from(BigInteger256([17777828392327568627, 16627717095858580698, 4681589546529737349, 3246193870656461876])),
        Fr::from(BigInteger256([11380190865577589152, 13564113464235123429, 8513501830494655808, 7326066161964088070])),
    ];

    let prover_msg0 = ProverMsg {
        evaluations: message_conv0.clone(),
    };
    let prover_msg1 = ProverMsg {
        evaluations: message_conv1.clone(),
    };
    let prover_msg2 = ProverMsg {
        evaluations: message_conv2.clone(),
    };
    let prover_msgs = vec![prover_msg0, prover_msg1, prover_msg2];

    let evaluations_conv0 = vec![
        Fr::from(BigInteger256([14469840959729059458, 7241863742675658615, 8193713268951945319, 4529788808291147678])),
        Fr::from(BigInteger256([12268488139568157289, 6271456098419909563, 17891249045267394514, 2965255401490357032])),
        Fr::from(BigInteger256([14373839931923971033, 9491838466838245283, 4732924535367613350, 7418357324836566283])),
        Fr::from(BigInteger256([993866400300901131, 6817867426598335980, 16844905660891225302, 5750390531503033797])),
        Fr::from(BigInteger256([1033551897516233545, 16322542264230190307, 10473540767647223549, 2577564032979769411])),
        Fr::from(BigInteger256([14228063181317532484, 7509904582851902758, 13458039095111693354, 1793632184310827709])),
        Fr::from(BigInteger256([13408131294770464261, 9037098000466505765, 3873077174065216543, 5986854214054680052])),
        Fr::from(BigInteger256([11312869732295635521, 12956534067736714224, 7525644385240815833, 4866355586838443014])),
    ];
    let evaluations_conv1 = vec![
        Fr::from(BigInteger256([9890643471239944486, 15455253112483188315, 783863192337820097, 6986117074087810066])),
        Fr::from(BigInteger256([4431792157819506087, 16922992651569175518, 18138566871476890010, 781673380728451842])),
        Fr::from(BigInteger256([13045268801453209672, 2816716209640087216, 6802344446386711405, 6270149232327880186])),
        Fr::from(BigInteger256([638583411655636803, 8574472444620403652, 7905652258033885178, 4925579040976415537])),
        Fr::from(BigInteger256([14332357578802766553, 126065029170868464, 2694815015667834239, 1536493800685261736])),
        Fr::from(BigInteger256([11177732248589501367, 12764601932013969563, 15123077835328494547, 2252461642445191615])),
        Fr::from(BigInteger256([3783286796420553304, 3707212794411164003, 8013785587880329086, 2087390993883036478])),
        Fr::from(BigInteger256([4431792157819506087, 16922992651569175518, 18138566871476890010, 781673380728451842])),
    ];

    let input_mle = DenseOrSparseMultilinearExtension::from(
        DenseMultilinearExtension::from_evaluations_vec(3, evaluations_conv0)
    );
    let kernel_mle = DenseOrSparseMultilinearExtension::from(
        DenseMultilinearExtension::from_evaluations_vec(3, evaluations_conv1)
    );

    let mut poly = ListOfProductsOfPolynomials::new(input_mle.num_vars());
    let mut prod = Vec::new();

    prod.push(Rc::new(input_mle));
    prod.push(Rc::new(kernel_mle));

    // coefficient
    poly.add_product(prod, Fr::from(1 as u32));


    let correct_prover_output = LayerProverOutput { 
        claimed_values: claimed_value_conv, 
        prover_msgs: prover_msgs.clone(), 
        polynomial: poly.clone()
    };


    let mut layer_verifier = LayerVerifier::new(
        correct_prover_output,
        MLE_output_eval
    );
    
    let mut fs_rng = Blake2s512Rng::setup();
    
    let _ = layer_verifier.verify_SC(&mut fs_rng).unwrap();
    
    println!("First verif done");






    let wrong_prover_output = LayerProverOutput { 
        claimed_values: wrong_claimed_value_conv, 
        prover_msgs: prover_msgs, 
        polynomial: poly 
    };

    let mut layer_verifier = LayerVerifier::new(
        wrong_prover_output,
        MLE_output_eval
    );
    
    let mut fs_rng = Blake2s512Rng::setup();
    let result = layer_verifier.verify_SC(&mut fs_rng);

    result.expect_err("Wrong proof failed to verify");
    
    // assert_eq!(Err(Error), layer_verifier.verify_SC(&mut fs_rng));
    // assert!(false);

}


#[test]
fn test_CNN_prover() {

    let mut rng: rand::prelude::StdRng = test_rng();
    type E = Bls12_381;

    let input1 = vec![
        F!(0), F!(2), F!(0), F!(2),
        F!(1), F!(3), F!(2), F!(2),
        F!(1), F!(1), F!(2), F!(2),
        F!(1), F!(0), F!(1), F!(2),

        F!(1), F!(0), F!(1), F!(2),
        F!(3), F!(1), F!(2), F!(2),
        F!(1), F!(0), F!(2), F!(2),
        F!(1), F!(0), F!(1), F!(2),
    ];

    let dim_input1 = (2, 4, 4);
    let input1_mat = field_matrix_encoder_multichannel(input1.as_slice(), dim_input1);

    let kernel1 = &[
        F!(1),F!(2),
        F!(1),F!(0),

        F!(3),F!(0),
        F!(1),F!(2),

        F!(1),F!(1),
        F!(2),F!(1),


        F!(1),F!(2),
        F!(0),F!(1),

        F!(0),F!(3),
        F!(3),F!(1),
        
        F!(3),F!(1),
        F!(3),F!(2),
        ];
    let dim_kernel1 = (2, 3, 2, 2);

    let kernel1_mat = field_kernel_encoder_multichannel(kernel1, dim_kernel1);
    
    let strides1 = (1,1);
    let padding1 = (0,0);

    let dim_output1 = (
        dim_kernel1.1,
        get_output_size!(dim_input1.1, dim_kernel1.2, padding1.0, strides1.0),
        get_output_size!(dim_input1.2, dim_kernel1.3, padding1.1, strides1.1),
    );

    let output1 = full_convolution_computation(
        input1, 
        kernel1.to_vec(), 
        dim_input1, 
        dim_kernel1, 
        dim_output1, 
        strides1, 
        padding1,
    );

    let kernel2 = &[
        F!(1),F!(2),
        F!(1),F!(0),

        F!(3),F!(0),
        F!(1),F!(2),


        F!(1),F!(1),
        F!(2),F!(1),

        F!(1),F!(2),
        F!(0),F!(1),


        F!(0),F!(3),
        F!(3),F!(1),
        
        F!(3),F!(1),
        F!(3),F!(2),
        ];
    let dim_kernel2 = (3, 2, 2, 2);

    let kernel2_mat = field_kernel_encoder_multichannel(
        kernel2.as_slice(), 
        dim_kernel2
    );

    let input2: Vec<ark_ff::Fp256<ark_bls12_381::FrParameters>> = output1.clone().into_iter().flatten().flatten().collect();
    let dim_input2 = dim_output1.clone();
    let input2_mat = field_matrix_encoder_multichannel(input2.as_slice(), dim_input2);
    

    let strides2 = (1,1);
    let padding2 = (0,0);

    let dim_output2 = (
        dim_kernel2.1,
        get_output_size!(dim_input2.1, dim_kernel2.2, padding2.0, strides2.0),
        get_output_size!(dim_input2.2, dim_kernel2.3, padding2.1, strides2.1),
    );

    let mut output_randomness = Vec::new();
    for _ in 0..(log2i!(dim_output2.0) + log2i!(dim_output2.1 * dim_output2.2)) {
        output_randomness.push(Fr::rand(&mut rng));
    }

    let output2 = full_convolution_computation(
        input2, 
        kernel2.to_vec(), 
        dim_input2, 
        dim_kernel2, 
        dim_output2, 
        strides2, 
        padding2
    );

    let MLE_output = layer_output_to_mle(&output2, dim_output2, false);

    let MLE_output_eval = MLE_output.fix_variables(&output_randomness)[0];


    let layer1 = LayerInfoConv {
        name: "layer1".to_string(),
        input: input1_mat,
        kernel: kernel1_mat,
        output: output1,
        dim_input: dim_input1,
        dim_kernel: dim_kernel1,
        dim_output: dim_output1,
        padding: padding1,
        strides: strides1,
    };

    let layer2 = LayerInfoConv {
        name: "layer2".to_string(),
        input: input2_mat,
        kernel: kernel2_mat,
        output: output2,
        dim_input: dim_input2,
        dim_kernel: dim_kernel2,
        dim_output: dim_output2,
        padding: padding2,
        strides: strides2,
    };

    let layer_info = vec![layer1, layer2];


    let mut conv_prover = ProverCNN::<E, Fr>::new(
        layer_info.clone(),
        Some(output_randomness.clone())
    );

    let mut fs_rng = Blake2s512Rng::setup();

    let _proof = conv_prover.prove_CNN(&mut fs_rng);

    let proof = conv_prover.layer_outputs.unwrap();

    let claimed_value0_layer1_conv = BigInteger256([5704332772101507324, 7295560598164279407, 7937726300828828800, 1332755173956016179]);
    let claimed_value1_layer1_conv = BigInteger256([11811932763940125553, 16015642421286212788, 12362016604263004873, 8019592712196430650]);
    let claimed_values_layer1_conv = (Fr::from(claimed_value0_layer1_conv), Fr::from(claimed_value1_layer1_conv));

    assert_eq!(proof[0].0.claimed_values, claimed_values_layer1_conv);


    let message0_layer1_conv = vec![
        Fr::from(BigInteger256([7901200363979451809, 14618099655631556076, 6895033451454454075, 6024299575476075821])),
        Fr::from(BigInteger256([8957829696163803963, 15584501080165480522, 4425199564736206386, 3983102902799700799])),
        Fr::from(BigInteger256([6614425390154674456, 932644592925476562, 7986740969217316606, 4240677477057206122])),
    ];
    let message1_layer1_conv = vec![
        Fr::from(BigInteger256([7114222182088304449, 4209993156657698518, 5860130493691328282, 4921835955166069030])),
        Fr::from(BigInteger256([13857063250592130980, 16845318405226358335, 3840165835393142626, 267235321803392869])),
        Fr::from(BigInteger256([5035883813197048584, 5336792706980814367, 4782732819158053786, 2034435164579798868])),
    ];
    let message2_layer1_conv = vec![
        Fr::from(BigInteger256([505644481580602295, 4568600581224210428, 10334339064553688610, 1411636120224541898])),
        Fr::from(BigInteger256([1490939548669420999, 702418334527936036, 13287588443067672788, 5165557326873807027])),
        Fr::from(BigInteger256([3791859573757168428, 14810796079898116095, 770295535518341913, 7553628903283333131])),
    ];

    assert_eq!(proof[0].0.prover_msgs[0].evaluations, message0_layer1_conv);
    assert_eq!(proof[0].0.prover_msgs[1].evaluations, message1_layer1_conv);
    assert_eq!(proof[0].0.prover_msgs[2].evaluations, message2_layer1_conv);


    let evaluations0_layer1_conv = vec![
        Fr::from(BigInteger256([2414972831130067501, 373399252390990750, 9637787143994849160, 8335319469524412399])),
        Fr::from(BigInteger256([8038279517403167403, 4605653395328472154, 13121004287153043231, 3897084645581147343])),
        Fr::from(BigInteger256([10532623342963899725, 5541658249273619942, 10016061644659888886, 5886751118059030970])),
        Fr::from(BigInteger256([11273184235080303119, 3451859230104859897, 13897009278484817586, 8039182085361707400])),
        Fr::from(BigInteger256([7066031076756222048, 14103311690362020521, 4998625180763680692, 2341100350428123157])),
        Fr::from(BigInteger256([3539572849276274643, 4914812830979560718, 10608543032490058036, 3656586348524984657])),
        Fr::from(BigInteger256([7615992398344984307, 2918239475365553116, 9551895031618116433, 3076758017032131736])),
        Fr::from(BigInteger256([140788824326906398, 1300571910347266088, 17202621479535944322, 4887403384676839129])),
        Fr::from(BigInteger256([4313244416010154523, 12685135324978348503, 2853415532581286988, 6142116514518137851])),
        Fr::from(BigInteger256([1917776613743514846, 1158348782208736712, 13238215890070343130, 4675797673139415344])),
        Fr::from(BigInteger256([13592156046995216782, 12453658463291824325, 12691569891615791470, 2556036352502639700])),
        Fr::from(BigInteger256([16368112544609146492, 1106855741553464752, 1409341657538383148, 2336585853284873130])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
    ];
    let evaluations1_layer1_conv = vec![
        Fr::from(BigInteger256([18389368023955520574, 10719342146522064591, 17934797030031901814, 4232440793241660294])),
        Fr::from(BigInteger256([1, 0, 0, 0])),
        Fr::from(BigInteger256([57376045459063750, 13761561335725569327, 4203165942317421454, 4121076066222789057])),
        Fr::from(BigInteger256([18389368023955520573, 10719342146522064591, 17934797030031901814, 4232440793241660294])),
        Fr::from(BigInteger256([9137307964371212891, 3838561478659279928, 6609842022018415279, 2171902760130265766])),
        Fr::from(BigInteger256([3, 0, 0, 0])),
        Fr::from(BigInteger256([57376045459063751, 13761561335725569327, 4203165942317421454, 4121076066222789057])),
        Fr::from(BigInteger256([9194684009830276640, 17600122814384849255, 10813007964335836733, 6292978826353054823])),
        Fr::from(BigInteger256([1, 0, 0, 0])),
        Fr::from(BigInteger256([57376045459063750, 13761561335725569327, 4203165942317421454, 4121076066222789057])),
        Fr::from(BigInteger256([9194684009830276640, 17600122814384849255, 10813007964335836733, 6292978826353054823])),
        Fr::from(BigInteger256([1, 0, 0, 0])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
    ];

    assert_eq!(proof[0].0.polynomial.flattened_ml_extensions[0].evaluations(), evaluations0_layer1_conv);
    assert_eq!(proof[0].0.polynomial.flattened_ml_extensions[1].evaluations(), evaluations1_layer1_conv);

    let claimed_value0_layer1_reshape = BigInteger256([12609711049603505326, 8678121095083528247, 9198918310932471426, 3723462296715956776]);
    let claimed_value1_layer1_reshape = BigInteger256([13001212903437090322, 775184040658528096, 16591086176753458929, 3165457879461286507]);
    let claimed_values_layer1_reshape = (Fr::from(claimed_value0_layer1_reshape), Fr::from(claimed_value1_layer1_reshape));

    assert_eq!(proof[0].1.claimed_values, claimed_values_layer1_reshape);

    let message0_layer1_reshape = vec![
        Fr::from(BigInteger256([9691800523237667350, 18070051511279791003, 2554467945865109274, 7848597588855921825])),
        Fr::from(BigInteger256([14459276318278424295, 13706412569132122322, 9074477253603491178, 1837674444564543706])),
        Fr::from(BigInteger256([15292867004241848313, 13737720571228608579, 7461063612160334330, 7586789754626305747])),
    ];
    let message1_layer1_reshape = vec![
        Fr::from(BigInteger256([5308000896909239971, 13078304120821211235, 3375901558195543007, 5714756246664666318])),
        Fr::from(BigInteger256([4046382252034722455, 8894723618772624686, 3150780381190237402, 7556685088498852396])),
        Fr::from(BigInteger256([15037374481269766627, 5381345218394528198, 17035283982769665985, 4265456314416149811])),
    ];
    let message2_layer1_reshape = vec![
        Fr::from(BigInteger256([3669441261048964787, 17165705335468592380, 12302133714649118921, 4651775808477793711])),
        Fr::from(BigInteger256([9650101392294013005, 7467738064692286282, 6644126866198613345, 7152425984603863456])),
        Fr::from(BigInteger256([17930234965844369502, 4808603954777226823, 233760687362150596, 1960269035971705247])),
    ];

    assert_eq!(proof[0].1.prover_msgs[0].evaluations, message0_layer1_reshape);
    assert_eq!(proof[0].1.prover_msgs[1].evaluations, message1_layer1_reshape);
    assert_eq!(proof[0].1.prover_msgs[2].evaluations, message2_layer1_reshape);


    let evaluations0_layer1_reshape = vec![
        Fr::from(BigInteger256([17696327499451897342, 930795238318861907, 15144717787092627424, 6827566759200340344])),
        Fr::from(BigInteger256([11100273370504059490, 16675450954078683656, 8079361532226368905, 4212466981589379946])),
        Fr::from(BigInteger256([4320027354895574367, 17449891738551947387, 12027512067182527306, 3868193165058872280])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([10186786694738057090, 9482266854690616827, 54728114367593949, 236306875104587801])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([504840928719297419, 2270664516951790828, 14219649775068488594, 234301636845055496])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([4736321994335008408, 15285637482943822729, 5088329089892456986, 3543923057124667433])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([18386641121821016150, 16527092125103579427, 12150220986232669566, 3483126746917047415])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([8446529236909965710, 1949966548967688373, 17809379262423880919, 3182150081361576974])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([12989180065205582419, 18149737315796030471, 3024541048935742936, 6695585632659053053])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
    ];
    let evaluations1_layer1_reshape = vec![
        Fr::from(BigInteger256([6004776391053043386, 12405218135372485709, 7557300437010976782, 7463679789284839923])),
        Fr::from(BigInteger256([2751911338292212650, 15486538870854460453, 12119305186903872387, 6115374266980139911])),
        Fr::from(BigInteger256([6149914294414718756, 14569064388763972372, 7541848502230069614, 1539203179275510602])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([367545362035583633, 7103699081365488348, 16307991887116349508, 5380574073782890341])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([3851088513469827679, 17196028147119817984, 1418824842981359766, 1309266331645194046])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([8699105309304254974, 13976320182834246495, 10801511873063481201, 2248855491384045759])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([15335150651588039540, 13395671163822326058, 14751417636213506503, 236924530332885070])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([6691411544783149398, 17551894952152689034, 13776188705518235503, 4150214262411427769])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([5489328799007955653, 4685923080414600836, 4277462818359441807, 4969975512760863983])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
    ];

    assert_eq!(proof[0].1.polynomial.flattened_ml_extensions[0].evaluations(), evaluations0_layer1_reshape);
    assert_eq!(proof[0].1.polynomial.flattened_ml_extensions[1].evaluations(), evaluations1_layer1_reshape);
    
    let claimed_value0_layer2_conv = BigInteger256([9472613605248589408, 5428956025794511592, 2465031818788007704, 3875903137570307720]);
    let claimed_value1_layer2_conv = BigInteger256([9306310936484015938, 10322526009252915010, 65884931561638135, 7071230868851203899]);
    let claimed_values_layer2_conv = (Fr::from(claimed_value0_layer2_conv), Fr::from(claimed_value1_layer2_conv));

    assert_eq!(proof[1].0.claimed_values, claimed_values_layer2_conv);


    let message0_layer2_conv = vec![
        Fr::from(BigInteger256([13565111685664802300, 13954944741902148624, 2610581594596017271, 5674246197448184463])),
        Fr::from(BigInteger256([17491343433353287347, 757335761719461925, 10279555614976225808, 6402732958732221665])),
        Fr::from(BigInteger256([14728572073815148986, 17135386233833819239, 17231154202040426949, 1943647361248454890])),
    ];
    let message1_layer2_conv = vec![
        Fr::from(BigInteger256([8803263987162442334, 6352446451905069564, 3542377093698199394, 7874924695412216554])),
        Fr::from(BigInteger256([10507313075657985035, 15439481503217137964, 12703562983457721673, 7493342699026895770])),
        Fr::from(BigInteger256([18185428309865821401, 2207230285009603772, 11140497830699526559, 972519137913489794])),
    ];
    let message2_layer2_conv = vec![
        Fr::from(BigInteger256([14885841398392649590, 9731700822916127083, 9995923799698137621, 2925190509430589223])),
        Fr::from(BigInteger256([14869523138985057753, 9003389449933545196, 5991272274650865867, 7791647494867365617])),
        Fr::from(BigInteger256([15516152396294749705, 11229501907198050429, 17839819847799322861, 6550141322635277749])),
    ];

    assert_eq!(proof[1].0.prover_msgs[0].evaluations, message0_layer2_conv);
    assert_eq!(proof[1].0.prover_msgs[1].evaluations, message1_layer2_conv);
    assert_eq!(proof[1].0.prover_msgs[2].evaluations, message2_layer2_conv);


    let evaluations0_layer2_conv = vec![
        Fr::from(BigInteger256([11799709134995833798, 1431057053709634266, 17875493009420728195, 6188108252860828991])),
        Fr::from(BigInteger256([5209029165989874338, 18331084846864072159, 17029158966523524319, 4267713779810387934])),
        Fr::from(BigInteger256([10329034575270669233, 12630473634609506469, 7280987540528308865, 6723743304359284886])),
        Fr::from(BigInteger256([3770278577286662016, 4005199176762434683, 13435580900625243757, 4244898206772400077])),
        Fr::from(BigInteger256([175790672333342858, 8187226079842953361, 6517712894192803945, 1394702315878931114])),
        Fr::from(BigInteger256([1761933330264181019, 13349788872732118352, 2796549808379176219, 8254106160540669578])),
        Fr::from(BigInteger256([11976656114693548006, 7750152924465760321, 15299137452899032658, 4491510695847459297])),
        Fr::from(BigInteger256([9369532861584968785, 17834777852958036837, 11847650272524681622, 2986743225081550150])),
 
    ];
    let evaluations1_layer2_conv = vec![
        Fr::from(BigInteger256([12298866447131430906, 14966025094182782264, 2327818951155929140, 1686508785477896630])),
        Fr::from(BigInteger256([8702864964521273816, 5700479499319960843, 16822264905999518910, 5467196323640378879])),
        Fr::from(BigInteger256([15764229266865396926, 16374382578234797023, 4072335298992813109, 7756847644647274395])),
        Fr::from(BigInteger256([2618751068098686603, 17403592453026831153, 11230093177721089123, 8051945591699949909])),
        Fr::from(BigInteger256([2048865602234681569, 1593678564295031990, 11131073868192988989, 8227319160321115706])),
        Fr::from(BigInteger256([17030246856210267165, 185321080243017230, 9386557520356105020, 2156980301151737941])),
        Fr::from(BigInteger256([6590235627836091455, 13403872484038923111, 17302400615880660501, 4695153540202038125])),
        Fr::from(BigInteger256([8702864964521273816, 5700479499319960843, 16822264905999518910, 5467196323640378879])),
    ];

    assert_eq!(proof[1].0.polynomial.flattened_ml_extensions[0].evaluations(), evaluations0_layer2_conv);
    assert_eq!(proof[1].0.polynomial.flattened_ml_extensions[1].evaluations(), evaluations1_layer2_conv);

    let claimed_value0_layer2_reshape = BigInteger256([3343089411522787842, 8482100651773483362, 9689181486900247863, 4308564379317189725]);
    let claimed_value1_layer2_reshape = BigInteger256([8215182055317823036, 16760354782601028852, 1510863688268609724, 2837837268128385490]);
    let claimed_values_layer2_reshape = (Fr::from(claimed_value0_layer2_reshape), Fr::from(claimed_value1_layer2_reshape));

    assert_eq!(proof[1].1.claimed_values, claimed_values_layer2_reshape);


    let message0_layer2_reshape = vec![
        Fr::from(BigInteger256([10636452914593408083, 5293364805426005993, 7250202198497791697, 5616877659024779003])),
        Fr::from(BigInteger256([17282904760069765646, 6169750628906587901, 17352792592639539276, 6612542338009978068])),
        Fr::from(BigInteger256([17160872524688759537, 102726335397328471, 965653310003520659, 5286767411505279598])),
    ];
    let message1_layer2_reshape = vec![
        Fr::from(BigInteger256([13184092301251715143, 3782135688394914122, 3014317565453000280, 407605533067465352])),
        Fr::from(BigInteger256([4831220811170069894, 9946665963633528726, 16345238040769397293, 3094718973545584333])),
        Fr::from(BigInteger256([3189636202035759890, 10476081579470879842, 9908237942857816231, 6026855172054128252])),
    ];
    let message2_layer2_reshape = vec![
        Fr::from(BigInteger256([12857496899572166681, 5637337948947014274, 3010242050459318587, 1996128647481863717])),
        Fr::from(BigInteger256([18437250186783071773, 15376979179977373465, 12790678587798355583, 76468174385669558])),
        Fr::from(BigInteger256([3011439646558730361, 8617887953520457362, 9666744856194065524, 3365508964092637670])),
    ];

    assert_eq!(proof[1].1.prover_msgs[0].evaluations, message0_layer2_reshape);
    assert_eq!(proof[1].1.prover_msgs[1].evaluations, message1_layer2_reshape);
    assert_eq!(proof[1].1.prover_msgs[2].evaluations, message2_layer2_reshape);


    let evaluations0_layer2_reshape = vec![
        Fr::from(BigInteger256([17586436581806474048, 14968602006845752035, 7289872532056620099, 6305253186980831132])),
        Fr::from(BigInteger256([1, 0, 0, 0])),
        Fr::from(BigInteger256([16726129094198363776, 5456300531443870152, 10888526165473468546, 4256989514497212912])),
        Fr::from(BigInteger256([1, 0, 0, 0])),
        Fr::from(BigInteger256([17586436581806474048, 14968602006845752035, 7289872532056620099, 6305253186980831132])),
        Fr::from(BigInteger256([2, 0, 0, 0])),
        Fr::from(BigInteger256([2, 0, 0, 0])),
        Fr::from(BigInteger256([1, 0, 0, 0])),
        Fr::from(BigInteger256([1720614975216220548, 577858877094212150, 11249436806875854723, 4096527344967236439])),
        Fr::from(BigInteger256([860307487608110274, 9512301475401881883, 14848090440292703169, 2048263672483618219])),
        Fr::from(BigInteger256([1720614975216220549, 577858877094212150, 11249436806875854723, 4096527344967236439])),
        Fr::from(BigInteger256([0, 0, 0, 0])),
        Fr::from(BigInteger256([2, 0, 0, 0])),
        Fr::from(BigInteger256([2, 0, 0, 0])),
        Fr::from(BigInteger256([2, 0, 0, 0])),
        Fr::from(BigInteger256([2, 0, 0, 0])),
    ];
    let evaluations1_layer2_reshape = vec![
        Fr::from(BigInteger256([8878410162431014539, 7349928973940197436, 6789866011093918757, 7936324303729488580])),
        Fr::from(BigInteger256([6281615291728992398, 15408260556718795492, 4687945359603712970, 2768356809280595042])),
        Fr::from(BigInteger256([14028252669351196967, 7763024163887451694, 5829135454353560953, 5183982702597323585])),
        Fr::from(BigInteger256([1459172201193109962, 11536257152129717817, 7959881781988292051, 4315130384652183739])),
        Fr::from(BigInteger256([5704833561525399189, 3226814408122663267, 8680135283662818573, 6656147367137395586])),
        Fr::from(BigInteger256([3333321450284409328, 4557071482466184867, 10535894412027806368, 2750524873646387490])),
        Fr::from(BigInteger256([13910601254268528511, 11437591064749655044, 6215247756752158923, 5673123805703436010])),
        Fr::from(BigInteger256([158023118707327435, 2405738858172838578, 15530631941986128680, 6708334006607230948])),
        Fr::from(BigInteger256([6407323774343469241, 8665247416763120208, 867461355011532425, 6904842435211765766])),
        Fr::from(BigInteger256([9959472414551313708, 7321460936031727055, 2406508485361162222, 1917637482891753076])),
        Fr::from(BigInteger256([3315666676975479589, 4154828807374322009, 5619094129918401469, 1436326715695939669])),
        Fr::from(BigInteger256([6700876932612806117, 1479477385462791299, 12137246699088866459, 6738563917502744980])),
        Fr::from(BigInteger256([10059662617838997630, 12704126094937835299, 14644424004956798708, 4026243686053249290])),
        Fr::from(BigInteger256([11653365118370245353, 7084881321116334682, 16756616298376691580, 7837686788871436171])),
        Fr::from(BigInteger256([5718737131587374778, 1715466105134749026, 16060962759119844354, 4754221146463552082])),
        Fr::from(BigInteger256([11035035793468595906, 5274411185110933358, 3921829652093555558, 5302666893666594938])),
    ];

    assert_eq!(proof[1].1.polynomial.flattened_ml_extensions[0].evaluations(), evaluations0_layer2_reshape);
    assert_eq!(proof[1].1.polynomial.flattened_ml_extensions[1].evaluations(), evaluations1_layer2_reshape);


    let mut verifier = VerifierCNN::new(
        layer_info, 
        proof, 
        Some(MLE_output_eval), 
        Some(output_randomness)
    );

    let mut fs_rng = Blake2s512Rng::setup();

    let _ = verifier.verify_SC(&mut fs_rng);

}


#[test]
fn test_general_prover() {

    let mut rng: rand::prelude::StdRng = test_rng();
    type E = Bls12_381;

    let input1 = vec![
        F!(0), F!(2), F!(0), F!(2),
        F!(1), F!(3), F!(2), F!(2),
        F!(1), F!(1), F!(2), F!(2),
        F!(1), F!(0), F!(1), F!(2),

        F!(1), F!(0), F!(1), F!(2),
        F!(3), F!(1), F!(2), F!(2),
        F!(1), F!(0), F!(2), F!(2),
        F!(1), F!(0), F!(1), F!(2),
    ];

    let dim_input1 = (2, 4, 4);
    let input1_mat = field_matrix_encoder_multichannel(input1.as_slice(), dim_input1);

    let kernel1 = &[
        F!(1),F!(2),
        F!(1),F!(0),

        F!(3),F!(0),
        F!(1),F!(2),

        F!(1),F!(1),
        F!(2),F!(1),


        F!(1),F!(2),
        F!(0),F!(1),

        F!(0),F!(3),
        F!(3),F!(1),
        
        F!(3),F!(1),
        F!(3),F!(2),
        ];
    let dim_kernel1 = (2, 3, 2, 2);

    let kernel1_mat = field_kernel_encoder_multichannel(kernel1, dim_kernel1);
    
    let strides1 = (1,1);
    let padding1 = (0,0);

    let dim_output1 = (
        dim_kernel1.1,
        get_output_size!(dim_input1.1, dim_kernel1.2, padding1.0, strides1.0),
        get_output_size!(dim_input1.2, dim_kernel1.3, padding1.1, strides1.1),
    );

    let output1 = full_convolution_computation(
        input1, 
        kernel1.to_vec(), 
        dim_input1, 
        dim_kernel1, 
        dim_output1, 
        strides1, 
        padding1,
    );

    let kernel2 = &[
        F!(1),F!(2),
        F!(1),F!(0),

        F!(3),F!(0),
        F!(1),F!(2),


        F!(1),F!(1),
        F!(2),F!(1),

        F!(1),F!(2),
        F!(0),F!(1),


        F!(0),F!(3),
        F!(3),F!(1),
        
        F!(3),F!(1),
        F!(3),F!(2),
        ];
    let dim_kernel2 = (3, 2, 2, 2);

    let kernel2_mat = field_kernel_encoder_multichannel(
        kernel2.as_slice(), 
        dim_kernel2
    );

    let input2: Vec<ark_ff::Fp256<ark_bls12_381::FrParameters>> = output1.clone().into_iter().flatten().flatten().collect();
    let dim_input2 = dim_output1.clone();
    let input2_mat = field_matrix_encoder_multichannel(input2.as_slice(), dim_input2);
    

    let strides2 = (1,1);
    let padding2 = (0,0);

    let dim_output2 = (
        dim_kernel2.1,
        get_output_size!(dim_input2.1, dim_kernel2.2, padding2.0, strides2.0),
        get_output_size!(dim_input2.2, dim_kernel2.3, padding2.1, strides2.1),
    );

    let mut output_randomness = Vec::new();
    for _ in 0..(log2i!(dim_output2.0) + log2i!(dim_output2.1 * dim_output2.2)) {
        output_randomness.push(Fr::rand(&mut rng));
    }

    let output2 = full_convolution_computation(
        input2, 
        kernel2.to_vec(), 
        dim_input2, 
        dim_kernel2, 
        dim_output2, 
        strides2, 
        padding2
    );

    let MLE_output = layer_output_to_mle(&output2, dim_output2, false);

    let _MLE_output_eval = MLE_output.fix_variables(&output_randomness)[0];


    let layer1 = LayerInfoConv {
        name: "layer1".to_string(),
        input: input1_mat,
        kernel: kernel1_mat,
        output: output1,
        dim_input: dim_input1,
        dim_kernel: dim_kernel1,
        dim_output: dim_output1,
        padding: padding1,
        strides: strides1,
    };

    let layer2 = LayerInfoConv {
        name: "layer2".to_string(),
        input: input2_mat,
        kernel: kernel2_mat,
        output: output2,
        dim_input: dim_input2,
        dim_kernel: dim_kernel2,
        dim_output: dim_output2,
        padding: padding2,
        strides: strides2,
    };

    let layer_info = vec![LayerInfo::LIC(layer1), LayerInfo::LIC(layer2)];

    let mut general_prover = GeneralProver::<E,Fr>::new(
        layer_info,
        vec![],
        output_randomness,
    );

    general_prover.setup();

    let _gp_output = general_prover.prove_model();


}