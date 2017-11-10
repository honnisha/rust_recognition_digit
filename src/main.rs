extern crate mnist;
extern crate rulinalg;
extern crate rusty_machine;

use mnist::{Mnist, MnistBuilder};
use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};

use rusty_machine::learning::nnet::{NeuralNet, BCECriterion};
use rusty_machine::learning::toolkit::regularization::Regularization;
use rusty_machine::learning::optim::grad_desc::StochasticGD;
use rusty_machine::learning::SupModel;

fn main() {
    let (trn_size, rows, cols) = (50_000, 28, 28);

    // Deconstruct the returned Mnist struct.
    let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(trn_size)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    // Get the label of the first digit.
    let first_label = trn_lbl[0];
    println!("The first digit is a {}.", first_label);

    // Convert the flattened training images vector to a matrix.
    let trn_img = Matrix::new((trn_size * rows) as usize, cols as usize, trn_img);

    // Get the image of the first digit.
    let row_indexes = (0..27).collect::<Vec<_>>();
    // let first_image = trn_img.select_rows(&row_indexes);
    // println!("The image looks like... \n{}", first_image);

    // Convert the training images to f32 values scaled between 0 and 1.
    let trn_img: Matrix<f32> = trn_img.try_into().unwrap() / 255.0;

    // Get the image of the first digit and round the values to the nearest tenth.
    let first_image = trn_img.select_rows(&row_indexes)
        .apply(&|p| (p * 10.0).round() / 10.0);
    println!("The image looks like... \n{}", first_image);

    let inputs = rusty_machine::linalg::Matrix::new((trn_size * rows) as usize, cols as usize, trn_img);
    let targets = rusty_machine::linalg::Matrix::new(rows as usize, cols as usize, vec![1.,0.,0.,0.,1.,0.,0.,0.,1.,
                                        0.,0.,1.,0.,0.,1.]);

    // Set the layer sizes - from input to output
    let layers = &[3,5,11,7,3];

    // Choose the BCE criterion with L2 regularization (`lambda=0.1`).
    let criterion = BCECriterion::new(Regularization::L2(0.1));

    // We will just use the default stochastic gradient descent.
    let mut model = NeuralNet::new(layers, criterion, StochasticGD::default());

    // Train the model!
    model.train(&inputs, &targets).unwrap();

    let test_inputs = rusty_machine::linalg::Matrix::new(2,3, vec![1.5,1.5,1.5,5.1,5.1,5.1]);

    // And predict new output from the test inputs
    let outputs = model.predict(&test_inputs).unwrap();
    println!("outputs: {}", outputs);
}
