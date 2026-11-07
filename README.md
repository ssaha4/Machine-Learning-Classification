# Machine Learning Classification

Three Python experiments that compare a statistical classifier, a neural network implemented in NumPy, and a Keras convolutional neural network. Each experiment uses a different classification task or dataset split, so the accuracies are best read within each experiment.

| Experiment | Approach | Archived test result |
| --- | --- | ---: |
| [PCA and Bayesian classification](pca-bayesian/SrijaSaha_FSLproject1.py) | MNIST digits 0 and 1 with a two-component PCA projection and class-conditional Gaussian densities | 99.20% accuracy |
| [Multilayer perceptron from scratch](mlp/SrijaSaha_FSL_project2.py) | Two-feature binary classification with a NumPy network and manually implemented backpropagation | 82.60% accuracy with 10 hidden units |
| [MNIST CNN experiments](cnn/) | Ten-class digit recognition with a Keras baseline and five training configurations | 98.88% baseline and 98.97% best observed accuracy |

## PCA and Bayesian classification

The script flattens each 28 × 28 image into 784 features. It computes per-feature means and standard deviations from the training set and applies them to both splits. It then eigendecomposes the covariance matrix of the standardized training data and projects both splits onto the two eigenvectors with the largest eigenvalues.

For each digit, the classifier estimates a mean and covariance in the projected space. It combines the resulting multivariate Gaussian density with the class prior and predicts the class with the higher score. The script plots the projection and class densities and reports accuracy and F1.

Run the script from `pca-bayesian/` after supplying `training_data_0.mat`, `training_data_1.mat`, `testing_data_0.mat`, and `testing_data_1.mat`. The original arrays are named `nim0` or `nim1` and have dimensions 28 × 28 × sample count. The script assumes the original class counts. These MAT files were absent from the source archive.

## Multilayer perceptron

The network has two inputs, one ReLU hidden layer, and a sigmoid output. The implementation computes the forward pass, mean squared error, derivatives, and weight updates directly in NumPy. It uses momentum and stops when validation error first rises. The included two-feature datasets are split into 3,000 training, 1,000 validation, and 2,000 test samples.

| Hidden units | Archived test accuracy |
| ---: | ---: |
| 2 | 64.85% |
| 4 | 69.70% |
| 6 | 76.25% |
| 8 | 80.30% |
| 10 | 82.60% |

Run `python SrijaSaha_FSL_project2.py` from `mlp/` so the script can find the four text inputs.

## Convolutional neural network

The baseline uses Keras to load MNIST and classifies digits 0 through 9. The five variants change training settings or architecture, including batch size, epoch count, convolution kernel size, network depth, or optimizer. The archived test accuracies for the five settings were 98.97%, 98.94%, 98.95%, 98.86%, and 93.26%, respectively. The settings are separate experiments and some change several parameters at once.

Run `python baseline.py` or `python setting1.py` from `cnn/` in a TensorFlow/Keras environment. The first run may download MNIST. The scripts use older Keras APIs, so an environment compatible with those APIs may be needed.

The classical and NumPy scripts use NumPy and Matplotlib. The PCA script also uses SciPy and scikit-learn. Results above are from the original reports.
