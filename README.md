# Machine Learning Course at IPGP

This working groups mainly is based on the online book [Deep Learning](http://www.deeplearningbook.org) by Ian Goodfellow, Yoshua Bengio and Aaron Courville.

### Course 1 (Nov, 24)

 * Read part 5.1
 * **dataset**: data points/features/design matrix
 * **learning algorithm**: task, performance, experience, classification, regression, supervised, unsupervised

### Course 2 (Dec, 2)
 
* Read part 5.2 (Capacity, overfitting and underfitting, The No Free Lunch Theorem, Regularization)
* Skimmed trough part 5.3 (Hyperparameters and Validation Sets)

### Course 3 (Jan, 13)

* Read part 3.13 (information, entropy, KL divergence)
* Read 5.5 (loss, maximum likelihood)
* what we learned: the model can be a probability distribution that is parametrized e.g. through its mean and variance.
  The maximum likelihood method tries to find the model that makes the observed data most likely. I.e. for independent
  data points, this is maximizing the product of the probabilities to observe a specific data point.

### Course 4 (Jan, 20) 

* Started working with SCIKIT-LEARN on real datasets
* linear regression on SCARDEC database [scardec_analysis.py](scardec/scardec_analysis.py)
  * `design matrix = (n_events, npoints_stf) = [n_samples, n_features]`
  * `label = magnitude = [n_events]`
* what we learned: linear regression in very high dimensional space. The coefficients of the
  learned model describe the 'gradient' of the source time function with respect to the magnitude

### Course 5 (Jan, 27)

* linear regression **with regularization** on SCARDEC database [scardec-lsq+damping.py](scardec/scardec-lsq+damping.py)
* what we learned:
 * the gradient becomes positive definite. This means that the magnitude is 'sensitive' to the integral of the source time    function over time.
 * the damping stabilizes the gradient
 * damping and regularization can fundamentally alter the result

### Course 6 (Feb, 3)
* microseism database. Example with linear regression between spectrum and ocean depth [microseisms/analysis.py](microseisms/analysis.py)
* Read 5.7 (Supervised Learning) 
* discussion on non-linearities
* XOR problem
* Kernel-trick: move to higher dimension to include non-linearities in a linear regression 
* [Kenerl-trick illustration](https://www.youtube.com/watch?v=9NrALgHFwTo)
* Kernel Ridge Regression on microseism database: [microseisms/analysis-KRR.py](microseisms/analysis-KRR.py)

### Course 7 (Feb, 10)
* Kernel-trick : theoretical overview
* template-matching explained with kernel-trick
* Kernel examples : RBF (Radial Basis Functions), multipolynomials, etc.
* Illustration with many 2D examples with (x) and (.) labels
* What is a **direction** and an **amplitude** in high dimensional space ? What is the direction of a function ?
* Discussions about PCA (unsupervised, finds linear directions in feature space) and KPCA (Kernel-PCA)

### Course 8 (Feb, 20) 
* Came from PCA to KPCA on real dataset. 
 * The principal component are showing the directions of the greatest variance. 
 * The projection of the matrix onto the principal components shows which principal axes best fit the sample.

* Revisited kernel-trick in 2D. 
  * a kernal of infinite polynomial combination of features is the RBF kernel
  * the RBF kernel selects closely-located samples in high-dimensional space)
  * Cosine kernel looks to take into account direction only

* Non-linear methods allow to better "predict the data"

### Course 9 (Thu, 28)
* Finished reading chapter 5 (move to deep learning)
* machine learning algorithm:
  * data + model + cost + optimization. Tensorflow is designed to solve this problem.
* A high dimensional feature space is often only sparsely covered with training data. The actual
  allowable values often live on a manifold and don't cover the whole space. Deep learning can
  better retrieve such a manifold and there generalize better than some kind of nearest neighbour
  algorithm which most classical non-linear machine learning algorithms are based on.
* We implemented in tensorflow:
  * linear regression
  * convolutional neural network
  * 2 layered autoencoder that compares to the PCA. We might need to add some sparsity penalty to get
    a better categorical encoding. [description of autoencoders](http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity), [another description here](https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf)
    
### Course 10 (Mar 10, 14h room 3000)
* demonstrate tensorflow once more
* Started drawing a [mind map](https://github.com/IPGP/machine_learning_scripts/blob/master/machinelearning.mm)

### Course 11 (Mar 24, 14h room 3000)
* Read 6 (intro) and discussed XOR problem (6.1) 
* Introduced neural networks:
 * unit (neuron, vector input, scalar output, activation function)
 * layer (collection of units, can be defined with a matrix $W$ and bias $b$ with an activation function)
 * activation function examples: ReLU, sigmoid, tanh (and linear)

### Course 12 (Mar 28, 16h observatories)
* simple neural network with TensorFlow
* fit sine function with neural network
* we saw the importance of initial values

### Course 13 (Apr 4, 15h, room 3000)
* how to chose the initial values and the activation functions together
* activation functions for output units
* fly above chapter 6: architectures of neural networks

### Course 14 (Apr 13 (?))
* convolutional neural networks
* autoencoder

### Course 15 (May 5)
* VGG 16
* style transfer

### some remaining objectives
- [] restricted Boltzmann machines
- [] autoencoder loss functions
- [] sparse coding
