# machine_learning_scripts

This working groups mainly is inspired from the online book [Deep Learning](http://www.deeplearningbook.org) Ian Goodfellow, Yoshua Bengio and Aaron Courville.

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

### Course 4 (Jan, 20) 

* Started working with SCIKIT-LEAN on real datasets
* linear regression on SCARDEC database [scardec_analysis.py](scardec/scardec_analysis.py)
  * `design matrix = (n_events, npoints_stf) = [n_samples, n_features]`
  * `label = magnitude = [n_events]`
* what we learned: linear regression in very high dimensional space. The coefficients of the
  learned model describe the 'gradient' of the source time function with respect to the magnitude

### Course 5 (Jan, 27) 

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
* What is a **diretion** and an **amplitude** in high dimensional space ? What is the direction of a function ?
* Discussions about PCA (unsupervized finding linear directions direct feature space) and KPCA (Kernel-PCA)


### Course 8 (Feb, 20) 
* Test KPCA on real dataset. This would summarize the last courses about kernel-tricks.
* Test k-means (see 5.8) clustering (ont the same dataset?)
* Read small transition to deep-learning (part 5.11)

### General objectives
- [ ] Go towards deep leanring
- [ ] Move to [Tensor Flow](https://www.tensorflow.org)
- [ ] General theory of neural networks 



 
 


