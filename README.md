## Classification Using Bayesian Neural Networks

The advantage of using a BNN for classification is its ability to simultaneously quantify uncertainty in relation to both the outputs and weights. To understand how this is done, two types of uncertainty must be defined:

- Epistemic (systemic) uncertainty: this represents the variation in the model weights with repeated training attempts. The idea is that instead of learning specific weight and bias values, the BNN learns weight distributions then used to produce an output that encodes this weight uncertainty

- Aleatoric uncertainty: this represents the variation in model outputs with repeated training attempts. It can effectively be considered a measure of "confidence" in predictive output

A standard 'deterministic' BNN only captures epistemic uncertainty; it produces a point estimate prediction for a given sample. The two primary methods of doing this are prior (fixed) and posterior (dynamic) inference:

- Since the task is classification rather than regression, the prior weight distribution will remain normalized at 0 with a std-dev of 1. This is untrainable, and remains fixed

- Posterior inference is approximated using a variational distribution, which minimizes KL divergence (relative entropy: given P and Q, KL divergence is the cross-entropy of P and Q minus the cross-entropy of P with itself) as a sort of distance function.

On the other hand, a 'probabilistic' BNN captures aleatoric uncertainty as well, counting for the stochastic noise in the process. This involves a few modifications to the output layer of the 'deterministic' BNN:

- Two output layers are instantiated to learn both mean and variance of the distribution

- Since our task is classification rather than regression, both will be initialized as independent Bernoulli distributions

As the outputs are measures of probability, the accuracy of the model can be measured to specific confidence intervals. The output uses an independent Bernoulli distribution, which has two centers over 0 and 1. This will output means and deviations for the classifications, which function as confidence intervals. The more 'confident' the model is, the smaller the deviations will be.

All experimentation will be done using a [mushroom dataset](https://www.kaggle.com/datasets/uciml/mushroom-classification) on Kaggle.
