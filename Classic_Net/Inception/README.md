

# Inception 

## Overview
The key innovation of the Inception model is the so-called "Inception module". Traditional convolutional layers use filters of a single size at each layer (for example, 3x3 or 5x5). In contrast, an Inception module carries out several convolutions with different filter sizes in parallel, and then concatenates the results. This allows the model to learn features at different scales and complexities.

The Inception model also introduced two other important concepts:

- Auxiliary classifiers: These are additional classifiers that are added part-way through the network, which contribute to the final classification score. They help to mitigate the vanishing gradient problem during training.

- Efficient 'bottleneck' layers: Before expensive convolutions with large filters (e.g., 5x5), the model uses 1x1 convolutions to reduce the dimensionality of the input, making the computation more efficient.


## Architecture 

![inception][assets/inception.png]