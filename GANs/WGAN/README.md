
# WGAN 
The principal difference between wgan and generic gan is the wassertein loss (a way to measure the distance between two gaussian) which is more stable for computing the gradient that the traditional gan loss (JS divergence). 
For the loss, in the original paper they imposed a condition on the discriminator or the critic, that is to be 1-Lipshitz as defined in the loss below :

$$\left(W_{loss}\right) = \max_{norm(f) \le 1} \left( \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{x \sim P_\theta}[f(x)] \right)$$
