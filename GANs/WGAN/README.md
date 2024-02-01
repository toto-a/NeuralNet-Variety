
# WGAN 
The principal difference between wgan and generic gan is the wassertein loss (a way to measure the distance between two gaussian) which is more stable for computing the gradient that the traditional gan loss (JS divergence). 
For the loss, in the original paper they imposed a condition on the discriminator or the critic, that is to be 1-Lipshitz as defined in the loss below :

![equation](https://latex.codecogs.com/svg.image?%5Cinline%20%5Cbg%7Bwhite%7D%5Cleft(W_%7Bloss%7D%5Cright)=%5Cmax_%7Bnorm(f)%5Cle%201%7D%5Cleft(%5Cmathbb%7BE%7D_%7Bx%5Csim%20P_r%7D%5Bf(x)%5D-%5Cmathbb%7BE%7D_%7Bx%5Csim%20P_%5Ctheta%7D%5Bf(x)%5D%5Cright))

There are two way to do it one is to clip the gradient (bad !) or interpolate between the fake and real image calculate the gradient and utilize it as a regularization term in the loss 
