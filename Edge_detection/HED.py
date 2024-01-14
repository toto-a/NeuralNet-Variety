import cv2
from matplotlib import pyplot as plt
import numpy as np

###########################################################
#Check traditional edge detectors first. e.g., canny, sobel, roberts, etc.

# load the input image and grab its dimensions, for future use while defining the blob
img = cv2.imread("1-1.png")
print(img.dtype)
plt.imshow(img)

#Canny
canny_edge = cv2.Canny(img,50,150)
plt.imshow(canny_edge)

#Autocanny
sigma = 0.3
median = np.median(img)

# apply automatic Canny edge detection using the computed median
lower = int(max(0, (1.0 - sigma) * median))
upper = int(min(255, (1.0 + sigma) * median))
auto_canny = cv2.Canny(img, lower, upper)
plt.imshow(auto_canny)


################################################################
"""
Steps for edge detection using HED:
    
1. Define the network and load the pre-trained model.​
2. Create blob from the image – basically create a preprocessed image​
3. Load pretrained model (you need both the proto text and caffe model files)​
4. Pass the blob image through model​ (forward pass)
5. Get output​
"""
# The pre-trained model that OpenCV uses has been trained in Caffe framework
#Download from the link above
protoPath = "hed_model/deploy.prototxt"
modelPath = "hed_model/hed_pretrained_bsds.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


# load the input image and grab its dimensions, for future use while defining the blob
img = cv2.imread("1-1.png")
plt.imshow(img)
(H, W) = img.shape[:2] #Height, width

# construct a blob out of the input image 
#blob is basically preprocessed image. 
#OpenCV’s new deep neural network (dnn ) module contains two functions that 
#can be used for preprocessing images and preparing them for 
#classification via pre-trained deep learning models.
# It includes scaling and mean subtraction
#How to calculate the mean?
mean_pixel_values= np.average(img, axis = (0,1))
blob = cv2.dnn.blobFromImage(img, scalefactor=0.7, size=(W, H),
                             #mean=(mean_pixel_values[0], mean_pixel_values[1], mean_pixel_values[2]),
                             mean=(105, 117, 123),
                             swapRB= False, crop=False)

#View image after preprocessing (blob)
blob_for_plot = np.moveaxis(blob[0,:,:,:], 0,2)
plt.imshow(blob_for_plot)


# set the blob as the input to the network and perform a forward pass
# to compute the edges
net.setInput(blob)
hed = net.forward()
hed = hed[0,0,:,:]  #Drop the other axes 
#hed = cv2.resize(hed[0, 0], (W, H))
hed = (255 * hed).astype("uint8")  #rescale to 0-255

plt.imshow(hed, cmap='gray')


plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.title("original mage")
plt.imshow(img)
plt.subplot(132)
plt.title("Canny")
plt.imshow(auto_canny,cmap='gray')
plt.subplot(133)
plt.title("HED")
plt.imshow(hed,cmap='gray')
plt.show()