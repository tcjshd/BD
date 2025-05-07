from google.colab.patches import cv2_imshow
import cv2
from google.colab import files
import random
uploaded = files.upload()
filename = next(iter(uploaded))
image = cv2.imread(filename)


cv2_imshow(image)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray)


import numpy as np
def convolution(image,filter,padding=0,stride=1):
  img_h,img_w=image.shape
  filt_h,filt_w=filter.shape

  if padding>0:
    image=np.pad(image,((padding,padding),(padding,padding)), mode='constant',constant_values=0)
  out_h=(image.shape[0]-filt_h)//stride+1
  out_w=(image.shape[1]-filt_w)//stride+1
  result=np.zeros((out_h,out_w))

  for i in range(out_h):
    for j in range(out_w):
      patch=image[i*stride:i*stride+filt_h,j*stride:j*stride+filt_w]
      value=np.sum(patch*filter)
      result[i,j]=max(0,value)
  return result

def pooling(image,pool_size=2,stride=1):
  img_h,img_w=image.shape
  out_h=((img_h-pool_size))//stride+1
  out_w=((img_w-pool_size))//stride+1
  result=np.zeros((out_h,out_w))

  for i in range(out_h):
    for j in range(out_w):
      patch=image[i*stride:i*stride+pool_size,j*stride:j*stride+pool_size]
      result[i,j]=np.max(patch)
  return result

conv_img=convolution(gray,np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))

pool_img=pooling(conv_img)

flattend=pool_img.flatten()
weight=np.random.rand(flattend.shape[0])
net=np.dot(flattend,weight)
out=1/(1+np.exp(-net))
print(out)