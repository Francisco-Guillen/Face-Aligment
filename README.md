# Evaluation of the Impact of Face Alignment Algorithms on the Performance of Face Recognition Methods

## Overview
With the advance of technology, facial recognition has been widely used in various areas, such as security, access control, crowd monitoring, payment systems and even on social networks, for the automatic identification of people in photos. The facial recognition process usually involves capturing an image or video of the person's face, which is then pre-processed in order to extract the relevant features. 
Data pre-processing is done through facial alignment, which usually consists of ensuring that the position of the eyes, mouth and nose are in the same position for all images. As such, image pre-processing consists of four points:
- Detecting the face in the image;
- Detecting the reference points of the face;
- Obtaining the target points;
- Applying the chosen transformation (Affine, Similarity, Projective).

In this project, the study of image pre-processing was divided into 3 stages: firstly, we studied some facial alignment methods, secondly, the transformations that can be applied and thirdly, how the choice of the number of facial reference points can influence the algorithm. 

## Getting Started

To run any this project, you'll need to install the required libraries listed in requirements.txt file. You can do this by writing the following command in your Linux terminal:

```bashrc
pip install -r requirements.txt
```
Once you have installed all the libraries, you'll also need to clone the LightCNN repository, which will be used to train the different methods. You can do this by writing the following command in your Linux terminal:

```bash
git clone https://github.com/AlfredXiangWu/LightCNN.git
```
Make sure to execute these commands in the correct directory where you want the files to be installed and cloned.


## Facial Alignment Methods

### InsightFace

InsightFace is an open source library integrated into Python for 2D and 3D facial analysis. 
This library uses Retina-Face for facial detection and the 2d106 and 3d68 models for extracting facial reference points. Thus, in the case of a 2D facial alignment, it is possible to extract 106 reference points from the face. 

### MTCNN (Multi-task Cascaded Convolutional Networks)

CNN-based facial detection and alignment method, which is capable of extracting 5 facial reference points, one for each eye, one for the nose and one for each corner of the mouth.

### Dlib

Facial detection and alignment method, capable of extracting up to 68 facial reference points. Dlib is a linear regression model that uses previously labelled fiducial points in images to learn patterns and thus be able to predict the location of fiducial points in new images. 

<div align="center" style="padding: 80px;">
  <img src="https://github.com/Francisco-Guillen/Face-Aligment/assets/83434031/d1aa0a03-7fb5-48cb-bb2b-8b5b9fe63f13" align="left">
  <img src="https://github.com/Francisco-Guillen/Face-Aligment/assets/83434031/e725f8ce-314b-4f4b-ae19-e5f4ef24032b" align= "center">
  <img src="https://github.com/Francisco-Guillen/Face-Aligment/assets/83434031/b83e7cbb-bdd1-482c-bc38-3ae12a3f3958" align="right">
</div>
