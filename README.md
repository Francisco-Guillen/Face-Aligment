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

## Datasets

**[VGGFace2](https://paperswithcode.com/dataset/vggface2-1):** is a dataset comprised of approximately 3.31 million images categorized into 9131 classes, each corresponding to an individual identity, and primarily used for training and validation.
Due to its considerable size, only the first 300 classes were utilized, totaling around 110,000 images. This selection was made to ease the handling of the dataset while maintaining a significant representation of the identities present.

**[LFW](https://vis-www.cs.umass.edu/lfw/):** On the other hand, LFW (Labeled Faces in the Wild) consists of around 13,000 facial images encompassing over 5,700 different individuals. This dataset serves as a benchmark for testing face recognition systems.


## Facial Alignment Methods

### InsightFace

InsightFace is an open source library integrated into Python for 2D and 3D facial analysis. 
This library uses Retina-Face for facial detection and the 2d106 and 3d68 models for extracting facial reference points. Thus, in the case of a 2D facial alignment, it is possible to extract 106 reference points from the face. 

### MTCNN (Multi-task Cascaded Convolutional Networks)

CNN-based facial detection and alignment method, which is capable of extracting 5 facial reference points, one for each eye, one for the nose and one for each corner of the mouth.

### Dlib

Facial detection and alignment method, capable of extracting up to 68 facial reference points. Dlib is a linear regression model that uses previously labelled fiducial points in images to learn patterns and thus be able to predict the location of fiducial points in new images. 
<div align="center">
  <table>
    <tr>
      <td style="text-align: center;">
        <img src="https://github.com/Francisco-Guillen/Face-Aligment/assets/83434031/d1aa0a03-7fb5-48cb-bb2b-8b5b9fe63f13"><br>
        <p>Figure 1: InisghtFace</p>
      </td>
      <td style="text-align: center;">
        <img src="https://github.com/Francisco-Guillen/Face-Aligment/assets/83434031/e725f8ce-314b-4f4b-ae19-e5f4ef24032b"><br>
        <p>Figure 2: MTCNN</p>
      </td>
      <td style="text-align: center;">
        <img src="https://github.com/Francisco-Guillen/Face-Aligment/assets/83434031/b83e7cbb-bdd1-482c-bc38-3ae12a3f3958"><br>
        <p>Figure 3: Dlib</p>
      </td>
    </tr>
  </table>
</div>

## Geometric Transformations

### Affine Transformation
This is a linear transformation that preserves parallelism between lines, angles and planes. It has a wide range of functionalities, such as rotation, scaling, translation and shearing, thus allowing you to adjust the position and orientation of facial elements.

### Similarity Transformation 
This is a type of geometric transformation that preserves angles and the ratio of distances between points. 

### Projective Transformation
This is the most advanced transformation, which, unlike the affine transformation, does not preserve the parallelism between the lines. 

<div align="center">
  <table>
    <tr>
      <td style="text-align: center;">
        <img src="https://github.com/Francisco-Guillen/Face-Aligment/assets/83434031/2d059767-9fc9-40b8-b189-3d539fc0d0d7"><br>
        <p>Figure 3: Affine Transformation</p>
      </td>
      <td style="text-align: center;">
        <img src="https://github.com/Francisco-Guillen/Face-Aligment/assets/83434031/95075b2d-3994-4215-9f19-daa8ca0de332"><br>
        <p>Figure 4: Similarity Transformation</p>
      </td>
      <td style="text-align: center;">
        <img src="https://github.com/Francisco-Guillen/Face-Aligment/assets/83434031/348c8334-4e63-4d6b-9832-069d81a8f813"><br>
        <p>Figure 5: Projective Transformation</p>
      </td>
    </tr>
  </table>
</div>

## Model Training

We utilize LightCNN, a Convolutional Neural Network developed by Alfred Xiang Wu, to train our models. It is designed to provide an efficient and lightweight solution for facial recognition tasks. To use this network for training the models, you can refer to the GitHub repository at [Light CNN for Deep Face Recognition, in PyTorch](https://github.com/AlfredXiangWu/LightCNN).

- Epochs: 30
- Batch size: 32
- Learning rate: 0.001

<div align="center">
  <table>
    <tr>
      <td style="text-align: center;">
        <img src="https://github.com/Francisco-Guillen/Face-Aligment/assets/83434031/837e436b-40aa-4f43-8b7c-78faf15ea1f7"><br>
        <p>Figure 6: Loss over the epochs in each method</p>
      </td>
      <td style="text-align: center;">
        <img src="https://github.com/Francisco-Guillen/Face-Aligment/assets/83434031/e29d4134-4fdd-4e74-aaf9-0359dc8c3d01"><br>
        <p>Figure 7: Loss over the epochs in each transformation</p>
      </td>
      <td style="text-align: center;">
        <img src="https://github.com/Francisco-Guillen/Face-Aligment/assets/83434031/ec1d291b-3bf3-4366-af66-33264d8defe1"><br>
        <p>Figure 8: Loss over the epochs using different facial reference points.</p>
      </td>
    </tr>
  </table>
</div>

## Model Evaluation
<div align="center">

<table>
<tr><td>

|| Accuracy |
|--|--|
|Affine (InsightFace)| 89.40%|
|Affine (MTCNN)| 89.83%|
|Affine (Dlib)| 90.10%|
|Projective (InsightFace)| 88.29%|
|Similarity (InsightFace)| 89.28%|
|Similarity (Dlib) 3 points (Eyes and Mouth)| 89.31%|
|Similarity (Dlib) 3 points (Eyes and Nose)|89.97%|
|Similarity (Dlib) 7 points|89.90%|
|Similarity (Dlib) 8 points|89.50%|
</td></tr> </table>
</div>

Features are extracted from the pre-processed LFW dataset by each of the algorithms in order to assess their performance. To this end, after extracting the features, it is possible to measure the accuracy of each of the algorithms using the (ROC) curve and the (AUC) based on the cosine distance between the embeddings of the image pairs.

It can be seen that the most efficient method for image pre-processing was dlib in conjunction with the affine transformation, with an accuracy of 90.10%, followed by the MTCNN method in conjunction with the affine transformation, which achieved an accuracy of 89.83%.
It is also possible to observe that choosing a greater number of facial reference points is not always the best, since using only three reference points, the outside of each eye and the centre of the nose, gives a better accuracy than using seven reference points or even 68. 
This may be because fewer points minimise the impact of noise and point detection errors, making the facial alignment process more robust. Or because by using fewer points, overfitting can be avoided, allowing for better generalisation for faces with different characteristics.

## Refferences
- [LFW](https://vis-www.cs.umass.edu/lfw/)
- [VGGFace2](https://paperswithcode.com/dataset/vggface2-1)
- [InsightFace](https://en.wikipedia.org/wiki/Netpbm](https://github.com/deepinsight/insightface))
- [LightCNN](https://www.geeksforgeeks.org/quad-tree/](https://github.com/AlfredXiangWu/LightCNN))
- [Face Alignment with OpenCV and Python](https://v2.ocaml.org/api/Array.html](https://pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/)https://pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/)
