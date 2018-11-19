# Face Recognition and Face Verification on Nvidia Jetson TX2

<br />Tensorflow implementation of Face Verification and Recognition using th on-board camera of TX2.Facenet and DeepFace implementations for the same are taken as inspiration.These models are compared to a naive K-means clustering approach for recognition tasks.



## Getting Started

The neural network was trained on Nvidia Titan X  GPU.This model was later used with nvidia Jetson TX2 Board.
The K-means clustering approach was directly implemented on Jetson TX2.

## Prerequisites

1.Python 3.5 <br />
2.Tensorflow 1.5<br />
3.Keras <br />
4.Scikit Learn<br />
5.Open CV 3.4.1<br />
4.Face recognition<br />

## Installing

##### I.Download 
Pre-trained model weights: <br/>

1.FaceNet model weights are directy downloaded from their Github repo.<br />
2.inception_model.py builds the complete network architecture as developed by Szegedy.<br />

 
##### II. Open CV<br />

1.Open CV installation for ubuntu is pretty standard and done in same way as shown on their website.<br />
2.Installation of Open CV on Jetson TX2 is a bit diffrent.Jetson Hacks has a pretty cool script on his Git repo.This enable open CV on GPU on the Jetson.<br />


##### III.Face recognition
<br />
```
sudo pip3 apt-get install face-recognition
```
This is required for the k-means clustering approach and not the Facenet.This is optional  <br />
 <br />




## Running the code

##### 1.Directory Structure
```
---------------------------------
Face_Recognition
|-fr_utils.py
|-main.py
|-inception_model.py
|-face_recognition_with_jetson.py
|-train-|
|       |-sid1.jpg
|       |-images to train
|-datasets-|
|          |-h5 models datasets as used
|-weights-|
|         |-csv files for tensorflow graph of pretrained Facenet          
------------------------------------

```


##### 2.Dataset pairs <br />

|-Training will use triplets of images (A,P,N):<br />
|---A is an "Anchor" image--a picture of a person.<br />
|---P is a "Positive" image--a picture of the same person as the Anchor image.<br />
|---N is a "Negative" image--a picture of a different person than the Anchor image.<br />

![alt text](https://github.com/siddharthbhonge/Face_Recognition_with_jetson_TX2/blob/master/triplet_loss.png)
The facenet aims to minimize this triplet loss.<br />
Here f(A,P,N) stands for the embeddings of each of the input image<br />


##### 3.Run Face verification<br />
```
verify("train/sid1.jpg", "siddharth", database, FRmodel)
It's kian, welcome home!
(0.08123432, True)
```

##### 4.Run Face Recognition<br />

|-Find the encoding from the database that has smallest distance with the target encoding.<br />
|--Initialize the min_dist variable to a large enough number (100). It will help you keep track of what is the closest encoding to the input's encoding.<br/>
|--Loop over the database dictionary's names and encodings. To loop use for (name, db_enc) in database.items().<br/>
|---->Compute L2 distance between the target "encoding" and the current "encoding" from the database.<br/>
|----If this distance is less than the min_dist, then set min_dist to dist, and identity to name.<br/>




##### 5.Using K-means clustering for Face reogniton<br />
Run the face_recognition_with_jetson.py directly on Jetson.<br />



## Results

Both The Facenet and k-means approaches work pretty well.Accuracy wise facenet is much more reliable over large datasets.But k-means is a lightweight option with pretty descent accuracy and does not require huge computation resources. 

## Authors

* **Siddharth Bhonge** - *Parser /Model* - https://github.com/siddharthbhonge


## Acknowledgments

* Andrew Ng  | Deeplearning.ai<br />
*Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering
*Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). DeepFace: Closing the gap to human-level performance in face verification<br />
*The pretrained model we use is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.<br />
*Our implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet<br />

