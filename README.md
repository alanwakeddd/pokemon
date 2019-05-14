# Pokemon Classification
This is a group project for ECE-GY 6143 Introduction to Machine Learning at NYU Tandon.<br />
<br />
Pokemon, a popular TV show, video game, and trading card series, originated from Japan
and has been spread all over the world since 1996. A Pokedex, a device that exists in the
world of Pokemon, is used to recognize Pokemon by scanning or learning from Pokemon
images. The current model we found on Pyimagesearch trains the CNN using Keras and
deep learning to build the underneath model of Pokedex. Because of the limited size of its
dataset and the single approach it applies, the model accuracy is not ideal. We would like to
build a better-performed model on top of the existing one by increasing the size of the
dataset and introducing different approaches: pre-trained VGG16 and PCA.

Group 77 members are as below: <br />
Qin Hu (N17006855) <br />
Bohan Zhang (N13992422) <br />
Yichi Zhang (N19888469) <br />
Xintong Song (N13489466) <br />

Method 1 Description: <br />
Loading the pre-trained parameters from VGG16 and applying its 18 layers to our 5000 data points.

Method 1 Results: <br />
![alt text](agg.png) <br />

Method 2 Description: <br />
Using PCA to reduce the dimensionality (each image is a 22500-dimension vector) of the dataset and achieve high accuracy at the mean time. <br />

Method 2 Results: <br />
Still working on...
