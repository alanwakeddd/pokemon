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
<br />
One of the primary limitations of the original project is the small amount of training data. I tested on various 
images and at times the classifications were incorrect. When this happened, the input image + network more closely 
and found that the color(s) most dominant in the image influence the classification dramatically. 

Firstly, we built a new dataset contains 6000+ images of 12 pokemons and tested it on original CNN network.
<br />
![alt text](cnn_new_dataset.png) <br />
<br />
Then, we tried several different model to deal with this multi-classification.
<br />
Method 1 Description: <br />
Loading the pre-trained parameters from VGG16 and applying its 18 layers to our 5000 data points.

Method 1 Results: <br />
![alt text](agg.png) <br />

Method 2 Description: <br />
Using PCA to reduce the dimensionality (each image is a 67500-dimension vector) of the dataset and achieve high accuracy at the mean time. <br />
<br/>
1st Approach: Using GridSearchCV to fit PCA.
Choosing 4 classes. Each picture is 50x50 size and tranfer to grayscale. So there are 2500 features.
Parameters are choosing by observation. Random select a range of npc, c, gamma at first. Then change the range by    observe the color map.
Results: <br />
![alt text](pca1.png) <br />
<br/>
2nd Approach: <br />
- Create two folders (train and test) and store all the pokemon images of the selected nine kinds (Arcanine, Bulbasaur, Charizard, Eevee, Lucario, Mew, Pikachu, Squirtle, and Umbereon) into separated folders
- Use ImageDataGenerator to tranform image data into data point matrices and combine train and test for scaling
- 
Method 2 Results: <br />
Still working on...
