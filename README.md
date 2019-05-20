<h2># Pokemon Classification</h2>
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

<h3>Group 77 members are as below: <br /></h3>
<ul>
<li>Qin Hu (N17006855) <br /></li>
<li>Bohan Zhang (N13992422) <br /></li>
<li>Yichi Zhang (N19888469) <br /></li>
<li>Xintong Song (N13489466) <br /></li>
</ul>
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
<h2>Method 1 Description: <br /></h2>
Loading the pre-trained parameters from VGG16 and applying its 18 layers to our 5000 data points.

Method 1 Results: <br/>>
![alt text](agg.png) <br/>

<h2>Method 2 Description: <br/></h2>
Using PCA to reduce the dimensionality (each image is a 67500-dimension vector) of the dataset and achieve high accuracy at the mean time. <br/>
<br/>
<h3>1st Approach: Using GridCV to do the optimum parameter search.<br/></h3>
<p>Due to low efficiency of GridSearch, even the Nvidia Tesla P100 GPU can run hours for a full size dataset( 6000,150,150,3) opreation. Therefore, we load only 4 of 12 classes,200 pictures per class as dataset for this particular method. Each picture is shrinked to 50x50 by pixel and transferred to grayscale. So there are 2500 features (800,50,50)->(800,2500).</p><br/>
<p>Parameters are choosed by observation. At first, guess the range of parameters (npc, c, gamma). Fit data using GridSearchCV, then we can find the converge trendency by observing the color map. It allows us to make a more "educated" guess. Fit data using the new range of parameters. Repeat this process until global maximum are showing in the map.</p><br/>
Results: <br/>
<img src="pca1.png" width=350" height="400">
<br/>
<br/>
<h3>2nd Approach: <br /></h3>
- Create two folders (train and test) and store all the pokemon images of the selected nine kinds (Arcanine, Bulbasaur, Charizard, Eevee, Lucario, Mew, Pikachu, Squirtle, and Umbereon) into separated folders
- Use ImageDataGenerator to transform image data into data point matrices and combine train and test for scaling. At this point, the entire mini batch has 1000 images, each of which has a dimension of 67,500 (150 * 150 * 3)
- Use StandardScaler() to rescale the data X and fit PCA to find the minimum number of PCs that make PoV greater or equal to 90%
- Create an array of number of PCs for test (2 to the minimum number we just found) and an array of Gamma.
- In the for loops of PC for test and Gamma for test, fit PCA on training data and SVD on transformed training data in each iteration and find the parameters which make the best accuracy <br />

Limitation: large dimension of almost 70,000 features but only 1,000 data points. PCA works as “feature selection” that gets rid of noises or correlations inside an image before applying any classifier. It does not work well in this case because some weird images (i.e.: pokemon on a T-shirt) are hard to detect. <br />

Results: <br />
![alt text](pca2.png) <br />
