#!/usr/bin/env python
# coding: utf-8

# In[19]:

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# Now load the appropriate tensorflow packages.

# In[20]:


from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense


# We also load some standard packages.

# In[21]:


import numpy as np
import matplotlib.pyplot as plt


# Clear the Keras session.

# In[22]:


import tensorflow as tf
tf.keras.backend.clear_session()


# In[23]:


nrow = 150
ncol = 150


# In[24]:


# Load the VGG16 network
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape = (nrow, ncol, 3))


# In[25]:


# Create a new model
model = Sequential()

# Loop over base_model.layers and add each layer to model
for layer in base_model.layers:
    model.add(layer)


# In[26]:


for layer in model.layers:
    layer.trainable = False


# In[27]:


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(12, activation = "sigmoid"))


# In[28]:


model.summary()


# In[36]:


train_data_dir = r'C:\Users\hans\Dropbox\pythonProjects\cnn-keras\new_dataset\train'
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
                        train_data_dir,
                        target_size=(nrow,ncol),
                        batch_size=batch_size,
                        class_mode='categorical')


# In[37]:


test_data_dir = r'C:\Users\hans\Dropbox\pythonProjects\cnn-keras\new_dataset\train'
batch_size = 32
test_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_generator = train_datagen.flow_from_directory(
                        test_data_dir,
                        target_size=(nrow,ncol),
                        batch_size=batch_size,
                        class_mode='categorical')


# In[38]:


# Display the image
def disp_image(im):
    if (len(im.shape) == 2):
        # Gray scale image
        plt.imshow(im, cmap='gray')    
    else:
        # Color image.  
        im1 = (im-np.min(im))/(np.max(im)-np.min(im))*255
        im1 = im1.astype(np.uint8)
        plt.imshow(im1)    
        
    # Remove axis ticks
    plt.xticks([])
    plt.yticks([])


# In[39]:


train_generator.reset()
X,y = train_generator.next()
plt.figure(figsize=(20,20))
nplot = 8
for i in range(nplot):
    plt.subplot(1,nplot,i+1)
    plt.title('y=%s' % y[i])
    disp_image(X[i])


# In[40]:


# Compile
from tensorflow.keras.optimizers import Adam
opt = Adam(lr=0.1e-3)
hist = model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])


# In[41]:


train_size = train_generator.n
test_size = test_generator.n

steps_per_epoch = train_size // batch_size
validation_steps =  test_size // batch_size


# In[ ]:


nepochs = 42 # Number of epochs

# Call the fit_generator function
hist = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=nepochs,
    validation_data=test_generator,
    validation_steps=validation_steps)


# In[45]:


import pickle


# In[46]:


# Save model
mod_name = 'basic'
h5_fn = ('cifar_%s.h5' % mod_name)
model.save(h5_fn)
print('Model saved as %s' % h5_fn)

# Save history
hist_fn = ('hist_%s.p' % mod_name)
with open(hist_fn, 'wb') as fp:
    hist_dict = hist.history
    pickle.dump(hist_dict, fp) 
print('History saved as %s' % hist_fn)    


# In[48]:


# Plot the training accuracy and validation accuracy curves on the same figure.

# TO DO
mod_name_plot = ['basic']
plt.figure(figsize=(10,5))
for iplt in range(2):
    
    plt.subplot(1,2,iplt+1)
    for i, mod_name in enumerate(mod_name_plot):

        # Load history
        hist_fn = ('hist_%s.p' % mod_name)
        with open(hist_fn, 'rb') as fp:        
            hist_dict = pickle.load(fp) 

        if iplt == 0:
            acc = hist_dict['acc'][:43]
        else:
            acc = hist_dict['val_acc'][:43]
        plt.plot(acc, '-', linewidth=3)
    
    n = len(acc)
    nepochs = len(acc)
    plt.grid()
    plt.xlim([0, nepochs])
    plt.legend(['baseline'])
    plt.xlabel('Epoch')
    if iplt == 0:
        plt.ylabel('Train accuracy')
    else:
        plt.ylabel('Test accuracy')
        
plt.tight_layout()

