#import necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

class LeNEt:
    @staticmethod
    def build(width,height, depth,classes):
        #initialize the model
        model=Sequential()
        inputShape=(height,width,depth)
        
        #update the input shape
        if K.image_data_format()=="channels_first":
            inputShape=(depth,height,width)
        
        #implement the first set of conv=> relu=>pool
        model.add(Conv2D(20,(5,5),padding="same",input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        
        #second set of conv=>relu=> pool layers
        model.add(Conv2D(50,(5,5),padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        
        #the set o FC=>Relu layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        
        #Softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        #return the constructed network architecture
        return model
        
        
        
        
                  
            
            
        