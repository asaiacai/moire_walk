import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model

def classification_on_real(dx, steps=50,fbm=False):
    N=np.shape(dx)[0]
    net_file = 'models/{}_new_model.h5'.format(steps)
    model = load_model(net_file)
    
    if fbm:
        fbm_model = load_model('models/{}_fbm_alpha.h5'.format(steps))
    
    predictions = []
    values = []
    for j in range(N):
        dummy = np.zeros((1,steps-1,1))
        dummy[0,:,:] = np.reshape(dx[j,:], (steps-1, 1))
        y_pred = model.predict(dummy) # get the results for 1D 

        ymean = np.mean(y_pred,axis=0) # calculate mean prediction of N-dimensional trajectory 
        values.append(ymean.round(decimals=2))
        prediction = np.argmax(ymean,axis=0) # translate to classification
        predictions.append((j, prediction))
       
        if fbm and prediction == 0:
            fbm_alpha_pred = fbm_model.predict(dummy)
    return values, predictions

def get_activations(dx, steps=50,fbm=False):
    N=np.shape(dx)[0]
    net_file = 'models/{}_new_model.h5'.format(steps)
    model = load_model(net_file)

    layer_name = 'concatenate_1'
    intermediate_layer_model = keras.Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    
    activations = []
    values = []
    for j in range(N):
        dummy = np.zeros((1,steps-1,1))
        dummy[0,:,:] = np.reshape(dx[j,:], (steps-1, 1))
        
        activations.append(intermediate_layer_model.predict(dummy)) # get the results for 1D
        
    return activations
def generate_dx(x):
    temp_x = x-np.mean(x)
    dx = np.diff(temp_x)
    dx = dx/np.std(dx)
    return dx