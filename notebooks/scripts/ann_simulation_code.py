import numpy as np
import pandas as pd
import shap
import subprocess
import sys
#import pydot


import keras    
from timeit import default_timer as timer
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.layers import Dense


seed = 12345

np.random.seed(seed)

my_init = keras.initializers.RandomUniform(seed=seed)

#from keras.models import Sequential
#from keras.layers import Dense


out_dir = "ann_output2/"

# baseline model
def ann_model():
    
    input = Input(shape=(features,), name='main_input')

    out = Dense(10, input_dim=10, activation='relu')(input)
    
    out = Dense(5, input_dim=10, activation='relu')(out)
    
    out = Dense(1, activation='sigmoid')(out)
    
    model = Model(inputs=input, outputs=out)
	# Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    return model

# Load the dataset
xnn_data_dir = '~/article-information-2019/data/xnn_output/'
#xnn_data_dir = ''
DATA=pd.read_csv(xnn_data_dir + 'train_simulated_transformed.csv')
print(list(DATA.columns))
#DATA = DATA.iloc[0:10000,:]
TEST=pd.read_csv(xnn_data_dir + 'test_simulated_transformed.csv')
print(list(TEST.columns))


selected_vars = ['binary1', 'binary2', 'cat1_0', 'cat1_1', 'cat1_2', 'cat1_3', 'cat1_4', 
                 'fried1_std', 'fried2_std', 'fried3_std', 'fried4_std', 'fried5_std']

target_var = 'outcome'

X=DATA[selected_vars].values
Y=DATA[target_var].values
TEST_X = TEST[selected_vars].values
TEST_Y = TEST[target_var].values
features = X.shape[1]


inputs = {'main_input': X}

# Fit model
model = ann_model()
model.fit(inputs, Y, epochs=2000, batch_size=1024, validation_split=0, verbose=1)

# Save results
test_preds = model.predict(TEST_X)
pd.DataFrame(pd.concat([TEST, pd.DataFrame(test_preds)], axis=1)).to_csv(out_dir + "simulated_ann_results.csv" , index=False)


