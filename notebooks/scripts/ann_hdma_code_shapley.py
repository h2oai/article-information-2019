import numpy as np
import pandas as pd
import shap
import subprocess
import sys

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

out_dir = "ann_output5/"


def ann_model():
	""" Create an artificial neural network model """

    input = Input(shape=(features,), name='main_input')

    out = Dense(20, input_dim=10, activation='relu')(input)
    
    out = Dense(20, input_dim=20, activation='relu')(out)
    
    out = Dense(1, activation='sigmoid')(out)
    
    model = Model(inputs=input, outputs=out)
	# Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    return model


# Load the dataset
xnn_data_dir = '~/article-information-2019/data/xnn_output/'
#xnn_data_dir = ''
DATA=pd.read_csv(xnn_data_dir + 'train_transformed.csv')
#DATA = DATA.iloc[0:10000,:]
TEST=pd.read_csv(xnn_data_dir + 'test_transformed.csv')


# Select features and split into target and feature sets
selected_vars = ['term_360', 'conforming']
selected_vars += ['debt_to_income_ratio_missing','loan_amount_std', 'loan_to_value_ratio_std']
selected_vars += ['no_intro_rate_period_std', 'intro_rate_period_std']
selected_vars += ['property_value_std', 'income_std', 'debt_to_income_ratio_std']

target_var = "high_priced"


# Split the data into feature and target values
X=DATA[selected_vars].values
Y=DATA[target_var].values
TEST_X = TEST[selected_vars].values
TEST_Y = TEST[target_var].values
features = X.shape[1]

inputs = {'main_input': X}

#Fit model
model = ann_model()
model.fit(inputs, Y, epochs=5000, batch_size=1024, validation_split=0, verbose=1)


# Find the predictions and Shapley values on the test set.
bg_samples = 1000
background = DATA[selected_vars].iloc[np.random.choice(DATA[selected_vars].shape[0], bg_samples, replace=False)]
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(TEST_X)
preds = model.predict(TEST_X)
preds = np.concatenate((preds, shap_values[0], preds), axis=1)
preds[:, -1] = explainer.expected_value


# Add the Shapley values and predictions to the dataset
TEST = pd.DataFrame(pd.concat([TEST, pd.DataFrame(preds)], axis=1))

Feature_names = selected_vars.copy()

TEST = TEST.rename(columns={0: "probability", 
                            1: Feature_names[0]+"_Shapley_score",
                            2: Feature_names[1]+"_Shapley_score",
                            3: Feature_names[2]+"_Shapley_score",
                            4: Feature_names[3]+"_Shapley_score",
                            5: Feature_names[4]+"_Shapley_score",
                            6: Feature_names[5]+"_Shapley_score", 
                            7: Feature_names[6]+"_Shapley_score",
                            8: Feature_names[7]+"_Shapley_score",
                            9: Feature_names[8]+"_Shapley_score",
                            10: Feature_names[9]+"_Shapley_score",
                            11: "Intercept_Shapley_score"})

    
# Save results
TEST.to_csv(out_dir + "hmda_ann_results_with_Shapley.csv" , index=False)


