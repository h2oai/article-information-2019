
import numpy as np
import pandas as pd
import shap

import keras
from keras import backend
from keras.layers import Activation, Add, Dense, Dropout, Input, Lambda, Concatenate
from keras.models import Model

#import plotly.plotly as py
#import plotly.tools as tls
#import matplotlib.pyplot as plt


import subprocess
import sys

def install(library):
    subprocess.call([sys.executable, "-m", "pip", "install", library])

try:
    import pydot
except ImportError:
    install('pydot')
    import pydot
    

from timeit import default_timer as timer
import tensorflow as tf
from keras import backend as K


# Output file label
lll="quick_info_first_"


seed = 12345

np.random.seed(seed)

my_init = keras.initializers.RandomUniform(seed=seed)


def projection_initializer(shape, dtype=None):
    print(shape)
    inps = shape[0]
    subs = shape[1]
    if subs > pow(inps, 2) - 1:
        raise ValueError("Currently we support only up to 2^features - 1 number of subnetworks.")
    
    weights = []
    # TODO impl when subs > inps
    for i in range(subs):
        w = [0] * inps
        w[i] = 1
        weights.append(w)
    return weights



class XNN:
    # define base model
    def __init__(self, features, ridge_functions=3, arch=[20,12], bg_samples=100, seed=None, is_categorical=False):
        self.seed = seed
        self.bg_samples = bg_samples
        self.is_categorical = is_categorical
        
        #
        # Prepare model architecture
        #
        # Input to the network, our observation containing all the features
        input = Input(shape=(features,), name='main_input')

        # Input to ridge function number i is the dot product of our original input vector times coefficients
        ridge_input = Dense(ridge_functions,
                            name="projection_layer",
                                activation='linear')(input)
        
        self.ridge_networks = []
        # Each subnetwork uses only 1 neuron from the projection layer as input so we need to split it
        ridge_inputs = Lambda( lambda x: tf.split(x, ridge_functions, 1), name='lambda_1' )(ridge_input)
        for i, ridge_input in enumerate(ridge_inputs):
            # Generate subnetwork i
            mlp = self._mlp(ridge_input, i, arch)
            self.ridge_networks.append(mlp)
            
        # Add results from all subnetworks
        # out = Add(name='main_output')(self.ridge_networks)
        
        added = Concatenate(name='concatenate_1')(self.ridge_networks)
        if self.is_categorical:
            out = Dense(1, activation='sigmoid', input_shape= (ridge_functions, ), name='main_output')(added)
        else:
            out = Dense(1, activation='linear', input_shape= (ridge_functions, ), name='main_output')(added)
        

        
        self.model = Model(inputs=input, outputs=out)
        
        #optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        
        optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        
        if self.is_categorical:
            self.model.compile(loss={'main_output': 'binary_crossentropy'}, optimizer=optimizer)
        else:
            self.model.compile(loss={'main_output': 'mean_squared_error'}, optimizer=optimizer)

        self.explainer = None

        #self.explainer2 = None
                
        
    def _mlp(self, input, idx, arch=[20,12], activation='relu'):
        if len(arch) < 1:
            return #raise exception
        
        # Hidden layers
        mlp = Dense(arch[0], activation=activation, name='mlp_{}_dense_0'.format(idx), kernel_initializer=my_init)(input)
        for i, layer in enumerate(arch[1:]):
            mlp = Dense(layer, activation=activation, name='mlp_{}_dense_{}'.format(idx, i+1), kernel_initializer=my_init)(mlp)
         
        init = keras.initializers.RandomUniform(minval=-5, maxval=5, seed=None)
        # Output of the MLP

        mlp = Dense(1, 
                    activation='linear', 
                    name='mlp_{}_dense_last'.format(idx), 
                    kernel_regularizer=keras.regularizers.l1(1e-3),
                    kernel_initializer=my_init)(mlp)
        
        return mlp
    
    def print_architecture(self):
        self.model.summary()
    
    def fit(self, X, y, epochs=5, batch_size=128, validation_split=0.0, verbose=0):
        inputs = {'main_input': X}

        self.model.fit(inputs, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose)
        
        #
        # Prepare the explainer
        # 
        np.random.seed(self.seed)
        if isinstance(X, pd.DataFrame):
            background = X.iloc[np.random.choice(X.shape[0], self.bg_samples, replace=False)]
        else:
            background = X[np.random.choice(X.shape[0], self.bg_samples, replace=False)]

        # Explain predictions of the model on the subset
        self.explainer = shap.DeepExplainer(self.model, background)
        
        #intermediate_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer("concatenate_1").output)
        #int_background = intermediate_layer_model.predict(background)
        #self.explainer2 = shap.DeepExplainer((self.model.layers[-2].input, self.model.layers[-1].output), int_background)
            
        
    def predict(self, X, pred_contribs=False):
        pred_start = timer()
        preds = self.model.predict(X)
        pred_end = timer()
        print("Predictions took {}".format(pred_end - pred_start))

        if pred_contribs:
            explainer_start = timer()
            print("Find shap 2 ways")
            self.shap_values = self.explainer.shap_values(X)

            explainer_end = timer()
            print("Explainer took {}".format(explainer_end - explainer_start))

            concat_start = timer()
            #preds_old = preds.copy()
            preds = np.concatenate((preds, self.shap_values[0], preds), axis=1)
            preds[:,-1] = self.explainer.expected_value
            #preds = np.concatenate((preds, self.shap_values2[0], preds_old), axis=1)
            #preds[:,-1] = self.explainer2.expected_value
            concat_end = timer()
            print("Concat took {}".format(concat_end - concat_start))
        return preds
    
    def plot_shap(self, X):
        shap.summary_plot(self.shap_values, X)
        #shap.summary_plot(self.shap_values2, X)
        
        
def alpha_beta(alpha, beta, X , R):
        
    positive_values = [item for item in X if item > 0]
        
    negative_values = [item for item in X if item < 0] 
        
    ans = np.array([0.0]*len(X))
        
    if len(positive_values)>0:
           
        ans += alpha*np.array([item / float(sum(positive_values)) if item > 0 else 0 for item in X])

    if len(negative_values)>0:
 
        ans += -beta * np.array([item / float(sum(negative_values)) if item < 0 else 0 for item in X]) 

    return ans*R





def deep_lift(X_bar, X , R):
    """ Deep lift calculation for xnn """   
    ans =  np.array(X) - np.array(X_bar)
    ans = ans / (sum(X) - sum(X_bar))     
    return ans*R





import math



#DATA_full=pd.read_csv('~/Documents/kaggle/champs/output/train2.csv')
#DATA_full=pd.read_csv('data_dir/UCI_Credit_Card.csv')

#DATA = DATA_full.iloc[0:int(.7*len(DATA_full)),].copy()
#TEST = DATA_full.iloc[int(.7*len(DATA_full)):,].copy()

#DATA = DATA_full.iloc[0:3000,].copy()
#TEST = DATA_full.iloc[3000:6000,].copy()

DATA=pd.read_csv('data_dir/train_transformed.csv')
#DATA = DATA.iloc[0:10000,:]
TEST=pd.read_csv('data_dir/test_transformed.csv')


#selected_vars = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4"]                   
#selected_vars += ["PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1"]
#selected_vars += ["PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
#selected_vars = ["PAY_0_std", "PAY_2_std",  "BILL_AMT1_std", "LIMIT_BAL_std",  "PAY_AMT1_std",  "PAY_AMT2_std"]
#selected_vars = ["LIMIT_BAL_std", "SEX_std", "EDUCATION_std", "MARRIAGE_std", "AGE_std", "PAY_0_std", "PAY_2_std", "PAY_3_std", "PAY_4_std"]                   
#selected_vars += ["PAY_5_std", "PAY_6_std", "BILL_AMT1_std", "BILL_AMT2_std", "BILL_AMT3_std", "BILL_AMT4_std", "BILL_AMT5_std", "BILL_AMT6_std", "PAY_AMT1_std"]
#selected_vars += ["PAY_AMT2_std", "PAY_AMT3_std", "PAY_AMT4_std", "PAY_AMT5_std", "PAY_AMT6_std"]

selected_vars = ['loan_to_value_ratio_std', 'property_value_std', 'loan_amount_std']
selected_vars += ['income_std', 'discount_points_std', 'intro_rate_period_std']
selected_vars += ['lender_credits_std', 'loan_term_std']

target_var = "high_priced"

X=DATA[selected_vars].values
Y=DATA[target_var].values
TEST_X = TEST[selected_vars].values
TEST_Y = TEST[target_var].values
features = X.shape[1]





"""
plt.plot(dataframe['x1'], f1(dataframe['x1']), 'o', label="f1(·)")
plt.plot(dataframe['x2'], f2(dataframe['x2']), 'o', label="f2(·)")
plt.plot(dataframe['x3'], f3(dataframe['x3']), 'o', label="f3(·)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend(loc='lower right')
plt.show()
"""


from keras.utils import plot_model

# Initialize the XNN
is_cat = True
xnn = XNN(features=features, ridge_functions=features,arch=[20, 12], is_categorical= is_cat)
print(features)
#xnn = XNN(features=features, ridge_functions=3,arch=[20,12,8])
#plot_model(xnn.model, to_file='model_regression.png')
xnn.print_architecture()


#xnn.fit(X, Y, epochs=3000, batch_size=64, validation_split=0.25, verbose=0)
xnn.fit(X, Y, epochs=10000, batch_size=32, validation_split=0.25, verbose=1)
#xnn.fit(X, Y, epochs=6, batch_size=32, validation_split=0.25, verbose=0)


# Print projection layers
x = list(map(lambda x: 'x' + str(x+1), range(features)))

intermediate_output = []

# Plot the projection weights
weight_list = []
for layer in xnn.model.layers:
    if "projection_layer" in layer.get_config()['name']:
        #intermediate_layer_model = Model(inputs=xnn.model.input, outputs=xnn.model.layer.output)
        #intermediate_output.append(intermediate_layer_model.predict(X))
        
        print(layer.get_config()['name'])
        
        # Transpose to get the right projection dimension
        weights = [np.transpose(layer.get_weights()[0])]
        weights2 = layer.get_weights()[1]
        wp=[]
        for i, weight in enumerate(weights[0]):
            weight_list.append(weight)
            wp.append(list(np.reshape(weight, (1,features))[0]))
        
            print(weight)
            """
            plt.bar(x, abs(np.reshape(weight, (1,features))[0]), 1, color="blue")
            plt.xlabel("Subnetowork {} coefficient".format(i))
            plt.ylabel("Weight value")
            plt.show()
            """
    if "main_output" in layer.get_config()['name']:
        weights_main = layer.get_weights()
        print(weights_main)
        
pd.DataFrame(wp).to_csv("wp_"+lll+".csv", index=False)


# Record output layers

import scipy as sp


int_output = {}
int_output2 = {}
int_weights = {}
int_bias = {}
int_input = {}

original_activations = {}

for layer in xnn.model.layers:
    
    layer_name = layer.get_config()['name']
    if layer_name != "main_input":
        print(layer_name)
        weights = layer.get_weights()
        #bias = layer.get_weights()[1]
        try:
            bias = layer.get_weights()[1]
            int_bias[layer_name] = bias
        except:
            print("No Bias")
        
        intermediate_layer_model = Model(inputs=xnn.model.input, outputs=xnn.model.get_layer(layer_name).output)
        if is_cat:
            int_output[layer_name] = sp.special.expit(intermediate_layer_model.predict(TEST_X))
            int_output[layer_name + "_p"] = intermediate_layer_model.predict(TEST_X)
        else:
            int_output[layer_name] = intermediate_layer_model.predict(TEST_X)
        
        if is_cat:
            original_activations[layer_name] = sp.special.expit(intermediate_layer_model.predict(X))   
            original_activations[layer_name + "_p"] = intermediate_layer_model.predict(X)
        else:
            original_activations[layer_name] = intermediate_layer_model.predict(X)        
        #intermediate_layer_model = Model(inputs=xnn.model.input,
        #                             outputs=xnn.model.get_layer('mlp_'+str(feature_num)+'_dense_last').output)
        #intermediate_output.append(intermediate_layer_model.predict(X))
        int_weights[layer_name] = weights
        int_input[layer_name] = layer.input
        int_output2[layer_name] = layer.output
        
        
        
        
        
        
# Calculate importance      
item = 0
int_weights["main_output"][0][0][0]
int_output["concatenate_1"][item][0]
overall_output = int_output["main_output"][item][0]

#deep_lift(X_bar, X , R)

feature_output = []
feature_output2 = []
feature_output3 = []

S_bar = sum(original_activations["main_output"])/len(original_activations["main_output"])
# original_activations[layer_name]
output_weights = np.array([int_weights["main_output"][0][ii][0] for ii in range(features)])
output_Z_bar = sum(original_activations["concatenate_1"]*output_weights)/len(original_activations["concatenate_1"])


input_Z_bar = {}
for ridge_num in range(features):   
    input_weights = np.array([int_weights["projection_layer"][0][ii][ridge_num] for ii in range(features)])
    input_Z_bar[ridge_num] = sum(X*input_weights)/len(X)
    
for test_num in range(len(TEST_X)):
    activation_list=[int_weights["main_output"][0][ii][0]*int_output["concatenate_1"][test_num][ii] for ii in range(features)]
    
    # For classification, change this to the inverse sigmoid of the output
    features_ab = alpha_beta(2, 1, activation_list , int_output["main_output"][test_num][0])
    features_ab2 = alpha_beta(2, 1, activation_list , int_output["main_output"][test_num][0]-S_bar)
    
    features_dl = deep_lift(output_Z_bar, activation_list , int_output["main_output"][test_num][0]-S_bar)
      
        
    input_scores = []
    input_scores_dl = []
    input_scores2 = []
    input_scores_dl2 = []
    for ridge_num in range(features):
        weights = int_weights["projection_layer"][0][ridge_num]
        output = TEST_X[test_num,:]
        
        # [int_weights["projection_layer"][0][ii][0] for ii in range(features)]
        
        act = TEST_X[test_num,:]*np.array([int_weights["projection_layer"][0][ii][ridge_num] for ii in range(features)])
    
        # Input relevance scores for a single ridge function
        input_scores += list(alpha_beta(2,1, act, features_ab[ridge_num]))
        input_scores_dl += list(deep_lift(input_Z_bar[ridge_num], act, features_dl[ridge_num]))
        input_scores2 += list(alpha_beta(2,1, act, features_ab2[ridge_num]))

        # print(sum(TEST_X[0,:]*np.array([int_weights["projection_layer"][0][ii][0] for ii in range(features)]))+int_bias["projection_layer"][0])
        
    input_sum = [sum(input_scores[ii+features*jj] for jj in range(features)) for ii in range(features)] 
    input_sum2 = [sum(input_scores2[ii+features*jj] for jj in range(features)) for ii in range(features)] 
    input_sum_dl = [sum(input_scores_dl[ii+features*jj] for jj in range(features)) for ii in range(features)] 
    input_abs_sum = [sum(abs(input_scores[ii+features*jj]) for jj in range(features)) for ii in range(features)] 
    feature_output.append(input_sum+input_abs_sum+[int_output["main_output"][test_num][0]]+list(features_ab)+input_scores)
    feature_output2.append(input_sum+list(features_ab)+input_sum_dl + list(features_dl))
    feature_output3.append(input_sum2+list(features_ab2)+input_sum_dl + list(features_dl))




# Plot shapes
intermediate_output = []


for feature_num in range(features):
    intermediate_layer_model = Model(inputs=xnn.model.input,
                                 outputs=xnn.model.get_layer('mlp_'+str(feature_num)+'_dense_last').output)
    intermediate_output.append(intermediate_layer_model.predict(X))



ridge_x = []
ridge_y = []
for weight_number in range(len(weight_list)):
    
    ridge_x.append(list(sum(X[:, ii]*weight_list[weight_number][ii] for ii in range(features))))
    ridge_y.append(list(intermediate_output[weight_number]))
    """
    plt.plot(sum(X[:, ii]*weight_list[weight_number][ii] for ii in range(features)), intermediate_output[weight_number], 'o')
    plt.xlabel("x")
    plt.ylabel("Subnetwork " + str(weight_number))
    plt.legend(loc='lower right')
    plt.show()
    """

pd.DataFrame(ridge_x).to_csv("ridge_x_"+lll+".csv", index=False)
pd.DataFrame(ridge_y).to_csv("ridge_y_"+lll+".csv", index=False)     
pd.DataFrame(feature_output2).to_csv("feature_output2_"+lll+".csv", index=False)
pd.DataFrame(feature_output3).to_csv("feature_output3_"+lll+".csv", index=False)       
# Run predictions on the XNN model and retrieve contributions of each feature
# First column contains model predictions, last one DeepSHAP bias, everything in between feature contributions
preds = xnn.predict(TEST_X, pred_contribs=True)

preds[0:5,:]

pd.DataFrame(preds).to_csv("preds_"+lll+".csv", index=False)
pd.DataFrame(TEST).to_csv("TEST_"+lll+".csv", index=False)

shap.initjs()
shap.summary_plot(xnn.shap_values, X)








y=xnn.shap_values
ind=1
print(y[0][ind])


print(feature_output2[ind])

#feature_output.append(input_sum+input_abs_sum+[int_output["main_output"][test_num][0]]+list(features_ab)+input_scores)

layerwise_average_input=np.array([0.0]*features)
layerwise_average_input2=np.array([0.0]*features)
layerwise_average_ridge=np.array([0.0]*features)
layerwise_average_ridge2=np.array([0.0]*features)
layerwise_average_shap=np.array([0.0]*features)
lift_average_input=np.array([0.0]*features)
lift_average_ridge=np.array([0.0]*features)

for ii in range(len(feature_output2)):
    layerwise_average_input += np.array(feature_output2[ii][0:features])
    layerwise_average_ridge += np.array(feature_output2[ii][features:(2*features)])
    layerwise_average_input2 += np.array(feature_output3[ii][0:features])
    layerwise_average_ridge2 += np.array(feature_output3[ii][features:(2*features)])
    lift_average_input += np.array(feature_output2[ii][(2*features):(3*features)])
    lift_average_ridge += np.array(feature_output2[ii][(3*features):(4*features)])
    layerwise_average_shap += np.array(y[0][ii])
     
layerwise_average_input = layerwise_average_input/len(feature_output2)
layerwise_average_ridge = layerwise_average_ridge/len(feature_output2)
layerwise_average_input2 = layerwise_average_input2/len(feature_output2)
layerwise_average_ridge2 = layerwise_average_ridge2/len(feature_output2)
layerwise_average_shap = layerwise_average_shap/len(feature_output2)
lift_average_input = lift_average_input/len(feature_output2)
lift_average_ridge = lift_average_ridge/len(feature_output2)


SCORES = [list(layerwise_average_input), list(layerwise_average_ridge),
          list(layerwise_average_input2), list(layerwise_average_ridge2),
          list(layerwise_average_shap), list(lift_average_input),
          list(lift_average_ridge)]

 
pd.DataFrame(SCORES).to_csv("scores_"+lll+".csv", index=False)
"""
plt.bar(x, abs(np.reshape(y[0][ind], (1,features))[0]), 1, color="blue")
plt.xlabel("Shap Score Example " + str(ind))
plt.ylabel("")
plt.show()

plt.bar(x, abs(np.reshape(feature_output2[ind][0:features], (1,features))[0]), 1, color="blue")
plt.xlabel("Input Layerwise Propagation Score Example " + str(ind))
plt.ylabel("")
plt.show()

plt.bar(x, abs(np.reshape(feature_output2[ind][features:(2*features)], (1,features))[0]), 1, color="blue")
plt.xlabel("Ridge Layerwise Propagation Score Example " + str(ind))
plt.ylabel("Weight value")
plt.show()

plt.bar(x, abs(np.reshape(feature_output2[ind][2*features:(3*features)], (1,features))[0]), 1, color="blue")
plt.xlabel("Deep Lift Input Score Example " + str(ind))
plt.ylabel("Weight value")
plt.show()


plt.bar(x, abs(np.reshape(feature_output2[ind][3*features:(4*features)], (1,features))[0]), 1, color="blue")
plt.xlabel("Deep Lift Ridge Score Example " + str(ind))
plt.ylabel("Weight value")
plt.show()
      
plt.bar(x, abs(np.reshape(layerwise_average_input, (1,features))[0]), 1, color="blue")
plt.xlabel("Input Layerwise Propagation Score Average")
plt.ylabel("")
plt.show()

plt.bar(x, abs(np.reshape(layerwise_average_ridge, (1,features))[0]), 1, color="blue")
plt.xlabel("Ridge Layerwise Propagation Score Average")
plt.ylabel("Weight value")
plt.show()


plt.bar(x, abs(np.reshape(layerwise_average_input2, (1,features))[0]), 1, color="blue")
plt.xlabel("Input Layerwise Propagation Score Average 2")
plt.ylabel("")
plt.show()

plt.bar(x, abs(np.reshape(layerwise_average_ridge2, (1,features))[0]), 1, color="blue")
plt.xlabel("Ridge Layerwise Propagation Score Average 2")
plt.ylabel("Weight value")
plt.show()


plt.bar(x, abs(np.reshape(lift_average_input, (1,features))[0]), 1, color="blue")
plt.xlabel("Input Lift Score Average")
plt.ylabel("")
plt.show()

plt.bar(x, abs(np.reshape(lift_average_ridge, (1,features))[0]), 1, color="blue")
plt.xlabel("Ridge Lift Score Average")
plt.ylabel("Weight value")
plt.show()

plt.bar(x, abs(np.reshape(layerwise_average_shap, (1,features))[0]), 1, color="blue")
plt.xlabel("Shapley Score Average")
plt.ylabel("Weight value")
plt.show()
"""





"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
#sklearn.tree.DecisionTreeRegressor

reg = LinearRegression().fit(X, Y)
reg.score(X, Y)

pred_linear = reg.predict(TEST_X)


MSE = sum(abs(np.array(TEST_Y)-np.array(pred_linear)))/len(TEST_Y)

reg = DecisionTreeRegressor().fit(X, Y)
reg.score(X, Y)

pred_tree = reg.predict(TEST_X)

MSE_tree = sum(abs(np.array(TEST_Y)-np.array(pred_tree)))/len(TEST_Y)

preds = xnn.predict(TEST_X)

MSE_xnn = sum(abs(np.array(TEST_Y.flatten())-np.array(preds.flatten())))/len(TEST_Y)
"""








