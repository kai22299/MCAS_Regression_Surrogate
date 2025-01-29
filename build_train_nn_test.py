#Import the python file to be tested
import pandas as pd
import build_train_nn
import feature_engineering

#Test the model creation function
test_model = build_train_nn.buildRegressionNN(351, [300, 200, 100], 'linear')

#Load data needed for NN
data = pd.read_csv('data/MA_Public_Schools_2017.csv', low_memory = False)   aldksjaslkdf la;dskfj lsadkj ;lsadjf kdkdk
feat, tar = feature_engineering.executeFeatureEng(data)

#Test the data split function
X_train, y_train, X_test, y_test, X_val, y_val = build_train_nn.dataSplit(feat, tar, train_per = 0.7, test_per = 0.15, val_per = 0.15)