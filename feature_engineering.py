#Import libraries needed for the feature engineering functions
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

#Feature fields and target field
features_fields = ['School Type', '% First Language Not English', '% English Language Learner', '% Students With Disabilities', '% High Needs', '% Economically Disadvantaged', 'Average Class Size', 'Average Salary', 'Total Pupil FTEs',
            'Average In-District Expenditures per Pupil', 'SAT Score']
target_field = 'MCAS Total CPI'

#Function to execute all the feature engineering methods
def executeFeatureEng(dataset):

    #Create target column
    dataset[target_field] = targetVar(dataset)
    
    #Replace missing demographic data with mean 
    mean_replacement = ['Average Class Size', 'Average Salary']
    for col in mean_replacement:
        dataset[col] = fillWithMean(dataset, col)

    #Replace missing demographic data with median 
    median_replacement = ['Average In-District Expenditures per Pupil', 'Total Pupil FTEs']
    for col in median_replacement:
        dataset[col] = fillWithMedian(dataset, col)

    #Impute missing SAT Scores
    dataset['SAT Score'] = satImpute(dataset)

    #drop rows missing the target variable
    cut_set = dropNARows(dataset, [target_field])

    #Cut down to only feature and target fields
    features_and_target = cut_set[features_fields + [target_field]]

    #Define and normalize numerical columns
    num_cols = ['% First Language Not English', '% English Language Learner', '% Students With Disabilities', '% High Needs', '% Economically Disadvantaged', 'Average Class Size', 'Average Salary', 'Total Pupil FTEs',
            'Average In-District Expenditures per Pupil', 'SAT Score']
    features_and_target[num_cols] = normalizeNum(features_and_target[num_cols])

    #OneHot Encode the categorical columns
    cat_cols = ['School Type']
    features_and_target = pd.concat([features_and_target.drop(cat_cols, axis = 1), oneHot(features_and_target[cat_cols])], axis = 1)
    features_and_target.to_csv('Test Feature Output.csv')

    #Convert Features and Target to Numpy Arrays
    features = features_and_target.drop(target_field, axis = 1).to_numpy()
    target = features_and_target[target_field].to_numpy()

    return features, target


#Function to create target variable column
def targetVar(dataset):

    cpi_target = (dataset['MCAS_10thGrade_Math_CPI'] + dataset['MCAS_10thGrade_English_CPI']) / 2

    return cpi_target

#Function to drop rows that are missing critical information including the target variable, and demographic feautures about the school
def dropNARows(dataset, crit_columns):

    dropped_set = dataset.copy()

    for col in crit_columns:
        dropped_set.dropna(subset = [col], inplace = True)

    #Reset the index before returning
    dropped_set.reset_index(inplace = True)

    return dropped_set

#Function to replace missing values of a column with the mean value by district and the mean value by state for districts without complete information
def fillWithMean(frame, col_name):

    #copy the dataframe
    func_frame = frame.copy()

    #get the list of unique districts 
    dist_avgs = {}
    districts = func_frame['District Name'].unique()

    #run through each district and calculate the mean value of the requested column and add to dict, for districts without an average, use the state average value
    state_avg = func_frame[col_name].mean()
    for d in districts:
        d_avg = func_frame[func_frame['District Name'] == d][col_name].mean()
        if np.isnan(d_avg):
            dist_avgs[d] = state_avg
        else:
            dist_avgs[d] = d_avg
        
    #fill the missing values with district averages in the requested column
    func_frame[col_name] = func_frame[col_name].astype(str).replace('nan', np.nan)
    func_frame[col_name].fillna(func_frame['District Name'], inplace = True)
    func_frame[col_name] = func_frame[col_name].apply(lambda x: dist_avgs[x] if x in dist_avgs.keys() else x)

    #Convert target series into float type
    func_frame[col_name] = func_frame[col_name].astype('float')    

    #Return updated series
    return func_frame[col_name]

#Function to replace missing values of a column with the median value of each district
def fillWithMedian(frame, col_name):

    #copy the dataframe
    func_frame = frame.copy()

    #get the list of unique districts 
    dist_meds = {}
    districts = func_frame['District Name'].unique()

    #run through each district and calculate the mean value of the requested column and add to dict, for districts without an average, use the state average value
    state_med = func_frame[col_name].median()
    for d in districts:
        d_med = func_frame[func_frame['District Name'] == d][col_name].median()
        if np.isnan(d_med):
            dist_meds[d] = state_med
        else:
            dist_meds[d] = d_med
        
    #fill the missing values with district averages in the requested column
    func_frame[col_name] = func_frame[col_name].astype(str).replace('nan', np.nan)
    func_frame[col_name].fillna(func_frame['District Name'], inplace = True)
    func_frame[col_name] = func_frame[col_name].apply(lambda x: dist_meds[x] if x in dist_meds.keys() else x)

    #Convert target series into float type
    func_frame[col_name] = func_frame[col_name].astype('float')    

    #Return updated series
    return func_frame[col_name]


#Function to imputte SAT scores for missing values
def satImpute(frame):

    #Create copy of dataframe
    func_frame = frame.copy()

    #Create SAT aggregate score
    func_frame['SAT Score'] = func_frame['Average SAT_Reading'] + func_frame['Average SAT_Writing'] + func_frame['Average SAT_Math']
    sat_impute_frame = func_frame[['% First Language Not English', '% English Language Learner', '% Students With Disabilities' ,'% High Needs' ,'% Economically Disadvantaged', 'Average Class Size', 'Average Salary', 'Average In-District Expenditures per Pupil',
                              'SAT Score']]

    #Define scalar and normalize columns that will be used for imputation
    scaler = MinMaxScaler()
    feat_cols = sat_impute_frame[['% First Language Not English', '% English Language Learner', '% Students With Disabilities' ,'% High Needs' ,'% Economically Disadvantaged', 'Average Class Size', 'Average Salary', 'Average In-District Expenditures per Pupil']]
    norm_cols = scaler.fit_transform(feat_cols)
    norm_frame = pd.DataFrame(norm_cols, columns = ['% First Language Not English', '% English Language Learner', '% Students With Disabilities' ,'% High Needs' ,'% Economically Disadvantaged', 'Average Class Size', 'Average Salary', 'Average In-District Expenditures per Pupil'])

    #Add the sat column to the normalized dataframe
    sat_reset_frame = sat_impute_frame.reset_index()
    norm_frame['SAT Score'] = sat_reset_frame['SAT Score']

    #Train and apply KNN imputer
    norm_imputer = KNNImputer()
    norm_imputer.fit(norm_frame)
    norm_imputed_sat_array = pd.DataFrame(norm_imputer.transform(norm_frame))
    norm_imputed_series = pd.Series(norm_imputed_sat_array[8])

    #return imputed SAT score column
    return norm_imputed_series

#Function to normalize numerical column
def normalizeNum(data):

    #Define scalar and normalize columns that will be used for imputation
    scaler = MinMaxScaler()
    norm_cols = scaler.fit_transform(data)

    return norm_cols

#Function to one hot encode categorical columns
def oneHot(data):

    #Define one hot encoder and fit it to the data
    encoder = OneHotEncoder(sparse_output = False)
    encoded = encoder.fit_transform(data)
    encoded_frame = pd.DataFrame(encoded, columns = encoder.get_feature_names_out(data.columns))

    return encoded_frame