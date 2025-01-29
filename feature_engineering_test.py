import pandas as pd
import feature_engineering

#Load data file
data = pd.read_csv('data/MA_Public_Schools_2017.csv', low_memory = False)

init_data = data.copy()

#Test target variable function
data['Target'] = feature_engineering.targetVar(data)
#print(data['Target'].describe())

#Test the drop rows function
#print(data.info())
dropped = feature_engineering.dropNARows(data, ['Target', '% High Needs'])
#cprint(dropped.info())

#Test mean imputation function
mean_impute_cols = ['Average Class Size', 'Average Salary']
#for i in mean_impute_cols:
#    print(data[i].describe())

for i in mean_impute_cols:
    data[i] = feature_engineering.fillWithMean(data, i)

#for i in mean_impute_cols:
#    print(data[i].describe())

#Test median impute function
median_impute_cols = ['Average In-District Expenditures per Pupil']
#for i in median_impute_cols:
#    print(data[i].describe())

for i in median_impute_cols:
    data[i] = feature_engineering.fillWithMedian(data, i)

#for i in median_impute_cols:
#    print(data[i].describe())

#Test SAT Impute function
#print(data.info())
data['SAT Score'] = feature_engineering.satImpute(data)
#print(data.info())
#     print(data['SAT Score'].describe())

#Test the normalizing function
num_cols = ['% First Language Not English', '% English Language Learner', '% Students With Disabilities', '% High Needs', '% Economically Disadvantaged', 'Average Class Size', 'Average Salary', 'FTE Count',
            'Average In-District Expenditures per Pupil']
data[num_cols] = feature_engineering.normalizeNum(data[num_cols])
#for c in num_cols[:3]:
#    print(data[c].describe())

#Test the oneHot function
#print(data.info())
cat_cols = ['School Type']
data = pd.concat([data.drop(cat_cols, axis = 1), feature_engineering.oneHot(data[cat_cols])], axis = 1)
#print(data.info())

#Test full pipeline function
feat, tar = feature_engineering.executeFeatureEng(init_data)
print(len(feat))