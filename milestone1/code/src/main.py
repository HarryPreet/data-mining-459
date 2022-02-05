import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import AutoLocator
from matplotlib.pyplot import hist
from shapely.geometry import Point
import geopandas as gpd
import seaborn as sns 
from geopandas import GeoDataFrame
from math import cos, asin, sqrt
from geopy.geocoders import Nominatim
from geopy import distance
import sys
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif as MIC, SelectKBest
import warnings
warnings.filterwarnings('ignore')

cases_train = pd.read_csv('../data/cases_2021_train.csv')
cases_test = pd.read_csv('../data/cases_2021_test.csv')
location = pd.read_csv('../data/location_2021.csv')

def main():
    global cases_train, cases_test, location
    section_one()
    section_four()
    section_five()
    section_six()
    section_seven()
#1.1

def section_one():
    print("Section 1: Intialising Outcome Group")
    global cases_train, cases_test, location
    cases_train.groupby('outcome').size()
    hospitalzed = ['Discharged', 'Discharged from hospital', 'Hospitalized', 'critical condition',
    'discharge', 'discharged']
    nonhospitalized = ['Alive', 'Receiving Treatment', 'Stable', 'Under treatment', 'recovering at home 03.03.2020', 'released from quarantine', 'stable', 'stable condition']
    deceased = ['Dead', 'Death', 'Deceased', 'Died', 'death', 'died']
    recovered = ['Recovered','recovered']
    cases_train.loc[cases_train["outcome"].isin(hospitalzed), "outcome_group"] ='hospitalized'
    cases_train.loc[cases_train["outcome"].isin(nonhospitalized), "outcome_group"] ='nonhospitalized'
    cases_train.loc[cases_train["outcome"].isin(deceased), "outcome_group"] ='deceased'
    cases_train.loc[cases_train["outcome"].isin(recovered), "outcome_group"] ='recovered'
    cases_train = cases_train.drop(columns="outcome")

#1.4

def section_four():
    global cases_train, cases_test, location
    print("Section 4: Cleaning Data")

    #Age
    ##Dropping entries with N/A Age Values
    cases_train = cases_train.dropna(subset=['age'])
    
    ## Reducing Age to a Standard Format
    cases_train = cases_train.dropna(subset=['age'])
    cases_train = cases_train.drop(cases_train[(cases_train['age'].str.len()>2)].index)
    test = cases_train.loc[(cases_train['age'].str.len()>2)]
    test['age'] = test['age'].apply(lambda x: int(float(x)))
    cases_train['age'] = cases_train['age'].apply(lambda x: int(float(x)))

    #Country 
    cases_train.loc[cases_train['country'].isna(),"country"] = "Taiwan"

    #Province 
    #print(cases_train.isna().sum())
    cases_train.loc[cases_train["province"].isna(), "province"] = cases_train['country']


    #Sex
    cases_train.loc[cases_train['sex'].isna(),"sex"] = cases_train['sex'].mode()[0]

    #Source
    cases_train.loc[cases_train['source'].isna(),"source"] = "None"

    #Additional Information
    cases_train.loc[cases_train['additional_information'].isna(),"additional_information"] = "None"

    #Date Confirmation
    cases_train.loc[cases_train['date_confirmation'].isna(),"date_confirmation"] = cases_train['date_confirmation'].mode()[0]
    
    cases_train['latitude'] = cases_train['latitude'].apply(lambda x: float(x))
    cases_train['longitude'] = cases_train['longitude'].apply(lambda x: float(x))
    
    
    #Age
    cases_test = cases_test.dropna(subset=['age'])
    cases_test = cases_test.drop(cases_test[(cases_test['age'].str.len()>3)].index)
    test = cases_test.loc[(cases_test['age'].str.len()>2)]
    test['age'] = test['age'].apply(lambda x: int(float(x)))
    cases_test.loc[(cases_test['age'].str.len()>3)] = test 
    

    #Reducing Age to a Standard Format
    test = cases_test.loc[(cases_test['age'].str.len()>4)]
    test['age'] = test['age'].str.split("-")
    test['age'] = test['age'].apply(lambda x: int(x[0]))
    cases_test.loc[(cases_test['age'].str.len()>4)] = test

    test = cases_test.loc[(cases_test['age'].str.len()>3)]
    test['age'] = test['age'].apply(lambda x: int(float(x)))
    cases_test.loc[(cases_test['age'].str.len()>3)] = test 



    #Date Confirmation
    cases_test.loc[cases_test['date_confirmation'].isna(),"date_confirmation"] = cases_test['date_confirmation'].mode()[0]

    #print(cases_test.isna().sum())
    #print(len(cases_test.index))

    #Sex
    cases_test.loc[cases_test['sex'].isna(),"sex"] = cases_test['sex'].mode()[0]

    #Source
    cases_test.loc[cases_test['source'].isna(),"source"] = "None"

    #Additional Information
    cases_test.loc[cases_test['additional_information'].isna(),"additional_information"] = "None"
    
    #Province
    cases_test = cases_test.drop(cases_test[cases_test['province'].isna()].index)

    #Country
    cases_test.loc[cases_test['country'].isna(),"country"] = "Taiwan"
   
    location['Incident_Rate'] = location['Incident_Rate'].fillna(location.groupby('Country_Region')['Incident_Rate'].transform('mean'))
    location['Case_Fatality_Ratio'] = location['Case_Fatality_Ratio'].fillna(location.groupby('Country_Region')['Case_Fatality_Ratio'].transform('mean'))
    location['Recovered'] = location['Recovered'].fillna(location.groupby('Country_Region')['Recovered'].transform('mean'))
    location['Active'] = location['Active'].fillna(location.groupby('Country_Region')['Active'].transform('mean'))
    location = location.drop(location[location['Lat'].isna()].index)
    location = location.drop(location[location['Long_'].isna()].index)
    location = location.drop(location[location['Case_Fatality_Ratio'].isna()].index)
  
    location.loc[location['Country_Region'] == "Korea, South", 'Country_Region'] = 'South Korea'
    location.loc[location['Country_Region'] == 'US', 'Country_Region'] = 'United States'
    location.loc[location['Country_Region'] == 'Taiwan*', 'Country_Region'] = 'Taiwan'
    location['Lat'] = location['Lat'].apply(lambda x: float(x))
    location['Long_'] = location['Long_'].apply(lambda x: float(x))
    
    
    cases_train.to_csv('../results/cases_2021_train_processed.csv')
    cases_test.to_csv('../results/cases_2021_test_processed.csv')
    location.to_csv('../results/location_2021_processed.csv')

    location_province_train()
    location_province_test()
    
def location_province_train():
    location = pd.read_csv('../results/location_2021_processed.csv')
    cases_train = pd.read_csv('../results/cases_2021_train_processed.csv')
    location= location.reset_index()
    cases_train = cases_train.reset_index()
    for i,x in location.iterrows():
        if(pd.isna(x['Province_State'])):
            pt1 = (x['Lat'],x['Long_'])
            trainTemp = cases_train.loc[cases_train['country'] == x['Country_Region']]
            if(len(trainTemp.index) > 0):
                min = sys.float_info.max
                for j,y in trainTemp.iterrows():
                    pt2 = (y['latitude'],y['longitude'])
                    if(distance(pt1[0],pt1[1],pt2[0],pt2[1])<min):
                        min = distance(pt1[0],pt1[1],pt2[0],pt2[1])
                        index = y
                        location.at[i,'Province_State'] = index['province']
                        location = location.drop(location[location['Province_State'].isna()].index)
    location.to_csv('../results/location_2021_processed.csv')

def location_province_test():
    location = pd.read_csv('../results/location_2021_processed.csv')
    cases_test = pd.read_csv('../results/cases_2021_test_processed.csv')
    location= location.reset_index()
    cases_test = cases_test.reset_index()
    for i,x in location.iterrows():
        if(pd.isna(x['Province_State'])):
            pt1 = (x['Lat'],x['Long_'])
            testTemp = cases_test.loc[cases_test['country'] == x['Country_Region']]
            if(len(testTemp.index) > 0):
                min = sys.float_info.max
                for j,y in testTemp.iterrows():
                    pt2 = (y['latitude'],y['longitude'])
                    if(distance(pt1[0],pt1[1],pt2[0],pt2[1])<min):
                        min = distance(pt1[0],pt1[1],pt2[0],pt2[1])
                        index = y
                        location.at[i,'Province_State'] = index['province']
                        location = location.drop(location[location['Province_State'].isna()].index)
    location.to_csv('../results/location_2021_processed.csv')

def section_five():
    print("Section 5: Detecting and Removing Outliers")
    cases_train = pd.read_csv('../results/cases_2021_train_processed.csv')
    cases_test = pd.read_csv('../results/cases_2021_test_processed.csv')
    #Uncomment to produce plots
    percentile25 = cases_train['age'].quantile(0.25)
    percentile75 = cases_train['age'].quantile(0.75)
    iqr = percentile75 - percentile25
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    cases_train[cases_train['age'] > upper_limit]
    cases_train[cases_train['age'] < lower_limit]
    train_trim = cases_train[cases_train['age'] < upper_limit]
    train_trim.shape
    #Uncomment to produce plots
    
    plt.figure(figsize=(16,8))
    plt.subplot(2,2,1)
    sns.distplot(cases_train['age'])
    plt.subplot(2,2,2)
    sns.boxplot(cases_train['age'])
    plt.subplot(2,2,3)
    sns.distplot(train_trim['age'])
    plt.subplot(2,2,4)
    sns.boxplot(train_trim['age'])
    plt.savefig('../plots/task1.5/age1.png')
    #plt.show()

    train_trim.to_csv('../results/cases_2021_train_processed.csv')
    percentile25 = cases_test['age'].quantile(0.25)
    percentile75 = cases_test['age'].quantile(0.75)
    iqr = percentile75 - percentile25
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    cases_test[cases_test['age'] > upper_limit]
    cases_test[cases_test['age'] < lower_limit]
    test_trim = cases_test[cases_test['age'] < upper_limit]
    test_trim.shape
    #Uncomment to produce plots
    plt.figure(figsize=(16,8))
    plt.subplot(2,2,1)
    sns.distplot(cases_test['age'])
    plt.subplot(2,2,2)
    sns.boxplot(cases_test['age'])
    plt.subplot(2,2,3)
    sns.distplot(test_trim['age'])
    plt.subplot(2,2,4)
    sns.boxplot(test_trim['age'])
    plt.savefig('../plots/task1.5/age2.png')
    
     #plt.show()
    test_trim.to_csv('../results/cases_2021_test_processed.csv')
    location = pd.read_csv('../results/location_2021_processed.csv')
    #Uncomment to produce plots
    plt.figure(figsize=(21,5))
    plt.subplot(1,3,1)
    sns.distplot(location['Incident_Rate'])
    plt.subplot(1,3,2)
    sns.distplot(location['Case_Fatality_Ratio'])
    plt.subplot(1,3,3)
    sns.distplot(location['Active'])
    plt.figure(figsize=(32,8))
    plt.subplot(1,4,1)
    plt.subplot(1,4,2)
    plt.subplot(1,4,3)
    sns.boxplot(location['Active'])
    plt.subplot(1,4,4)
    sns.boxplot(location['Confirmed'])
    plt.savefig('../plots/task1.5/all_test.png')
    percentile25 = location['Incident_Rate'].quantile(0.25)
    percentile75 = location['Incident_Rate'].quantile(0.75)
    iqr = percentile75 - percentile25
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    location[location['Incident_Rate'] > upper_limit]
    location[location['Incident_Rate'] < lower_limit]
    new_df_location = location[location['Incident_Rate'] < upper_limit]
    new_df_location.shape
    percentile25 = new_df_location['Case_Fatality_Ratio'].quantile(0.25)
    percentile75 = new_df_location['Case_Fatality_Ratio'].quantile(0.75)
    iqr = percentile75 - percentile25
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    new_df_location[new_df_location['Case_Fatality_Ratio'] > upper_limit]
    new_df_location[new_df_location['Case_Fatality_Ratio'] < lower_limit]
    new_df_location = new_df_location[new_df_location['Case_Fatality_Ratio'] < upper_limit]
    new_df_location.shape
    new_df_location.to_csv('../results/location_2021_processed.csv')

def section_seven():
    print("Section 7: Calculating MI scores and dropping features")
    clean_train=pd.read_csv('../results/cases_2021_train_processed.csv')
    clean_test=pd.read_csv('../results/cases_2021_test_processed.csv')
    data_train=clean_train[['sex','province','country','date_confirmation','additional_information','source','chronic_disease_binary']].values
    oe = OrdinalEncoder()
    oe.fit(data_train)
    data_train_encoding = oe.transform(data_train)
    target_train=clean_train[['outcome_group']].values
    le = LabelEncoder()
    le.fit(target_train)
    target_train_encoding = le.transform(target_train)
    categorical_MI = [0,0,0,0,0,0,0]
    for i in range(1,6):
        fs = SelectKBest(score_func=MIC, k='all')
        fs.fit(data_train_encoding, target_train_encoding)
        data_train_fs = fs.transform(data_train_encoding)
        categorical_MI = np.add(categorical_MI, fs.scores_)
    categorical_MI = categorical_MI/i
    plt.figure()
    plt.bar([i for i in range(len(categorical_MI))], categorical_MI)
    plt.savefig('../plots/task1.7/categorical_MI.png')
    
    #To show plots please uncomment this line
    #
    data_train2=clean_train[['age','latitude','longitude','Confirmed','Recovered','Deaths','Active','Incident_Rate','Case_Fatality_Ratio','lat_prov','long_prov']].values
    numerical_MI = [0,0,0,0,0,0,0,0,0,0,0]
    for i in range(1,6):
        mi=MIC(data_train2,target_train_encoding)
        numerical_MI = np.add(numerical_MI, mi)
    numerical_MI = numerical_MI/i
    plt.figure()
    plt.bar([i for i in range(len(numerical_MI))], numerical_MI)
    clean_train = clean_train.drop(columns='sex')
    clean_train = clean_train.drop(columns='chronic_disease_binary')
    clean_train = clean_train.drop(columns='age')
    clean_test = clean_test.drop(columns='sex')
    clean_test = clean_test.drop(columns='chronic_disease_binary')
    clean_test = clean_test.drop(columns='age')
    clean_train.to_csv('../results/cases_2021_train_processed_features.csv')
    clean_test.to_csv('../results/cases_2021_test_processed_features.csv')
    plt.savefig('../plots/task1.7/numerical_MI.png')
    #To show plots please uncomment this line:
    #plt.show()

def section_six():
    print("Section 6: Merging Test and Train with Location")
    location_clean = pd.read_csv('../results/location_2021_processed.csv').rename(
        columns={'Country_Region': 'country', 'Province_State': 'province','Lat':'lat_prov','Long_':'long_prov'})

    cases_train1 = pd.read_csv('../results/cases_2021_train_processed.csv')
    cases_test1 = pd.read_csv('../results/cases_2021_test_processed.csv')
    
    location_clean = location_clean.groupby(['country','province']).agg({'Confirmed':'sum','Recovered':'sum',
                                                        'Deaths':'sum', 'Active':'sum',
                                                        'Incident_Rate':'mean',
                                                        'Case_Fatality_Ratio':'mean','lat_prov':'mean','long_prov':'mean'})                                                 

    merged_train = pd.merge(cases_train1,location_clean, how='inner', on=['country', 'province'])
    merged_train.drop(columns=merged_train.columns[0], axis=1, inplace=True)
    print("Number of rows in cases_2021_train_processed: ", len(merged_train.index))
    merged_train.to_csv('../results/cases_2021_train_processed.csv')
    merged_test = pd.merge(cases_test1, location_clean, how='inner', on=['country', 'province'])
    merged_test.drop(columns=merged_test.columns[0], axis=1, inplace=True)
    print("Number of rows in cases_2021_test_processed: ", len(merged_test.index))
    merged_test.to_csv('../results/cases_2021_test_processed.csv')


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(hav))
main()





