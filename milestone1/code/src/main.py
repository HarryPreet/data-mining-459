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

cases_train = pd.read_csv('../data/cases_2021_train.csv')
cases_test = pd.read_csv('../data/cases_2021_test.csv')
location = pd.read_csv('../data/location_2021.csv')

def main():
    global cases_train, cases_test, location
    section_one()
    section_four()
    cases_train.to_csv('cases_train_clean.csv')
    cases_test.to_csv('cases_test_clean.csv')
    location.to_csv('location_clean.csv')
    section_six()
#1.1

def section_one():
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
    #Cleaning Training Dataset
    #print("Train Dataset(before cleaning):")
    #print(len(cases_train.index))
    #print(cases_train.isna().sum())


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
    #print("Train Dataset(after cleaning):")
    #print(cases_train.isna().sum())
    #print(len(cases_train.index))
    cases_train['latitude'] = cases_train['latitude'].apply(lambda x: float(x))
    cases_train['longitude'] = cases_train['longitude'].apply(lambda x: float(x))
    

    #Cleaning Test Dataset
    #print("Test Dataset(before cleaning):")
    #print(len(cases_test.index))
    #print(len(cases_train.index))
    
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
    #print("Test Dataset(after cleaning):")
    #print(cases_test.isna().sum())
    #print(len(cases_test.index))

    #Cleaning Location Data
    #print("Location Dataset(before cleaning):")
    #print(location.isna().sum())
    #print(len(location.index))
    location['Incident_Rate'] = location['Incident_Rate'].fillna(location.groupby('Country_Region')['Incident_Rate'].transform('mean'))
    location['Case_Fatality_Ratio'] = location['Case_Fatality_Ratio'].fillna(location.groupby('Country_Region')['Case_Fatality_Ratio'].transform('mean'))
    location['Recovered'] = location['Recovered'].fillna(location.groupby('Country_Region')['Recovered'].transform('mean'))
    location['Active'] = location['Active'].fillna(location.groupby('Country_Region')['Active'].transform('mean'))
    location = location.drop(location[location['Lat'].isna()].index)
    location = location.drop(location[location['Long_'].isna()].index)
    location = location.drop(location[location['Case_Fatality_Ratio'].isna()].index)
    #print(location.isna().sum())
    #print("Location Dataset(after cleaning):")
    #print(location.isna().sum())
    #print(len(location.index))
    location.loc[location['Country_Region'] == "Korea, South", 'Country_Region'] = 'South Korea'
    location.loc[location['Country_Region'] == 'US', 'Country_Region'] = 'United States'
    location.loc[location['Country_Region'] == 'Taiwan*', 'Country_Region'] = 'Taiwan'
    location['Lat'] = location['Lat'].apply(lambda x: float(x))
    location['Long_'] = location['Long_'].apply(lambda x: float(x))
    #Provinces
    #location = location.drop(location[location['Province_State'].isna()].index)
    
    #Find entries na in province of location
    
    location= location.reset_index()
    cases_train = cases_train.reset_index()
    #pass long and lat to closest 
    #print(provinceTemp.head())
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
    #print(location.isna().sum())
def section_six():
    df1 = pd.read_csv('location_clean.csv').rename(
        columns={'Country_Region': 'country', 'Province_State': 'province'})

    cases_train1 = pd.read_csv('cases_train_clean.csv')
    #cases_test1 = pd.read_csv('milestone1/code/src/cases_test_clean.csv')
    new_df1 = df1
    new_df1 = df1.groupby(['country','province']).agg({'Confirmed':'sum','Recovered':'sum',
                                                        'Deaths':'sum', 'Active':'sum',
                                                        'Incident_Rate':'mean',
                                                        'Case_Fatality_Ratio':'mean'})                                                 
    #print(cases_train.isna().sum())
    #print(new_df1.isna().sum())
    #print(new_df1.head())
    #print(cases_train1.head())
    #new_df1= new_df1.reset_index()
    #cases_train1 = cases_train1.reset_index()
    print(len(location.index))
    print(len(cases_train1.index))
    merged = pd.merge(cases_train1,new_df1, how='inner', on=['country', 'province'])
    print(len(merged.index))
    print(merged.head())
    #merged1 = pd.merge(cases_test1, new_df1, how='left', on=['country', 'province'])
    print(merged.loc[merged['country'] == "Yemen"])
    merged.to_csv('merged_train.csv')
    #merged1.to_csv('milestone1/code/data/merged_test.csv')
    # df2 = df1.groupby(['province', 'country']).sum()
    # df3 = df1.groupby(['province', 'country']).mean(numeric_only=True)
    # new_df1['Confirmed'] = df2['Confirmed']
    # new_df1['Recovered'] = df2['Recovered']
    # new_df1['Deaths'] = df2['Deaths']
    # new_df1['Active'] = df2['Active']
    # new_df1['Incident_Rate'] = df3['Incident_Rate']
    # new_df1['Case_Fatality_Ratio'] = df3['Case_Fatality_Ratio']

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(hav))
main()





