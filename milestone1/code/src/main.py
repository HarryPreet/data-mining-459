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
    
    
    cases_train.to_csv('cases_train_clean.csv')
    cases_test.to_csv('cases_test_clean.csv')
    location.to_csv('location_clean.csv')

    location_province_train()
    location_province_test()
    
def location_province_train():
    location = pd.read_csv('../data/location_2021.csv')
    cases_train = pd.read_csv('cases_train_clean.csv')
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
    location.to_csv('location_clean_train.csv')

def location_province_test():
    location = pd.read_csv('../data/location_2021.csv')
    cases_test = pd.read_csv('cases_test_clean.csv')
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
    location.to_csv('location_clean_test.csv')

def section_six():
    
    location_clean = pd.read_csv('location_clean.csv').rename(
        columns={'Country_Region': 'country', 'Province_State': 'province'})

    cases_train1 = pd.read_csv('cases_train_clean.csv')
    cases_test1 = pd.read_csv('cases_test_clean.csv')
    
    location_clean = location_clean.groupby(['country','province']).agg({'Confirmed':'sum','Recovered':'sum',
                                                        'Deaths':'sum', 'Active':'sum',
                                                        'Incident_Rate':'mean',
                                                        'Case_Fatality_Ratio':'mean'})                                                 

    merged_train = pd.merge(cases_train1,location_clean, how='inner', on=['country', 'province'])
    merged_train.drop(columns=merged_train.columns[0], axis=1, inplace=True)
    merged_train.to_csv('merged_train.csv')
    merged_test = pd.merge(cases_test1, location_clean, how='inner', on=['country', 'province'])
    merged_test.drop(columns=merged_test.columns[0], axis=1, inplace=True)
    merged_test.to_csv('merged_test.csv')


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(hav))
main()





