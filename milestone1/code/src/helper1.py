import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.datasets import load_breast_cancer as LBC


pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.feature_selection import mutual_info_classif as MIC, SelectKBest

clean_train=pd.read_csv('merged_train.csv')
X=clean_train[['sex','province','country','date_confirmation','additional_information','source','chronic_disease_binary']].values
oe = OrdinalEncoder()
oe.fit(X)
X_enc = oe.transform(X)
y=clean_train[['outcome_group']].values
le = LabelEncoder()
le.fit(y)
y_enc = le.transform(y)

fs = SelectKBest(score_func=MIC, k='all')
fs.fit(X_enc, y_enc)
X_train_fs = fs.transform(X_enc)


X2=clean_train[['age','latitude','longitude','Confirmed','Recovered','Deaths','Active','Incident_Rate','Case_Fatality_Ratio','lat_prov','long_prov']].values
mi=MIC(X2,y_enc)

scores={'age':0.01661622,'sex':0.001387,'province':0.054282, 'country':0.019652, 'date_confirmation':0.034015,
'additional_information': 0.036494,
'source': 0.064742,
'chronic_disease_binary': 0.003042,
'latitude':0.05018424,'longitude': 0.05646562,'Confirmed': 0.05209936 , 'Recovered':0.0539951,'Deaths': 0.05326773,
 'Active':0.05192683,'Incident_Rate': 0.05220169,'Case_Fatality_Ratio': 0.05131681,'lat_prov': 0.05184165 ,'long_prov':0.05488547
}
scoresList=list(scores.items())
featuredf=pd.DataFrame(scoresList,columns=['feature','MI score'])
featuredf=featuredf.sort_values(by='MI score',ascending=False)
print(featuredf)
