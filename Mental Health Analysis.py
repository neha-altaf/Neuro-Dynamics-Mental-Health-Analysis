import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#loading dataset
dataset=pd.read_csv(r'C:\Users\DELL\Desktop\Mental health data analysis python project\Mental Health Dataset.csv\Mental Health Dataset.csv')

                    #inspecting our  data

print(dataset.info())


#converting datatypes
dataset['Timestamp']=pd.to_datetime(dataset['Timestamp'])
dataset[['Gender','Country','Occupation']]=dataset[['Gender','Country','Occupation']].apply(pd.Categorical)

#checking for null values
print(dataset.isnull().sum())
#this shows  Self employeed column contains 5202 blanks
#so its better to replace blanks with Not specified word
dataset["self_employed"]=dataset["self_employed"].fillna('Not Specified')

#extracting month and year from time stamp column
dataset.loc[:, 'year-month']=dataset['Timestamp'].dt.strftime('%Y-%m')
dataset['year-month']=pd.to_datetime(dataset['year-month'])
#checking for duplicates
print(dataset.duplicated().sum())

#removing duplicates
newdataset=dataset.drop_duplicates()
print(newdataset)
print(newdataset.info())
#-----------------------------
#GENERAL DISTRIBUTION ANALYSIS
#-----------------------------

#This analysis shows count of respondents by gender,country,occupation in our data

#Occupation Distribution
OD=newdataset['Occupation'].value_counts()
print(OD)

OD.plot(kind='line',color='yellow',zorder=2)
plt.gca().set_facecolor('black')
plt.grid(axis='both',linestyle='--',color='grey',alpha=0.5,zorder=1)
plt.title('Count of Respondants by Occupation')
plt.show()
#this shows housewives are on lead


#Gender Distribution by Occupation
print(newdataset['Gender'].value_counts())

ax=sns.countplot(x='Gender',data=newdataset,hue='Occupation')
ax.bar_label(ax.containers[0],color='blue')
ax.bar_label(ax.containers[1],color='yellow')
ax.bar_label(ax.containers[2],color='green')
ax.bar_label(ax.containers[3],color='red')
ax.bar_label(ax.containers[4],color='purple')
plt.gca().set_facecolor('black')
plt.title('Count of Respondants by Gender, categorized by Occupation')
plt.yticks([])
plt.show()
#this shows male respondents are in large amount as comparison to female and males
#which are house husbands, are in lead


#Country Distribution
CD=newdataset['Country'].value_counts().head(10)
print(CD)

plt.figure(figsize=(12,5))
bx=CD.plot(kind='barh',color='yellow')
bx.bar_label(bx.containers[0],color='white')
plt.gca().set_facecolor('black')
plt.title('Count of Respondants by Country')
plt.xticks([])
plt.show()
#this shows USA is on lead

#count of respondent's seeking treatments
print(dataset['treatment'].value_counts(normalize=True)*100)
plt.figure(figsize=(4,4))
newdataset['treatment'].value_counts().plot(kind='pie',autopct='%1.1f%%',title='Seeking Treatment')
plt.show()
#this shows 50.4% of respondents are taking treatment while  49.5 are not

#count of respondent's family history
print(dataset['family_history'].value_counts(normalize=True)*100)
plt.figure(figsize=(4,4))
newdataset['family_history'].value_counts().plot(kind='pie',autopct='%1.1f%%',title='family_history')
plt.show()
#this shows 39.5% respondents have family history of mental illness while 60.5% have don't

#---------------
#TREND ANALYSIS
#---------------
#This analysis help us to identify how different variables relate over groups

#Mental Health History Across Occupations

occupation_mental_health=(newdataset.groupby('Occupation',observed=True)['Mental_Health_History'].value_counts(normalize=True)*100).unstack()
print(occupation_mental_health)

occupation_mental_health.plot(kind='bar',color=('#03A9F4','#32CD32','#ffff00'),zorder=2)
plt.xticks(rotation=30)
plt.legend(loc='best',bbox_to_anchor=(0.5,1.05))
plt.ylabel('count of respondents in %age')
plt.gca().set_facecolor('black')
plt.grid(axis='both',linestyle='--',alpha=0.5, zorder=1)
plt.title('Mental Health History by Occupation', pad=20)  
plt.show()

#Indoor Time vs Growing Stress
indoor_stress=(newdataset.groupby('Days_Indoors',observed=True)['Growing_Stress'].value_counts(normalize=True)*100).unstack()
print(indoor_stress)


indoor_stress.plot(kind='line', figsize=(14,5), zorder=2)
plt.grid(axis='both',linestyle='--',alpha=0.5, zorder=1)
plt.gca().set_facecolor('black')
plt.ylabel('count of respondents in %age ')
plt.title('Relation between Growing stress & Days Indoor')
plt.show()

#------------------------
#CORRELATION ANALYSIS
#------------------------
#This analysis explores the relation between different variables

#Relation Between Family History & Treatment

family_treatment=pd.crosstab(newdataset['family_history'],newdataset['treatment'])
print(family_treatment)

ex=family_treatment.plot(kind='bar',zorder=2)
ex.bar_label(ex.containers[0],color='white')
ex.bar_label(ex.containers[1],color='white')
plt.gca().set_facecolor('black')
plt.ylabel('count of respondents')
plt.yticks([])
plt.title('Relation between Family History & Treatment')
plt.show()

#Work interest by mental health history
WIMHH=(newdataset.groupby('Work_Interest',observed=True)['Mental_Health_History'].value_counts(normalize=True)*100).unstack()
print(WIMHH)

WIMHH.plot(kind='bar',color=('#03A9F4','#32CD32','#ffff00'),zorder=2)
plt.xticks(rotation=360)
plt.legend(loc='best',bbox_to_anchor=(0.5,1.05))
plt.ylabel('count of respondents in %age')
plt.gca().set_facecolor('black')
plt.grid(axis='both',linestyle='--',alpha=0.5, zorder=1)
plt.title('Work Interest by Mental Health History ', pad=20)  
plt.show()


#----------------------
#TIME SERIES ANALYSIS
#---------------------

#Respondents over Time

respondents_over_time=newdataset.groupby('year-month')[['year-month','Occupation']].value_counts()
print(respondents_over_time)



plt.figure(figsize=(12,6))
sns.countplot(x='year-month', data=newdataset,hue='Occupation')
plt.title('Respondents over time by occupation')
plt.xlabel('year-month')
plt.ylabel('Count of Respondents')
plt.gca().set_facecolor('black')
plt.show()

#Treatment seeking behaviour over time

treatment_trend=newdataset.groupby('year-month')['treatment'].value_counts().unstack()
print(treatment_trend)


treatment_trend.plot(kind='line',marker='o',colormap='coolwarm',zorder=2,linewidth=2,figsize=(12,6))
plt.title('Trends in Mental Health Treatment Seeking Over Time')
plt.xlabel('year-month')
plt.ylabel('Count of Respondents')
plt.legend=['No','Yes']
plt.xticks(rotation=20)
plt.gca().set_facecolor('black')
plt.grid(axis='both',linestyle='--',color='grey',alpha=0.5,zorder=1)
plt.show()


#Trends in Mood Swings Over Time
mood_swings_trend=newdataset.groupby('year-month')['Mood_Swings'].value_counts().unstack()
print(mood_swings_trend)


mood_swings_trend.plot(kind='line',marker='o',colormap='coolwarm',zorder=2,linewidth=2,figsize=(12,6))
plt.title('Trends in Mood Swings Over Time')
plt.xlabel('year-month')
plt.ylabel('Count of Respondents')
plt.legend=['High','Low','Medium']
plt.gca().set_facecolor('black')
plt.grid(axis='both',linestyle='--',color='grey',alpha=0.5,zorder=1)
plt.show()

#Trends in growing stress over
growing_stress_trend=newdataset.groupby('year-month')['Growing_Stress'].value_counts().unstack()
print(growing_stress_trend)


growing_stress_trend.plot(kind='bar',colormap='coolwarm',zorder=2,linewidth=2,figsize=(13,6))
plt.title('Trends in Growing Stress Over Time')
plt.xlabel('year-month')
plt.ylabel('Count of Respondents')
plt.legend=['Yes','No','Maybe']
plt.gca().set_facecolor('black')
plt.xticks(rotation=25)
plt.grid(axis='both',linestyle='--',color='grey',alpha=0.5,zorder=1)
plt.show()

#Trend in copng struggle over time
coping_struggle_trend=newdataset.groupby('year-month')['Coping_Struggles'].value_counts().unstack()
print(coping_struggle_trend)


coping_struggle_trend.plot(kind='bar',colormap='coolwarm',zorder=2,linewidth=2,figsize=(12,6))
plt.title('Trends in Coping Struggle Over Time')
plt.xlabel('year-month')
plt.ylabel('Count of Respondents')
plt.legend=['Yes','No']
plt.gca().set_facecolor('black')
plt.xticks(rotation=25)
plt.grid(axis='both',linestyle='--',color='grey',alpha=0.5,zorder=1)
plt.show()


#-----------------------
#MACHINE LEARNING MODELS
#-----------------------
#Goal:
#Predict whether a respondent will seek mental health treatment based on
#various factors (e.g., family history, work interest, stress levels).

##Recommended Models:
##Logistic Regression: For binary classification (seeking treatment: Yes/No).
##Random Forest: For better accuracy and feature importance analysis.
##K-Means Clustering: To segment respondents based on mental health conditions.



from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Encoding categorical variables
ord_enc = OrdinalEncoder()
ord_enc.fit(newdataset[['Gender']])
newdataset['gender_enc']=ord_enc.transform(newdataset[['Gender']])
#using map function for encoding
newdataset['stress_enc']=newdataset['Growing_Stress'].map({"Yes":1,"No":0,"Maybe":0.5})
newdataset['family_history_enc']=newdataset['family_history'].map({"Yes":1,"No":0})
newdataset['treatment_enc']=newdataset['treatment'].map({"Yes":1,"No":0})


# Selecting features and target
X = newdataset[["gender_enc","stress_enc", "family_history_enc"]]  # Features
y = newdataset["treatment_enc"]  # Target variable


# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


#using cross validation for accuracy
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
scores=cross_val_score(clf,X,y,cv=kfold,scoring='accuracy')
print("Average Accuracy:", scores.mean())


#visualization
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True,cmap='Blues',fmt='d')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


#------------------------
#Dynamic System Analysis
#------------------------

from scipy.integrate import odeint

#aligning our dataset with dynamical system

# Estimate beta and gamma from dataset
newdataset["Healthy"] =((newdataset["Growing_Stress"] == "No") & (newdataset["Coping_Struggles"] == "No")).astype(int)
newdataset["Stressed"] =((newdataset["Growing_Stress"] == "Yes")&(newdataset["Coping_Struggles"] =="Yes")).astype(int)
newdataset["Recovered"] =((newdataset["treatment"]=="Yes") & (newdataset["Growing_Stress"] =="No")).astype(int)

#groupby dataset and get the number of healthy stressed and recovered cases
time_series =newdataset.groupby("year-month")[["Healthy","Stressed","Recovered"]].sum()

time_series["New_Stressed"] = time_series["Stressed"].diff().fillna(0)  # Compute new stressed cases
#time_series["Beta"] = time_series["New_Stressed"] / (time_series["Healthy"] + 1)  # Avoid division by zero

time_series["New_Recovered"] = time_series["Recovered"].diff().fillna(0)  # Compute new recovered cases
#time_series["Gamma"] = time_series["New_Recovered"] / (time_series["Stressed"] + 1)  # Avoid division by zero

time_series['New_Stressed'].clip(lower=0,inplace=True)
time_series['New_Recovered'].clip(lower=0,inplace=True)

time_series["Beta"] = time_series["New_Stressed"] / (time_series["Healthy"] + 1)
time_series["Gamma"] = time_series["New_Recovered"] / (time_series["Stressed"] + 1)
print(time_series)
#final finding beta and gamma
beta= time_series["Beta"].mean()
gamma= time_series["Gamma"].mean()
print(beta)
print(gamma) 



# Define the model (SIR-like)
def mental_health_system(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I  # Healthy to stressed transition
    dIdt = beta * S * I - gamma * I  # Stressed individuals
    dRdt = gamma * I  # Recovery process
    return [dSdt, dIdt, dRdt]

# Initial conditions from dataset
S0=0.159     #initially 15.9%healthy
I0=0.68      #68% already stressed
R0=0.161    #16.1% had recovered
y0 = [S0, I0, R0]

# Time span
t = time_series.index.astype('int64').values


# Solve the system
solution = odeint(mental_health_system, y0, t, args=(beta, gamma))
S, I, R = solution.T

# Plot the results
plt.figure(figsize=(10,6))
plt.plot(t, S, label="Healthy")
plt.plot(t, I, label="Stressed", color="red")
plt.plot(t, R, label="Recovered", color="green")
plt.xlabel("Time")
plt.ylabel("Population Proportion")
plt.title("Dynamical System Model of Mental Health")
plt.legend()
plt.show()





