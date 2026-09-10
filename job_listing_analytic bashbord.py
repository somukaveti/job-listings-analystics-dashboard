import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import random

url = "https://realpython.github.io/fake-jobs/"
res = requests.get(url)
soup = BeautifulSoup(res.text, 'html.parser')
cards = soup.find_all('div', class_='card-content')

job_title = []
company_name = []
location = []
job_desc = []
apply_link = []
salary = []
exp_level = []
job_type = []
work_mode = []

for card in cards:

    title = card.find('h2', class_='title').text.strip()
    company = card.find('h3', class_='subtitle').text.strip()
    loc = card.find('p', class_='location').text.strip()
    desc = card.find('div', class_='content').text.strip()
    link = card.find_all('a')[1]['href']

    sal = random.choice([np.nan, 45000, 65000, 85000, 110000, 50000])
    exp = random.choice(['Entry-Level', 'Mid-Level', 'Senior', np.nan])
    jtype = random.choice(['Full-time', 'Part-time', 'Contract'])
    wmode = random.choice(['Remote', 'Hybrid', 'On-site', np.nan])

    job_title.append(title)
    company_name.append(company)
    location.append(loc)
    job_desc.append(desc)
    apply_link.append(link)
    salary.append(sal)
    exp_level.append(exp)
    job_type.append(jtype)
    work_mode.append(wmode)

    # TAPPU - Nuvvu chesindi
print(len(title))  # Idi "Product manager" letters = 15      

df = pd.DataFrame({
    'Job Title': job_title,
    'Company Name': company_name,
    'Location': location,
    'Job Description': job_desc,
    'Apply Link': apply_link,
    'Salary': salary,
    'Experience Level': exp_level,
    'Job Type': job_type,
    'Work Mode': work_mode
})

print(df.head())
print(f"Columns: {len(df.columns)} - {list(df.columns)}")
print(f"Shape: {df.shape}") # 10

df.to_csv('raw_dataset.csv', index=False)
print("raw_dataset.csv Saved - 9 columns")

print(df.isnull().sum())
mean_sal = df['Salary'].mean()

# --- FIXED CLEANING CODE ---
mean_salary = df['Salary'].mean()
mode_exp = df['Experience Level'].mode()[0]
mode_work = df['Work Mode'].mode()[0]

df['Salary'] = df['Salary'].fillna(mean_salary)
df['Experience Level'] = df['Experience Level'].fillna(mode_exp)
df['Work Mode'] = df['Work Mode'].fillna(mode_work)

df = df.drop_duplicates()
print(df.info())
df.to_csv('cleaned_dataset.csv', index=False)

df['Salary Category'] = pd.cut(
    df['Salary'], 
    bins=[0,60000,90000,200000], 
    labels=['Low','Medium','High']
)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['Experience Encoded'] = le.fit_transform(df['Experience Level'])

encoded = pd.get_dummies(df['Work Mode'], prefix='Mode')

df = pd.concat([df, encoded], axis=1)

df['Is_Active'] = 1

df.to_csv('processed_dataset.csv', index=False)

print("Processed - 9 + new columns")

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('cleaned_dataset.csv')
print(df.head())

df["Salary Category"] = pd.cut(
    df["Salary"],
    bins=[0, 60000, 90000, 200000],
    labels=["Low", "Medium", "High"]
)

df["Job Category"] = df["Job Title"].apply(lambda x: 'Tech' if 'Python' in x or 'Developer' in x else 'Non-Tech')
df["High Salary"] = df["Salary"] > 90000

print(df["Salary Category"].head(10))

le = LabelEncoder()
df["Experience Label"] = le.fit_transform(df["Experience Level"])
df["WorkMode Label"] = le.fit_transform(df["Work Mode"])
print(df.head())

encoded = pd.get_dummies(df["Work Mode"], prefix="Mode")
print(encoded.head())
encoded2 = pd.get_dummies(df["Salary Category"], prefix="SalaryCat")
df = pd.concat([df, encoded, encoded2], axis=1)

df["High Salary"] = df["High Salary"].astype(int)
df["Is_Active"] = True
df["Is_Active"] = df["Is_Active"].astype(int)
df["salary_per_exp"] = df["Salary"] / (df["Experience Label"]+1)

scaler = StandardScaler()
df["Scaled_Salary"] = scaler.fit_transform(df[["Salary"]])

print(df.info())
print(df.head())
df.to_csv('processed_dataset.csv', index=False)

import pandas as pd
import numpy as np

df = pd.read_csv('processed_dataset.csv')

print(df.head())
print(df.tail())
print(df.info())
print(df.describe())

print(df["Salary"].min())
print(df["Salary"].max())
print(df["Salary"].mean())
print(df["Salary"].median())

print(df["Experience Level"].value_counts())
print(df["Experience Level"].unique())

print(df.groupby('Experience Level')['Salary'].mean())
print(df.agg({'Salary': ['mean','max','min','std']}))
print(df.sort_values(by='Salary', ascending=False).head(5))
print(df.sort_values(by='Salary', ascending=True).head(5))
print(df[df['Salary']>80000].head())

print(pd.pivot_table(df, values='Salary', index='Experience Level', aggfunc=np.mean))
print(pd.crosstab(df['Experience Level'], df['Salary Category']))
print(df[['Salary','Experience Label','WorkMode Label']].corr())


import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

plt.figure()
sns.histplot(df["Salary"], kde=True)
plt.title("Histogram - Salary")
plt.savefig("hist_salary.png")
st.pyplot(plt)

plt.figure()
sns.countplot(x="Salary Category", data=df)
plt.title("Count Plot - Salary Category")
plt.savefig("count_salarycat.png")
st.pyplot(plt)

plt.figure()
sns.boxplot(y="Salary", data=df)
plt.title("Box Plot - Salary")
plt.savefig("box_salary.png")
st.pyplot(plt)

plt.figure()
sns.barplot(x="Experience Level", y="Salary", data=df)
plt.title("Bar Plot - Experience vs Salary")
plt.savefig("bar_exp_sal.png")
st.pyplot(plt)

plt.figure()
sns.scatterplot(data=df, x="Experience Label", y="Salary")
plt.title("Scatter Plot - Experience vs Salary")
plt.savefig("scatter.png")
st.pyplot(plt)

plt.figure()
sns.barplot(x="Work Mode", y="Salary", data=df)
plt.title("Bar Plot - Work Mode vs Salary")
st.pyplot(plt)

plt.figure()
sns.boxplot(x="Experience Level", y="Salary", data=df)
plt.title("Box Plot - Exp vs Salary")
st.pyplot(plt)

plt.figure()
sns.violinplot(x="Experience Level", y="Salary", data=df)
plt.title("Violin Plot")
plt.savefig("violin.png")
st.pyplot(plt)

sns.pairplot(df[["Salary","Experience Label","WorkMode Label"]])
plt.savefig("pairplot.png")
st.pyplot(plt)

plt.figure()
sns.heatmap(df[["Salary","Experience Label","WorkMode Label","Is_Active"]].corr(), annot=True, cmap="coolwarm")
plt.title("Heatmap")
plt.savefig("heatmap.png")
st.pyplot(plt)

g = sns.FacetGrid(df, col="Salary Category")
g.map_dataframe(sns.histplot, x="Salary")
g.savefig("facetgrid.png")
st.pyplot(plt)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('processed_dataset.csv')
st.title("Job Analytics - 100 Rows, 9 Columns")
menu = st.sidebar.radio(
    "Go to", 
    ["Home","Dataset","Stats","Filter","Charts","Insights","Download"]
)
if menu=="Home":
    st.metric("Total", len(df))
    st.metric(
        "Avg Salary", 
        int(df
            ['Salary'].
            mean()
            )
)
    st.write(df.shape)
elif menu=="Dataset":
    st.dataframe(df)
elif menu=="Stats":
    st.write(df.describe())
    st.write(df['Experience Level'].
             value_counts()
)
elif menu=="Filter":
    exp = st.selectbox("Exp", df['Experience Level'].unique())
    st.dataframe(df[df['Experience Level']==exp])

elif menu=="Charts":
    fig, ax = plt.subplots()
    sns.histplot(df['Salary'],
                 ax=ax)
    st.pyplot(fig)
    fig2, ax2 = plt.subplots()
    sns.countplot(
        x='Salary Category', 
        data=df, ax=ax2)
    st.pyplot(fig2)

elif menu=="Insights":
    st.write(f"High Salary jobs: {len(df[df['Salary']>90000])}")
    st.write(f"Most common Exp: {df['Experience Level'].mode()[0]}")
st.download_button(
    "Download CSV", 
    df.to_csv(index=False), 
    "processed_dataset.csv"
    )
        
