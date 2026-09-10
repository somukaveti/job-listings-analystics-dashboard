#  Job Listings Analytics Dashboard

A Streamlit-based interactive dashboard to analyze fake job listings dataset (from Real Python).

###  Live Demo
`localhost:8501` after running locally

###  Features
- **KPIs:** Total Jobs, Average Salary, Total Companies, Locations
- **Filters:** Job Type, Experience Level, Work Mode
- **Visualizations:**
    - Salary Distribution (Histogram)
    - Jobs by Experience Level (Bar Chart)
    - Top 10 Hiring Companies (Horizontal Bar)
    - Job Type Distribution (Pie Chart)
    - Salary vs Experience Correlation
- **Data Preview:** Interactive table with 1000+ job records

###  Tech Stack
- Python 3.10+
- Streamlit
- Pandas, Matplotlib, Seaborn
- Scikit-learn (for Label Encoding)

###  Folder Structure
job-listings-analytics-dashboard/
│
├── app.py  (mundu file ni rename chey - space teesi)
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── processed_dataset.csv
│
├── notebooks/
│   └── EDA.ipynb  (nuvvu analysis chesina file)
│
└── screenshots/
    ├── dashboard.png
    └── charts.png
