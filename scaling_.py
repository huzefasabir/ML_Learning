import pandas as pd
import numpy as np
import matplotlib.pyplot as mat
import seaborn as sb

# This code snippet is reading a CSV file named "loan.csv" located at "E:\Study\Jupyter Noteboook\" into a pandas DataFrame called `dataset`. 
# It then displays the first few rows of the dataset using `dataset.head()`. Following that, it checks for any missing values in the dataset using `dataset.isnull().sum()`.
dataset=pd.DataFrame(pd.read_csv("loan.csv"))
#dataset.head()

#dataset.isnull().sum()6
mat.subplot(1,2,1)
sb.kdeplot(data=dataset,x="ApplicantIncome")
mat.subplot(1,2,2)
sb.distplot(dataset["ApplicantIncome"])
mat.show()





# This code snippet is performing feature scaling on the "ApplicantIncome" column in the dataset using StandardScaler from scikit-learn. Here's a breakdown of each step:
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
scale.fit(dataset[["ApplicantIncome"]])
dataset["ApplicantIncome_ss"]=scale.transform(dataset[["ApplicantIncome"]])
print(dataset.head())
print(dataset.describe())


# The code snippet you provided is creating a side-by-side 
# comparison of the distribution of the "ApplicantIncome"
#  column before and after applying feature scaling using StandardScaler.

mat.subplot(1,2,1)
sb.distplot(dataset["ApplicantIncome"])
mat.subplot(1,2,2)
sb.distplot(dataset["ApplicantIncome_ss"])
mat.show()