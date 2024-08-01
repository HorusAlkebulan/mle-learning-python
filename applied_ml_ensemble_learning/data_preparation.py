import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

"""
## Prepare Data: Clean the data

Using the Titanic dataset from [this](https://www.kaggle.com/c/titanic/overview) Kaggle competition (we are only using the training set).

This dataset contains information about 891 people who were on board the ship when departed on April 15th, 1912. As noted in the description on Kaggle's website, some people aboard the ship were more likely to survive the wreck than others. There were not enough lifeboats for everybody so women, children, and the upper-class were prioritized. Using the information about these 891 passengers, the challenge is to build a model to predict which people would survive based on the following fields:

- **Name** (str) - Name of the passenger
- **Pclass** (int) - Ticket class
- **Sex** (str) - Sex of the passenger
- **Age** (float) - Age in years
- **SibSp** (int) - Number of siblings and spouses aboard
- **Parch** (int) - Number of parents and children aboard
- **Ticket** (str) - Ticket number
- **Fare** (float) - Passenger fare
- **Cabin** (str) - Cabin number
- **Embarked** (str) - Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
"""

PROJECT_ROOT = os.path.dirname(__file__)
HR = "\n-----------------------------------------------------------------------"

titantic_path = os.path.join(PROJECT_ROOT, "titanic.csv")
titanic = pd.read_csv(titantic_path)

print(f"Data loaded from {titantic_path}:")
print(titanic.head())

# fill in missing age with mean value
print(HR)
print("Missing age data:")
print(titanic.isnull().sum())

mean_age = titanic["Age"].mean()
print(f"Setting missing age to mean value: {mean_age}")

# NOTE: Rewritten from original after Pandas 3.0 warning
# The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
# For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.

fill_dict = {"Age": mean_age}
titanic.fillna(fill_dict, inplace=True)
print(f"Updated data:")
print(titanic.head(10))

# combine SibSp and Parch since they are correlated as shown in the plot
print(HR)
for i, col in enumerate(["SibSp", "Parch"]):
    plt.figure(i)
    sns.catplot(
        x=col,
        y="Survived",
        data=titanic,
        kind="point",
        aspect=2,
    )
    sns_path = os.path.join(PROJECT_ROOT, f"{col}_correlation.png")
    print(f"Saving plot to {sns_path}")
    plt.savefig(sns_path)

titanic["FamilyCount"] = titanic["SibSp"] + titanic["Parch"]
titanic.drop(["PassengerId", "SibSp", "Parch"], axis=1, inplace=True)
print("Updated dataframe:")
print(titanic.head(10))

# clean up the categorical variables
print(HR)
print(f"Show missing counts for Cabin:\n {titanic["Cabin"].isnull().sum()}")

# group by survival and cabin null
survival_group_by_cabin_isnull_df = titanic.groupby(titanic["Cabin"].isnull())["Survived"]
group_by_mean_df = survival_group_by_cabin_isnull_df.mean()
print(f"group_by_mean:\n{group_by_mean_df}")

# create new column HasCabin
titanic["HasCabin"] = np.where(titanic["Cabin"].isnull(), 0, 1)
print(f"Updated data:\n{titanic.head()}")

# convert sex to numeric using a map
print(HR)
gender_map = {
    "male": 0,
    "female": 1,
}
titanic["Sex"] = titanic["Sex"].map(gender_map)
print(f"Updated dataframe after mapping sex:\n{titanic.head()}")

# lastly drop unnecessary variables
titanic.drop(["Cabin", "Embarked", "Name", "Ticket"], axis=1, inplace=True)
print(f"Updated dataframe after dropping last unneeded variables:\n{titanic.head()}")

# export clean data
print(HR)
clean_data_path = os.path.join(PROJECT_ROOT, "titanic_cleaned.csv")
print(f"Saving cleaned data to {clean_data_path}")
titanic.to_csv(clean_data_path)