import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt 
import numpy as np

PROJECT_ROOT = os.path.dirname(__file__)
HR = "-----------------------------------------"
if __name__ == "__main__":

    loan_data_path = os.path.join(PROJECT_ROOT, "loan.csv")
    print(f"Loading data from {loan_data_path}")
    loan_data_df = pd.read_csv(loan_data_path)
    print(HR)
    print(f"loan_data:\n{loan_data_df.head()}")
    print(HR)
    print(f"info:\n{loan_data_df.info()}")
    print(HR)
    print(f"describe:\n{loan_data_df.describe()}")

    ax = sns.boxplot(
        loan_data_df,
        x="Default",
        y="Income"
    )
    plt.title("Seaborn Box Plot")
    # plt.show()
    plot_path = os.path.join(PROJECT_ROOT, "boxplot_loan_default_income.png")
    print(f"Saved plot to {plot_path}")
    plt.savefig(plot_path)
    

    ax = sns.boxplot(
        loan_data_df,
        x="Default",
        y="Loan Amount",
    )
    plot_path = os.path.join(PROJECT_ROOT, "boxplot_loan_default_loan_amount.png")
    print(f"Saved plot to {plot_path}")
    plt.savefig(plot_path)

    ax = sns.scatterplot(
        x=loan_data_df["Income"],
        y=np.where(loan_data_df["Default"]=="No", 0, 1),
        s=150,
    )
    plot_path = os.path.join(PROJECT_ROOT, "scatterplot_income_default_no.png")
    print(f"Saved plot to {plot_path}")
    plt.savefig(plot_path)
