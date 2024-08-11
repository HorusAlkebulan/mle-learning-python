import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt 

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