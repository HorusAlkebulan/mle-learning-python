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

    fig, axs = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(11, 8.5)
    fig.subplots_adjust(hspace=0.25, wspace=0.25)

    sns.boxplot(
        loan_data_df,
        x="Default",
        y="Income",
        ax=axs[0, 0],
    )
    axs[0, 0].set_title("boxplot_loan_default_income")

    # plot_path = os.path.join(PROJECT_ROOT, "boxplot_loan_default_income.png")
    # print(f"Saved plot to {plot_path}")
    # plt.savefig(plot_path)
    
    sns.boxplot(
        loan_data_df,
        x="Default",
        y="Loan Amount",
        ax=axs[0, 1],
    )
    # plot_path = os.path.join(PROJECT_ROOT, "boxplot_loan_default_loan_amount.png")
    # print(f"Saved plot to {plot_path}")
    # plt.savefig(plot_path)
    axs[0, 1].set_title("boxplot_loan_default_loan_amount")

    sns.scatterplot(
        x=loan_data_df["Income"],
        y=np.where(loan_data_df["Default"]=="No", 0, 1),
        s=150,
        ax=axs[1, 0],
    )
    axs[1, 0].set_title("scatterplot_income_default_no")

    plot_filename = "ch03_classifying_data_plots.png"
    plt.suptitle(plot_filename)

    plot_path = os.path.join(PROJECT_ROOT, plot_filename)
    print(f"Saving plot to {plot_path}")
    plt.savefig(plot_path)