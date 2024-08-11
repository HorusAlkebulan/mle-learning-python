import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = os.path.dirname(__file__)
HR = "----------------------------------------------------------------------------------"

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

    sns.scatterplot(
        x=loan_data_df["Loan Amount"],
        y=np.where(loan_data_df["Default"] == "No", 0, 1),
        s=150,
        ax=axs[1, 1]
    )
    axs[1, 1].set_title("scatterplot_loan_amount_default_no")

    plot_filename = "ch03_classifying_data_plots.png"
    plt.suptitle(plot_filename)

    plot_path = os.path.join(PROJECT_ROOT, plot_filename)
    print(f"Saving plot to {plot_path}")
    plt.savefig(plot_path)

    # 3. Prepare the Data

    y = loan_data_df["Default"]
    X = loan_data_df[["Income", "Loan Amount"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=0.7,
        stratify=y,
        random_state=123,
    )
    print(f"X_train.shape: {X_train.shape}")
    print(f"y_train.shape: {y_train.shape}")
    print(f"X_test.shape: {X_test.shape}")
    print(f"y_test.shape: {y_test.shape}")

    # 4. Train and evaluate model

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    pred = classifier.predict(X_test)
    print(HR)
    print(f"Test data for evaluation:\n{X_test}")
    print(f"Predictions from trained model:\n{pred}")

    accuracy = classifier.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")

    conf_mtx = confusion_matrix(y_test, pred)
    print(f"conf_mtx: {conf_mtx}")
    print(f"TN, FP\nFN, TP")

    log_coefs = classifier.coef_
    log_intercept = classifier.intercept_
    print(HR)
    print(f"log_intercept: {log_intercept}, log_coefs: {log_coefs}")

    log_odds = np.round(log_coefs[0], 2)
    log_odds_df = pd.DataFrame({"log_odds": log_odds}, index = X.columns)
    print(HR)
    print(f"log_odds_df:\n{log_odds_df}")

    odds = np.exp(log_odds)
    odds_df = pd.DataFrame({"odds": odds}, index=X.columns)
    print(HR)
    print(f"odds_df:\n{odds_df}")
