# ch03_linear_regression.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_NAME = "ch03_linear_regression"
PROJECT_ROOT = os.path.dirname(__file__)

def get_anscombe_quartet_df() -> pd.DataFrame:

    # Load anscombe's quartet
    x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
    y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
    y3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
    x4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
    y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]

    df_dict = {
        "x": x,
        "y1": y1,
        "y2": y2,
        "y3": y3,
        "x4": x4,
        "y4": y4,
    }
    anscombe = pd.DataFrame(df_dict)
    return anscombe

if __name__ == "__main__":

    # plot x y1
    fig, ax = plt.subplots(figsize=(12, 8))
    anscombe = get_anscombe_quartet_df()
    anscombe.plot.scatter(x="x", y="y1", ax=ax, color="k")
    ax.set_title("anscombe: X vs. y1")

    # compute slope
    x1 = anscombe["x"]
    y1 = anscombe["y1"]
    slope_numerator = ((x1 - x1.mean()) * (y1 - y1.mean())).sum()
    slope_denominator = ((x1 - x1.mean())**2).sum()
    slope = slope_numerator / slope_denominator
    print(f"x vs. y1 slope: {slope:0.2f}")

    # compute intercept
    # y = mx + b
    intercept = y1.mean() - (slope * x1.mean())
    print(f"x vs. y1 intercept: {intercept:0.2f}")

    # add line to plot
    x1 = np.linspace(4, 14, 100)
    y1 = slope * x1 + intercept
    ax.plot(x1, y1, color="r")

    # save plot
    fig_path = os.path.join(PROJECT_ROOT, SCRIPT_NAME + ".png")
    print(f"Saving plot fig to {fig_path}")
    fig.savefig(fig_path)
    # plt.show()