# https://github.com/HorusAlkebulan/applied-machine-learning-algorithms-3806104/blob/main/algos.ipynb

from sklearn.tree import DecisionTreeRegressor, plot_tree
import numpy as np

if __name__ == "__main__":

    script_name = "ch05_decision_trees"
    anscombe = None

    dt = DecisionTreeRegressor(max_depth=1)
    X = anscombe[["x"]]
    y = anscombe["y1"]
    dt.fit(X, y)
