import numpy as np
import plotly.express as px
import os
from summary_app.model import Summary
from flask import url_for
from summary_app.config import Config

def flatten_to_floats(data):
    if data is None:
        return []
    if isinstance(data, (float, int, np.integer)):
        return [float(data)]
    elif isinstance(data, (list, tuple, np.ndarray)):
        return [item for sublist in map(flatten_to_floats, data) for item in sublist]
    else:
        return []

def make_plot(summ):
    data_for_boxplot = {"Feature": [], "Value": []}
    for feature in summ.corr_list:
        if feature in summ.output_dir_correlations.get('pearson', {}).keys():
            values = flatten_to_floats(summ.output_dir_correlations['pearson'].get(feature, []))
            data_for_boxplot["Feature"].extend([feature] * len(values))
            data_for_boxplot["Value"].extend(values)
    data_for_boxplot["Feature"] = data_for_boxplot["Feature"][::-1]
    fig_selected_boxplot = px.box(data_for_boxplot, x="Value", y="Feature", orientation="h",
                  title="Selected Plot of Pearson Correlation Values")
    min_height = 100
    feature_count = len(set(data_for_boxplot["Feature"]))
    calculated_height = min_height + feature_count * 40
    fig_selected_boxplot.update_layout(height=calculated_height)

    plot_name = 'selected_boxplot.html'
    saved_plot_path = os.path.join(Config.PLOT_PATH, plot_name)
    fig_selected_boxplot.write_html(saved_plot_path)
    print(f'Saved at: {saved_plot_path}')
    return plot_name

def make_all_boxplot(summ):
    all_boxplot = {"Feature": [], "Value": []}
    for feature in summ.output_dir_correlations.get('pearson', {}).keys():
        values = flatten_to_floats(summ.output_dir_correlations['pearson'].get(feature, []))
        all_boxplot["Feature"].extend([feature] * len(values))
        all_boxplot["Value"].extend(values)
    all_boxplot["Feature"] = all_boxplot["Feature"][::-1]
    fig_all_boxplot = px.box(all_boxplot, x="Value", y="Feature", orientation="h",
                  title="All Matched Path Pearson Correlation Values")
    min_height = 100
    feature_count = len(set(all_boxplot["Feature"]))
    calculated_height = min_height + feature_count * 40
    fig_all_boxplot.update_layout(height=calculated_height)

    plot_name = 'all_boxplot.html'
    saved_plot_path = os.path.join(Config.PLOT_PATH, plot_name)
    fig_all_boxplot.write_html(saved_plot_path)
    return plot_name


