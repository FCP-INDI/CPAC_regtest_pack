
from flask import Blueprint
from flask import render_template, send_from_directory
import os
from summary_app.main.utils import make_plot, make_all_boxplot
from summary_app.model import Summary
from summary_app.config import Config

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('home.htm')

@main.route('/selected_boxplot')
def serve_selected_boxplot():
    summ = Summary(Config.CORR_DIR)
    plot = make_plot(summ)
    return send_from_directory(directory=Config.PLOT_PATH, path=plot)

@main.route('/all_boxplot')
def serve_all_boxplot():
    summ = Summary(Config.CORR_DIR)
    plot = make_all_boxplot(summ)
    return send_from_directory(directory=Config.PLOT_PATH, path=plot)