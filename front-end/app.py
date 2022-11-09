from flask import *

import shap
from shap.plots._force_matplotlib import draw_additive_plot


from model import give_shap_plot


app = Flask(__name__, template_folder='templates')




@app.route('/dashboard')

def displayshap():

    explainer, shap_values, X_train = give_shap_plot()

    def _force_plot_html(explainer, shap_values, ind,X_train,i):
        # force_plot = shap.plots.force(shap_values[ind], matplotlib=False)
        view_id = i
        force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1][ind, :], X_train.iloc[view_id, :],
                        plot_cmap=["#9932CC", "#FFD700"])

        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        return shap_html


    def _force_plot_group_html(explainer, shap_values, ind, X_train):
        F_plot = shap.force_plot(explainer.expected_value[1], shap_values[1][:500, :], X_train.iloc[:500, :],
                    plot_cmap=["#9932CC", "#FFD700"])
        shap_html = f"<head>{shap.getjs()}</head><body>{F_plot.html()}</body>"
        return shap_html

    shap_plots = {}
    for i in [72936]:
        ind = i
        shap_plots[i] = _force_plot_html(explainer, shap_values, ind, X_train,i)

    shap_plots[100] = _force_plot_group_html(explainer, shap_values, ind, X_train)


    return render_template('displayshap.html', shap_plots = shap_plots)








@app.route('/model_training')
def model_training():
    return render_template('model_training.html', message='model_training')

# @app.route('/raw_data')
# def raw_data():
#     return render_template('raw_data.html', message='raw_data')


@app.route('/model_explanation')
def model_explanation():
    return render_template('model_explanation.html', message='model_explanation')


@app.route('/predictive')
def predictive():
    return render_template('predictive.html', message='predictive')


@app.route('/')
def dashboard():
    return render_template('dashboard.html', message='dashboard')





if __name__ == '__main__':
    app.run(debug = True)