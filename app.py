import os
import warnings

import pandas as pd
from flask import Flask, flash, request, redirect, render_template

import prediction

PEOPLE_FOLDER = os.path.join('static')
ALLOWED_EXTENSIONS = {'excel', 'csv', 'txt'}

warnings.filterwarnings('ignore')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/show', methods=['GET', 'POST'])
def show():
    if request.method == 'GET':
        last_pred_json = prediction.forecastwithoption("MSFT", 30)
        return_json = {
            'data': last_pred_json
        }
        return return_json
    else:
        return  # show_the_login_form()


#
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global dataframe
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            dataframe = pd.read_csv(request.files['file'], header=0, index_col=0)
            file = request.files['file']
            filename = file.filename
            print('{} isimli dosya yuklendi'.format(filename))
            print(dataframe)
            dataframe = dataframe.dropna()
            prediction.forecastwithuploadcsv(dataframe, 30)
            # FOR PDF
            # rendered = render_template('index.html', result=fc_series, user_image=full_filename)
            # pdf = pdfkit.from_string(rendered,False)
            # response = make_response(pdf)
            # response.headers['Content-Type'] = 'application/pdf'
            # response.headers['Content-Disposition'] = 'attachment; filename=output.pdf'
            # return response


def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()
