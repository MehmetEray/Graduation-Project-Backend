import os
import warnings

import pandas as pd
from flask import Flask, flash, request, redirect

import prediction

PEOPLE_FOLDER = os.path.join('static')
ALLOWED_EXTENSIONS = {'excel', 'csv', 'txt'}

warnings.filterwarnings('ignore')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return 'Graduation Project Backend!'


@app.route('/all/<int:number>', methods=['GET', "POST"])
def showAll(number):
    if request.method == 'GET':

        last_pred_json_microsoft = prediction.forecastwithoption("MSFT", number)
        last_pred_json_apple = prediction.forecastwithoption("AAPL", number)
        last_pred_json_amazon = prediction.forecastwithoption("AMZN", number)
        return_json = {
            'microsoft': last_pred_json_microsoft,
            'apple': last_pred_json_apple,
            'amazon': last_pred_json_amazon
        }
        return return_json

    else:
        return  # show_the_login_form()


@app.route('/microsoft/<int:number>', methods=['GET', "POST"])
def showMSFT(number):
    if request.method == 'GET':

        last_pred_json_microsoft = prediction.forecastwithoption("MSFT", number)
        return_json = {
            'microsoft': last_pred_json_microsoft
        }
        return return_json

    else:
        return  # show_the_login_form()


@app.route('/apple/<int:number>', methods=['GET', "POST"])
def showAAPL(number):
    if request.method == 'GET':

        last_pred_json_apple = prediction.forecastwithoption("AAPL", number)
        return_json = {
            'apple': last_pred_json_apple
        }
        return return_json

    else:
        return  # show_the_login_form()


@app.route('/amazon/<int:number>', methods=['GET', "POST"])
def showAMZN(number):
    if request.method == 'GET':

        last_pred_json_amazon = prediction.forecastwithoption("AMZN", number)
        return_json = {
            'amazon': last_pred_json_amazon
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
