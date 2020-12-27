from flask import Flask,request
import prediction

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return #do_the_login()
    else:
        return #show_the_login_form()

if __name__ == '__main__':
    app.run()
