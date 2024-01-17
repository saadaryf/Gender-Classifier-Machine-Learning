from flask import Flask
from app import views
app = Flask(__name__) #webserver gateway interphase WSGI

# @app.route('/')
# def index():
#     return "Welcome to face recognition app"

app.add_url_rule(rule='/',endpoint='home',view_func=views.index)
app.add_url_rule(rule='/app',endpoint='app',view_func=views.app)
app.add_url_rule(rule='/app/gender',endpoint='gender',view_func=views.gender_app,
                 methods=['GET','POST'])

app.add_url_rule(rule='/video',endpoint='video',view_func=views.video)



if __name__ == "__main__":
    app.run(debug=True)