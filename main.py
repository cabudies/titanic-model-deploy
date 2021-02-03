from flask import Flask
import joblib

app = Flask(__name__)

model = joblib.load('titanic_model.sav')

@app.route("/")
def index():
    passengerId = 8
    age = 5
    gender = 0 # 1->female, 0->male
    prediction = model.predict([[passengerId, age, gender]])
    return "Person's prediction is: {}".format(str(prediction[0]))

if __name__ == "__main__":
    app.run()



