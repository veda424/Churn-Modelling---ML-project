import numpy as np
from flask import Flask,render_template,request
app = Flask(__name__)
import pickle


file = open('model.pkl','rb')
lr = pickle.load(file)
file.close()

file1 = open('sc.pkl','rb')
sc = pickle.load(file1)
file1.close()
@app.route('/' , methods=["GET","POST"])
def home():
    if request.method == "POST":
        creditscore = request.form.get('creditscore')
        age = request.form.get('age')
        tenure = request.form.get('tenure')
        balance = request.form.get('balance')
        estimatedsalary = request.form.get('estimatedsalary')
        numofproducts = request.form.get('numofproducts')
        hascrcard = request.form.get('hascrcard')
        isactivemember = request.form.get('isactivemember')
        features = [creditscore, age, tenure , balance, numofproducts,hascrcard,isactivemember,estimatedsalary]
        features = np.array(features).reshape(1,-1)
        features = sc.transform(features)
        pred = lr.predict_proba(features)
        prob = pred[0][0]
        ans = prob*100
        ans = round(ans,2)
        return render_template('show.html' , pred = ans)      
    return render_template('home.html')
@app.route('/about')
def about():
    return render_template('about.html')
if __name__ == '__main__':
    app.run(debug=True)