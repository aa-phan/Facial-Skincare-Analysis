from flask import Flask, render_template, request, url_for, redirect
import os
from product_selection import product_select
from llm_analysis import analyze_pareto_products

app = Flask(__name__)

uploadfolder = 'static/uploads'
app.config['uploadfolder'] = uploadfolder
app.config['types'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['types']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'photo' not in request.files:
        return redirect(url_for('quiz', error='No file part'))
    
    file = request.files['photo']

    if file.filename == '':
        return redirect(url_for('quiz', error='No selected file'))
    
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['uploadfolder'], filename)

    file.save(filepath)
    return redirect(url_for('quiz', filename=filename))

@app.route('/select_product', methods=['POST'])
def product():
    location = request.form.get('location')
    age = request.form.get('age')
    gender = request.form.get('gender')
    race = request.form.get('race')
    skin_sensitivity = request.form.get('skin_sensitivity')
    price = request.form.get('price')
    ingredients = request.form.get('desired_products')
    price = float(price)

    pareto_results = product_select(
        ingredients=ingredients,
        max_price=price,
        location=location,
        age=age,
        gender=gender,
        race=race,
        skin_sensitivity=skin_sensitivity,
        analyze_with_llm=True 
    )

    ai_response = analyze_pareto_products(pareto_results)

    return render_template('quiz.html', ai_response=ai_response)

if __name__ == '__main__':
    app.run(debug=True)
