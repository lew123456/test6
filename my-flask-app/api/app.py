from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# 加载模型
with open('api/model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('templates/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    # 将表单数据转换为模型输入的格式
    features = np.array([[
        float(data['Pregnancies']),
        float(data['Glucose']),
        float(data['BloodPressure']),
        float(data['SkinThickness']),
        float(data['Insulin']),
        float(data['BMI']),
        float(data['DiabetesPedigreeFunction']),
        float(data['Age'])
    ]])
    
    print("Input features:", features)
    
    # 使用模型进行预测
    prediction = model.predict(features)
    
    # 返回结果，并将 numpy.int64 转换为 int
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
