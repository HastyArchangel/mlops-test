from flask import Flask, request, jsonify
import pandas as pd
import joblib 
from sklearn.preprocessing import LabelEncoder

model = joblib.load("best_random_forest_model.pkl")
encoders = joblib.load("encoders.pkl")
categorical_features = ['international_plan', 'voice_mail_plan']

app = Flask(__name__)

@app.route('/api', methods=['GET'])
def predict():
    try:
        data = request.args.to_dict()

        expected_features = [
            'international_plan',
            'voice_mail_plan',
            'total_day_minutes',
            'total_eve_minutes',
            'total_intl_minutes',
            'number_customer_service_calls'
        ]
        
        missing_features = []
        for feature in expected_features:
            if feature not in data:
                missing_features.append(feature)
                print(f'feature : {feature}')
        if len(missing_features) > 0:
            error_message = 'Missing required features: '
            error_message.join(missing_features)
            return jsonify({'Parameter error': error_message}), 400


        for feature in categorical_features:
            data[feature] = encoders[feature].transform([data[feature]])[0]


        input_data = pd.DataFrame(
            [{feature: float(data[feature]) for feature in expected_features}],
            columns=expected_features
        )

        prediction = model.predict(input_data)

        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
