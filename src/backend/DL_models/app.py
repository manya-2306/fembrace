from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import shap
import lime
import lime.lime_tabular
from google import genai
import os

app = Flask(__name__)

CORS(app)  # Enable CORS for all routes


# Configure logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)



# -------------------

# MODEL LOADING (with caching)

# -------------------

models = {}

scaler = None

GEMINI_API_KEY = "AIzaSyCuCj0ZfT7xCNZ0TzUIk51ijJhIgH4tSBY"

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

def load_models():

    global models, scaler

    try:

        # Load your exact scikit-learn models

        models = {

            "lr": joblib.load('lr_model.pkl'),

            "rf": joblib.load("rf_model.pkl"),

            "gbm": joblib.load("gbm_model.pkl"),

            "svm": joblib.load("svm_model.pkl")

        }

        scaler = joblib.load("scaler.pkl")

        logger.info("All models loaded successfully")

    except Exception as e:

        logger.error(f"Error loading models: {str(e)}")

        raise



# Load models at startup

load_models()



# -------------------

# FEATURES CONFIG

# -------------------

features = [

    "Age (yrs)", "Weight (Kg)", "Height(Cm)", "BMI", "Blood Group", "Pulse rate(bpm)",

    "Cycle(R/I)", "Cycle length(days)", "Marraige Status (Yrs)", "Pregnant(Y/N)",

    "No. of aborptions", "Hip(inch)", "Waist(inch)", "Waist:Hip Ratio",

    "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)",

    "Hair loss(Y/N)", "Pimples(Y/N)", "Fast food (Y/N)", "Reg.Exercise(Y/N)"

]



categorical_map = {

    "Pregnant(Y/N)": {"Y": 1, "N": 0},

    "Weight gain(Y/N)": {"Y": 1, "N": 0},

    "hair growth(Y/N)": {"Y": 1, "N": 0},

    "Skin darkening (Y/N)": {"Y": 1, "N": 0},

    "Hair loss(Y/N)": {"Y": 1, "N": 0},

    "Pimples(Y/N)": {"Y": 1, "N": 0},

    "Fast food (Y/N)": {"Y": 1, "N": 0},

    "Reg.Exercise(Y/N)": {"Y": 1, "N": 0}

}



# Pre-computed statistics for better LIME explanations

feature_stats = {

    "means": [25.0, 60.0, 160.0, 23.4, 2.0, 70.0, 1.0, 30.0, 0.0, 0.0,

              0.0, 34.0, 30.0, 0.88, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],

    "stds": [5.0, 10.0, 10.0, 4.0, 1.0, 10.0, 0.5, 5.0, 1.0, 0.5,

             1.0, 3.0, 3.0, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

}



# -------------------

# HELPER FUNCTIONS

# -------------------

def prepare_input(data):

    """Convert raw input data to scaled features"""

    input_data = {}

    for feature in features:

        value = data.get(feature, 0)

        if feature in categorical_map:

            input_data[feature] = [categorical_map[feature].get(value, 0)]

        else:

            input_data[feature] = [float(value)]

    return pd.DataFrame(input_data)

def generate_lime_explanation(scaled_input):
    """Generate LIME explanation with improved sampling and error handling"""
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.random.normal(
                loc=feature_stats["means"],
                scale=feature_stats["stds"],
                size=(1000, len(features))
            ),
            feature_names=features,
            class_names=["No PCOS", "PCOS"],
            discretize_continuous=True,
            mode="classification",
            kernel_width=3,
            verbose=False
        )
        
        exp = explainer.explain_instance(
            scaled_input[0],
            models["rf"].predict_proba,
            num_features=10,
            top_labels=1
        )
        
        # Check which label LIME actually produced to avoid KeyError: 1
        labels = exp.available_labels()
        label_to_use = labels[0] if labels else 1
        
        return {k: float(v) for k, v in exp.as_list(label=label_to_use)}
    except Exception as e:
        logger.error(f"LIME inner error: {str(e)}")
        # Fallback to an empty dict so the rest of the app (SHAP) can still work
        return {}

def generate_shap_explanation(scaled_input):

    """Generate SHAP values using the Random Forest model"""

    explainer = shap.TreeExplainer(models["rf"])

    shap_values = explainer.shap_values(scaled_input)

   

    # Handle different SHAP output formats based on library version

    if isinstance(shap_values, list):

        # For binary classification, use the second element (class 1)

        if len(shap_values) > 1:

            return dict(zip(features, shap_values[1][0].tolist()))

        return dict(zip(features, shap_values[0][0].tolist()))

    else:

        # Array shape handling for newer SHAP versions

        if len(shap_values.shape) == 3:

            return dict(zip(features, shap_values[0, :, 1].tolist()))

        return dict(zip(features, shap_values[0].tolist()))



# -------------------

# ROUTES

# -------------------

@app.route("/")

def index():

    return render_template("index.html")



@app.route("/predict", methods=["POST"])

def predict():

    try:

        data = request.json

        if not data:

            return jsonify({"error": "No input data provided"}), 400



        input_df = prepare_input(data)

        scaled_input = scaler.transform(input_df)



        # Standard scikit-learn probability prediction

        def predict_model(model_name):

            model = models[model_name]

            return float(model.predict_proba(scaled_input)[:, 1][0])



        model_accuracies = {

            "lr": 0.85, "rf": 0.87,

            "gbm": 0.89, "svm": 0.84

        }



        # Run predictions in parallel

        predictions = {}

        with ThreadPoolExecutor() as executor:

            future_to_model = {

                executor.submit(predict_model, model): model

                for model in model_accuracies

            }

            for future in as_completed(future_to_model):

                model = future_to_model[future]

                predictions[model] = future.result()



        # Weighted ensemble calculation

        weighted_sum = sum(

            model_accuracies[model] * pred

            for model, pred in predictions.items()

        )

        final_prediction = weighted_sum / sum(model_accuracies.values())



        return jsonify({

            "probability": float(final_prediction),

            "prediction": "PCOS" if final_prediction > 0.5 else "No PCOS",

            "model_details": {

                "ensemble_method": "weighted_average",

                "weights": model_accuracies,

                "individual_predictions": predictions

            }

        })



    except Exception as e:

        logger.error(f"Prediction error: {str(e)}", exc_info=True)

        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500



@app.route("/explain", methods=["POST"])

def explain():

    try:

        data = request.json

        if not data:

            return jsonify({"error": "No input data provided"}), 400



        input_df = prepare_input(data)

        scaled_input = scaler.transform(input_df)



        # Generate explanations in parallel

        lime_explanation, shap_explanation = None, None

        with ThreadPoolExecutor(max_workers=2) as executor:

            lime_future = executor.submit(generate_lime_explanation, scaled_input)

            shap_future = executor.submit(generate_shap_explanation, scaled_input)

            lime_explanation = lime_future.result()

            shap_explanation = shap_future.result()



        return jsonify({

            "lime_explanation": lime_explanation,

            "shap_explanation": shap_explanation,

            "top_features": sorted(

                lime_explanation.items(),

                key=lambda x: abs(x[1]),

                reverse=True

            )[:5]

        })



    except Exception as e:

        logger.error(f"Explanation error: {str(e)}", exc_info=True)

        return jsonify({

            "error": f"Explanation failed: {str(e)}",

            "partial_explanation": {

                "lime": lime_explanation if 'lime_explanation' in locals() else None

            }

        }), 500



@app.route("/model_info", methods=["GET"])

def model_info():

    """Endpoint to get model metadata"""

    return jsonify({

        "models_loaded": list(models.keys()),

        "features": features,

        "categorical_mappings": categorical_map,

        "last_loaded": str(pd.Timestamp.now())

    })



# -------------------

# MAIN

# -------------------


@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        data = request.json

        user_message = data.get("message", "")
        prediction = data.get("prediction", "Unknown")
        probability = data.get("probability", 0)
        top_features = data.get("top_features", [])
        form_data = data.get("form_data", {})
        explanation = data.get("explanation", {})
        
        prompt = f"""
                You are FemBrace AI, a PCOS health assistant.

                Prediction Result:
                - Diagnosis: {prediction}
                - PCOS Probability: {round(probability * 100, 1)}%

                Patient Health Data:
                {form_data}

                Model Explanation (Important Risk Factors):
                {explanation}

                Top Features:
                {top_features}

                User Question:
                {user_message}

                Instructions:
                - Give personalized PCOS guidance
                - Use prediction + patient data + explanation
                - Do not prescribe medicines
                - Do not claim diagnosis certainty
                - Keep response clear, short, and practical
                - Recommend doctor consultation when needed 
            """
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return jsonify({
            "reply": response.text
        })

    except Exception as e:
        logger.error(f"Chatbot error: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5002, debug=True)