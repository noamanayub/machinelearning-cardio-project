# Cardiovascular Risk Predictor

This project is a web application for predicting cardiovascular disease risk using a machine learning model. Users can input their health data and receive a risk prediction, as well as view and manage previous prediction results.

---

## Features

- Predict cardiovascular risk based on user input (age, gender, blood pressure, cholesterol, etc.)
- View previous prediction results (last 10)
- Delete selected or all previous records
- Responsive and clean UI
- Data stored in CSV for easy access
- Jupyter Notebook included for model training and exploration

---

## Limitations

- Predictions are only as accurate as the data and model used; not a substitute for professional medical advice.
- Only supports the features present in the form; cannot handle missing or additional data.
- Model and preprocessor files (`cardio_risk_model.pkl`, `cardio_preprocessor.pkl`) must be present in the project root.
- No user authentication; all users share the same prediction history.
- Data is stored in a CSV file, which is not suitable for production or multi-user environments.
- The web app is intended for educational/demo purposes only.

---

## Project Structure

```
cardio_project/
│
├── app/
│   ├── app.py                # Flask application
│   ├── static/
│   │   └── style.css         # CSS styles
│   └── templates/
│       └── index.html        # Main HTML template
│
├── cardio_risk_model.pkl     # Trained ML model
├── cardio_preprocessor.pkl   # Preprocessing pipeline
├── cardio_project.ipynb      # Jupyter Notebook for model training and exploration
├── cardio_train.csv          # Training data (optional)
├── requirements.txt          # Python dependencies

```

---

## Installation

1. **Clone the repository**

   ```sh
   git clone machinelearning-cardio-project
   cd cardio_project
   ```

2. **Install dependencies**

   ```sh
   pip install -r requirements.txt
   ```

3. **Ensure model files exist**

   Place `cardio_risk_model.pkl` and `cardio_preprocessor.pkl` in the project root.

4. **Run the application**

   ```sh
   cd app
   python app.py
   ```

5. **Open in browser**

   Visit [http://localhost:5000](http://localhost:5000)

---

## Running the Jupyter Notebook

1. **Install Jupyter if not already installed**

   ```sh
   pip install notebook
   ```

2. **Start Jupyter Notebook**

   ```sh
   jupyter notebook
   ```

3. **Open and run `cardio_project.ipynb`**  
   Use this notebook to explore data, train, and export your own models.

---

## Usage

- Fill in your health data and submit the form to get a risk prediction.
- View your last 10 predictions in the "Previous Results" panel.
- Use the "Delete Selected" or "Clear All" buttons to manage your records.

---

## License

© 2025 noamanayub | This Project is Open Source and Educational Purpose.