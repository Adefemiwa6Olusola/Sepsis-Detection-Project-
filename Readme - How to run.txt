
Since it was confusing to most people, I laid out this txt file so we can all check out our model before we move on to the presentation.
So this is a simple step by step tutorial on how to test our LSTM model.

1. Step 1 - Download the model and the scaler files using the link below
	Download model & scaler and extract the zip.
	https://drive.google.com/file/d/1N4wPVe_gYIWQx1gjb1eRm4RoauGinGGJ/view?usp=drive_link

2. Go to https://colab.google.com and upload the notebook file "Model_Test_Notebook.ipynb" contained in this folder.
	Alternatively you can download the notebook file:
	https://drive.google.com/file/d/15lEIabTRHXPOd8MpvzEqwzrQvrdeosuW/view?usp=sharing

3. Before running the cells in the notebook on colab, upload the two files downloaded in step 1 to your colab session.
	"sepsis_lstm_model.keras" and "scalers.pkl"
	These files are also located in this folder.

4. Sit back and enjoy. You can run sample data and get output with the code below (Copy and paste into a new colab cell):


	



# Three test patient sequences: Low, Moderate, High risk

# 1️ Low risk (stable)
low_risk_sequence = [
    [80, 120, 16, 36.8, 98],
    [82, 118, 16, 37.0, 97],
    [81, 117, 17, 36.9, 97],
    [83, 116, 16, 37.1, 98],
    [82, 115, 17, 37.0, 97],
    [80, 116, 16, 36.9, 98],
    [81, 115, 16, 37.0, 97],
    [82, 117, 17, 37.1, 97],
    [83, 116, 16, 36.8, 98],
    [81, 115, 17, 37.0, 97]
]

# 2️ Moderate risk (sepsis-like)
moderate_risk_sequence = [
    [120, 70, 30, 39.5, 85],  
    [125, 65, 32, 39.8, 83],
    [130, 60, 34, 40.0, 82],
    [135, 55, 36, 40.2, 80],
    [140, 50, 38, 40.5, 78],
    [145, 45, 40, 40.8, 75],
    [150, 40, 42, 41.0, 73],
    [155, 35, 44, 41.2, 70],
    [160, 30, 46, 41.5, 68],
    [165, 25, 48, 42.0, 65]
]

# 3️ High risk (septic shock)
high_risk_sequence = [
    [140, 60, 35, 40.0, 82],
    [145, 55, 37, 40.5, 80],
    [150, 50, 39, 41.0, 78],
    [155, 45, 41, 41.2, 75],
    [160, 40, 43, 41.5, 73],
    [165, 35, 45, 41.8, 70],
    [170, 30, 47, 42.0, 68],
    [175, 25, 49, 42.3, 65],
    [180, 20, 51, 42.5, 63],
    [185, 15, 55, 43.0, 60]
]

# Run predictions
for label, seq in zip(["Low risk", "Moderate risk", "High risk"], 
                      [low_risk_sequence, moderate_risk_sequence, high_risk_sequence]):
    result = predict_sepsis(seq)
    print(f"{label} sequence prediction: {result}")

