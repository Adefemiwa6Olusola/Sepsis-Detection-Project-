This was out first model. A simple snapshot model based off of one set of vitals - No  time series or progression.
It was SANE but NOT PREDICTIVE ENOUGH cause it COULD NOT learn trends.
This was deployed locally using uvicorn and fastAPI.
These requirements are needed locally before running the model
	uvicorn
	fastAPI
	Python 3.14
	pandas
	numpy
	torch

Then cmd can be opened in the folder and run the code 
	uvicorn app:app --reload

This would start the localhost server 
The model can be opened on this url : http://127.0.0.1:8000/docs#/