# Chennai-house-price-prediction
This project aims to develop a machine learning model to predict house prices in Chennai, India based on various property features.

Key Features

Predicts the sales price of a house given its attributes like location, area, number of bedrooms, bathrooms, etc.
Provides an intuitive web-based interface for users to input property details and get the predicted price.
Utilizes a linear regression model trained on a real-world dataset of Chennai house sales.

Dataset
The dataset used in this project is based on real estate transactions in Chennai. It includes the following features:

PRT_ID: Unique property identifier
AREA: Location/area of the property
INT_SQFT: Interior square footage of the property
DATE_SALE: Date of sale
DIST_MAINROAD: Distance from main road
N_BEDROOM: Number of bedrooms
N_BATHROOM: Number of bathrooms
N_ROOM: Number of rooms
SALE_COND: Sale condition
PARK_FACIL: Parking facility
DATE_BUILD: Year of construction
BUILDTYPE: Type of building
UTILITY_AVAIL: Utilities available
STREET: Street type
MZZONE: Market zone
QS_ROOMS: Quality of rooms
QS_BATHROOM: Quality of bathrooms
QS_BEDROOM: Quality of bedrooms
QS_OVERALL: Overall quality
REG_FEE: Registration fee
COMMIS: Commission
SALES_PRICE: Sale price of the property

The raw dataset is available in the chennai.csv file in this repository. Feel free to explore and manipulate the data as per your requirements.
Usage

Ensure you have Python 3.x and the required dependencies installed (see requirements.txt).
Run the Jupyter Notebook main.ipynb to train the model and save it as model.pkl.
Run the Flask application app.py to start the web interface.
Access the web app at http://localhost:5000 and input the property details to get the predicted price.

Contributing
Contributions are welcome! If you find any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request.

License
This project is licensed under the MIT License.
