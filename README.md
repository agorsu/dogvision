# Dog Vision
Dog Vision is a simple and fun Machine Learning web application that tries to determine the dog breed from an uploaded dog photo.

## Description
Dog Vision was a small Machine Learning project built to help understand multi-class image classifier models using TensorFlow 2.x and TensorFlow Hub.
The model was trained on thousands of different dog images to learn to identify between 120 different dog breed types.
The data used in training the model is available from [Kaggle](https://www.kaggle.com/c/dog-breed-identification/overview)


## Getting Started

### Dependencies
* TensorFlow 2.x
* TensorFlow Hub
* StreamLit
* Pillow
* NumPy

## How to setup
Firstly download or clone this repo and then install the requirements;

1. Clone the repo
`git clone https://github.com/agorsu/dogvision.git`

2. Navigate to the project folder
`cd dogvision`

3. Install the requirements
`pip install -r requirements.txt`

4. Run from Streamlit
`streamlit run SL_Dogvision.py`

### Uploading Photo
Drag and Drop your doggo jpg's and the app will display the image and show the breed and % confidence score.


Give it a star :tada:
---------------------
Did you find this information useful, then give it a star 
