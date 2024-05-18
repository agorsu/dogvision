import streamlit as st
from PIL import Image
import tempfile, os

from util import classify, load_model

#page config
st.set_page_config(page_title='Dog Vision')
st.title('Dog Vision')

#upload image
uploaded_file = st.file_uploader('Please upload your dog image', type='jpg')

#load classifier
try:
    model = load_model('model/20230930-Dogvision.h5')
except Exception as e:
    tempfolder = os.path.join(tempfile.gettempdir(), "tfhub_modules")
    st.error(f'Error attempting to load model. Try clearing files in {tempfolder} and click re-run.')


#display image
with st.spinner("Loading...."):
    if uploaded_file != None:
        #classify image
        image = Image.open(uploaded_file).convert('RGB')
        class_name, conf_score = classify(image, model)

        # write classification
        st.write(f"## {class_name.title()}")
        st.progress(conf_score*0.01, text=f"### confidence {int(conf_score)}%")
        st.image(image, use_column_width=True)
