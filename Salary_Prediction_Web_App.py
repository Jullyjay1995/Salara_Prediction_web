import pickle
import sklearn
import numpy as np
import streamlit as st
# To Display Images
from PIL import Image

# loading the saved model
loaded_model = pickle.load(open('trained_salary_model.sav', 'rb'))


# creating a function for Prediction

def salary_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    return f'Your expected salary is {int(prediction)}'


def main():
    # display image
    img = Image.open("salary.jpeg")
    new_image = img.resize((700, 200))
    st.image(new_image)
    # let's display
    # st.image(img, width=700)

    # giving a title
    st.title('Salary Prediction Web App')

    # getting the input data from the user

    Age = st.text_input('Please Input Your Age')
    Gender = st.text_input('Gender: Input 1 - Male or 0 - Female')
    Education_Level = st.text_input('Eductaion Level: Input 0 - Bachelor or 1 - Masters - 2 - PhD')
    Years_of_Experience = st.text_input('Please Input Your Year of Experience')

    # code for Prediction
    salary = ''

    # creating a button for Prediction

    if st.button('Your Expected Salary'):
        salary = salary_prediction(
            [Age, Gender, Education_Level, Years_of_Experience])

    st.success(salary)


if __name__ == '__main__':
    main()
