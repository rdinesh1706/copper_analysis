import streamlit as st
import pickle
import numpy as np
import sklearn
from streamlit_option_menu import option_menu


def predict_status(country, item_type, application, width, product_ref, quantity_tons_log, customer_log, thickness_log, selling_price_log, item_date_day, item_date_month, item_date_year, delivery_date_day, delivery_date_month, delivery_date_year):

    itdd = int(item_date_day)
    itdm = int(item_date_month)
    itdy = int(item_date_year)

    dydd = int(delivery_date_day)
    dydm = int(delivery_date_month)
    dydy = int(delivery_date_year)

    with open("C:/Users/rdine/Data_Science/data_science_practise/DataScience_projects/copper/classification_model.pkl", "rb") as f:
        model_class = pickle.load(f)

    user_data = np.array([[country, item_type, application, width, product_ref, quantity_tons_log, customer_log, thickness_log,
                           selling_price_log, itdd, itdm, itdy, dydd, dydm, dydy]])

    y_pred = model_class.predict(user_data)

    if y_pred == 1:
        return 1
    else:
        return 0


def predict_selling_price(country, status, item_type, application, width, product_ref, quantity_tons_log, customer_log,
                          thickness_log, item_date_day, item_date_month, item_date_year, delivery_date_day, delivery_date_month, delivery_date_year):

    itdd = int(item_date_day)
    itdm = int(item_date_month)
    itdy = int(item_date_year)

    dydd = int(delivery_date_day)
    dydm = int(delivery_date_month)
    dydy = int(delivery_date_year)

    with open("C:/Users/rdine/Data_Science/data_science_practise/DataScience_projects/copper/Regression_Model.pkl", "rb") as f:
        model_regg = pickle.load(f)

    user_data = np.array([[country, status, item_type, application, width, product_ref, quantity_tons_log, customer_log, thickness_log,
                           itdd, itdm, itdy, dydd, dydm, dydy]])

    y_pred = model_regg.predict(user_data)

    ac_y_pred = np.exp(y_pred[0])

    return ac_y_pred


with st.sidebar:
    option = option_menu(
        'dinesh', options=["Predict selling price", "Predict status"])

if option == "Predict selling price":
    st.header("Predict selling price")
    st.write(" ")

    col1, col2 = st.columns(2)

    with col1:
        country = st.number_input(
            label="**Enter the Value for COUNTRY**/ Min:25.0, Max:113.0")
        status = st.number_input(
            label="**Enter the Value for STATUS**/ Min:0.0, Max:8.0")
        item_type = st.number_input(
            label="**Enter the Value for ITEM TYPE**/ Min:0.0, Max:6.0")
        application = st.number_input(
            label="**Enter the Value for APPLICATION**/ Min:2.0, Max:87.5")
        width = st.number_input(
            label="**Enter the Value for WIDTH**/ Min:700.0, Max:1980.0")
        product_ref = st.number_input(
            label="**Enter the Value for PRODUCT_REF**/ Min:611728, Max:1722207579")
        quantity_tons_log = st.number_input(
            label="**Enter the Value for QUANTITY_TONS (Log Value)**/ Min:-0.3223343801166147, Max:6.924734324081348", format="%0.15f")
        customer_log = st.number_input(
            label="**Enter the Value for CUSTOMER (Log Value)**/ Min:17.21910565821408, Max:17.230155364880137", format="%0.15f")

    with col2:
        thickness_log = st.number_input(
            label="**Enter the Value for THICKNESS (Log Value)**/ Min:-1.7147984280919266, Max:3.281543137578373", format="%0.15f")
        item_date_day = st.selectbox("**Select the Day for ITEM DATE**", ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
                                                                          "12", "13", "14", "15", "16", "17", "18", "19", "20",
                                                                          "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"))

        item_date_month = st.selectbox("**Select the Month for ITEM DATE**",
                                       ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"))

        item_date_year = st.selectbox(
            "**Select the Year for ITEM DATE**", ("2020", "2021"))

        delivery_date_day = st.selectbox("**Select the Day for DELIVERY DATE**", ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                                         "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"))

        delivery_date_month = st.selectbox(
            "**Select the Month for DELIVERY DATE**", ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"))

        delivery_date_year = st.selectbox(
            "**Select the Year for DELIVERY DATE**", ("2020", "2021", "2022"))

    button = st.button(
        ":red[***Predict the selling price]", use_container_width=True)

    if button:
        price = predict_selling_price(country, status, item_type, application, width, product_ref, quantity_tons_log,
                                      customer_log, thickness_log, item_date_day,
                                      item_date_month, item_date_year, delivery_date_day, delivery_date_month,
                                      delivery_date_year)

        st.write("selling price :", price)


if option == "Predict status":

    st.header("Predict the Status won or loss")
    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        country = st.number_input(
            label="**Enter the Value for COUNTRY**/ Min:25.0, Max:113.0")
        item_type = st.number_input(
            label="**Enter the Value for ITEM TYPE**/ Min:0.0, Max:6.0")

        application = st.number_input(
            label="**Enter the Value for APPLICATION**/ Min:2.0, Max:87.5")

        width = st.number_input(
            label="**Enter the Value for WIDTH**/ Min:700.0, Max:1980.0")

        product_ref = st.number_input(
            label="**Enter the Value for PRODUCT_REF**/ Min:611728, Max:1722207579")

        quantity_tons_log = st.number_input(
            label="**Enter the Value for QUANTITY_TONS (Log Value)**/ Min:-0.322, Max:6.924", format="%0.15f")

        customer_log = st.number_input(
            label="**Enter the Value for CUSTOMER (Log Value)**/ Min:17.21910, Max:17.23015", format="%0.15f")

        thickness_log = st.number_input(
            label="**Enter the Value for THICKNESS (Log Value)**/ Min:-1.71479, Max:3.28154", format="%0.15f")

    with col2:

        selling_price_log = st.number_input(
            label="**Enter the Value for SELLING PRICE (Log Value)**/ Min:5.97503, Max:7.39036", format="%0.15f")

        item_date_day = st.selectbox("**Select the Day for ITEM DATE**", ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
                                     "12", "13", "14", "15", "16", "17", "18", "19", "20",
                                                                          "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"))

        item_date_month = st.selectbox("**Select the Month for ITEM DATE**",
                                       ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"))

        item_date_year = st.selectbox(
            "**Select the Year for ITEM DATE**", ("2020", "2021"))

        delivery_date_day = st.selectbox("**Select the Day for DELIVERY DATE**", ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                                         "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"))

        delivery_date_month = st.selectbox(
            "**Select the Month for DELIVERY DATE**", ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"))

        delivery_date_year = st.selectbox(
            "**Select the Year for DELIVERY DATE**", ("2020", "2021", "2022"))

    button = st.button(
        ":red[***Predict the Status]", use_container_width=True)

    if button:
        status = predict_status(country, item_type, application, width, product_ref, quantity_tons_log,
                                customer_log, thickness_log, selling_price_log, item_date_day,
                                item_date_month, item_date_year, delivery_date_day, delivery_date_month,
                                delivery_date_year)

        if status == 1:
            st.header(":green[Won]")

        else:
            st.header(":red[Lose]")
