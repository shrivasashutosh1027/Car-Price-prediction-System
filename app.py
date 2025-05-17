import pandas as pd
import pickle
import streamlit as st 

#Load Model
with open ("Car_price_prediction_model.pkl","rb") as f:
    model = pickle.load(f)


st.sidebar.title("Navigation")#Side bar
page = st.sidebar.radio("Go to",["Introduction","Prediction"])
df=pd.read_csv("car data.csv")
#Introductiom Page
if page=="Introduction":
    st.title("Car Price Prediction Machine Learning Model") 
    st.header("Welcome to the Car Price Prediction Model")
    st.write("We have developed a Car Price Prediction model that helps you make informed decisions when purchasing a car.")
    
#Prediction Page 
elif page == "Prediction":
    st.header("Car Prediction")
    st.write("Predict")
    # def get_brand_name(Car_Name):
    #     Car_Name = Car_Name.split('')[0]
    #     return Car_Name.strip()
    # df['Car_Name'] = df['Car_Name'].apply(get_brand_name)

    #Input Fields
    Car_Name=st.selectbox("Select Car Brand",df['Car_Name'].unique())
    Year=st.slider('Car Manufactured Year', 2003,2018)
    Driven_kms=st.slider('Number of Kms Driven', 500,500000)
    Fuel_Type=st.selectbox('Fuel Type', df["Fuel_Type"].unique())
    Selling_type=st.selectbox('Seller Type', df["Selling_type"].unique())
    Transmission=st.selectbox("Transminssion Type",df['Transmission'].unique())
    Owner=st.selectbox("Owners",df['Owner'].unique())
    Present_Price=st.selectbox("Present_Price",df['Present_Price'].unique())

    #Predicted Button
    if st.button("Predict Price"):
        df=pd.DataFrame(
        [[Car_Name,Year,Present_Price,Driven_kms,Fuel_Type,Selling_type,Transmission,Owner]],
        columns=['Car_Name', 'Year','Present_Price','Driven_kms', 'Fuel_Type', 'Selling_type', 'Transmission', 'Owner'])
        df['Fuel_Type'].replace(['Diesel','Petrol','CNG'],[0,1,2],inplace=True)
        df['Selling_type'].replace(['Dealer','Individual'],[0,1],inplace=True)
        df['Transmission'].replace(['Manual','Automatic'],[0,1],inplace=True)
        df['Car_Name'].replace(['ritz', 'sx4', 'ciaz', 'wagon r', 'swift', 'vitara brezza',
                                's cross', 'alto 800', 'ertiga', 'dzire', 'alto k10', 'ignis',
                                '800', 'baleno', 'omni', 'fortuner', 'innova', 'corolla altis',
                                'etios cross', 'etios g', 'etios liva', 'corolla', 'etios gd',
                                'camry', 'land cruiser', 'Royal Enfield Thunder 500',
                                'UM Renegade Mojave', 'KTM RC200', 'Bajaj Dominar 400', 
                                'Royal Enfield Classic 350', 'KTM RC390', 'Hyosung GT250R',
                                'Royal Enfield Thunder 350', 'KTM 390 Duke ',
                                'Mahindra Mojo XT300', 'Bajaj Pulsar RS200',
                                'Royal Enfield Bullet 350', 'Royal Enfield Classic 500',
                                'Bajaj Avenger 220', 'Bajaj Avenger 150', 'Honda CB Hornet 160R',
                                'Yamaha FZ S V 2.0', 'Yamaha FZ 16', 'TVS Apache RTR 160',
                                'Bajaj Pulsar 150', 'Honda CBR 150', 'Hero Extreme',
                                'Bajaj Avenger 220 dtsi', 'Bajaj Avenger 150 street',
                                'Yamaha FZ  v 2.0', 'Bajaj Pulsar  NS 200', 'Bajaj Pulsar 220 F',
                                'TVS Apache RTR 180', 'Hero Passion X pro', 'Bajaj Pulsar NS 200',
                                'Yamaha Fazer ', 'Honda Activa 4G', 'TVS Sport ',
                                'Honda Dream Yuga ', 'Bajaj Avenger Street 220',
                                'Hero Splender iSmart', 'Activa 3g', 'Hero Passion Pro',
                                'Honda CB Trigger', 'Yamaha FZ S ', 'Bajaj Pulsar 135 LS',
                                'Activa 4g', 'Honda CB Unicorn', 'Hero Honda CBZ extreme',
                                'Honda Karizma', 'Honda Activa 125', 'TVS Jupyter',
                                'Hero Honda Passion Pro', 'Hero Splender Plus', 'Honda CB Shine',
                                'Bajaj Discover 100', 'Suzuki Access 125', 'TVS Wego',
                                'Honda CB twister', 'Hero Glamour', 'Hero Super Splendor',
                                'Bajaj Discover 125', 'Hero Hunk', 'Hero  Ignitor Disc',
                                'Hero  CBZ Xtreme', 'Bajaj  ct 100', 'i20', 'grand i10', 'i10',
                                'eon', 'xcent', 'elantra', 'creta', 'verna', 'city', 'brio',
                                'amaze', 'jazz'],[0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10,                                                
                                                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                  31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                                  41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                                  51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                                  61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                                  71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                                  81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                                  91, 92, 93, 94,95,96,97],inplace=True)
        

        Car_Price = model.predict(df)

        st.markdown("Car Price" + " :- " + str(Car_Price[0])+"Lakhs")