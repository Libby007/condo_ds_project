import base64
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu

import gspread
from df2gspread import df2gspread as d2g
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient import discovery
from itertools import chain

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import json
import random
from streamlit_chat import message
nltk.download('punkt')
nltk.download("wordnet") 
nltk.download('omw-1.4')
# menu
with st.sidebar:
    choose = option_menu("DS Project",
                         ["About", "Toronto condo price prediction", "Condo clustering", "Condo search engine"
                             , "Multi-criteria ranked condos", "Talk to AI Libby", "Algorithm theory and principle"],
                         icons=['fingerprint', 'house fill', 'kanban', 'google', 'award', 'robot', 'book'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
                             "container": {"padding": "5!important", "background-color": "#fafafa"},
                             "icon": {"color": "orange", "font-size": "18px"},
                             "nav-link": {"font-size": "14px", "text-align": "left", "margin": "0px",
                                          "--hover-color": "#eee"},
                             "nav-link-selected": {"background-color": "#02ab21"},
                         }
                         )
if choose == "About":

    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">About the Creator</p>', unsafe_allow_html=True)
    st.write("Libby Chen is a data science practitioner, enthusias."
             "\n\nTo read more about Libby's data science project posts, please visit her github at:
             https://github.com/Libby007/condo_ds_project ")

    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">User Guide</p>', unsafe_allow_html=True)
    #user_guide = "https://github.com/Libby007/condo_ds_project/blob/main/MergedImages.png"
    st.image("MergedImages.png", width=1000)

elif choose == "Toronto condo price prediction":

    pickle_inn = open('xgb.pkl', 'rb')
    model = pickle.load(pickle_inn)

    # predict price
    def main():
        # st.subheader('Project 1: Toronto condo price prediction')
        html_temp = """
        <div style="background-color:#025246 ;padding:10px">
        <h2 style="color:white;text-align:center;">
        Toronto Condo Price Prediction ML App </h2>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.info('You do not need to answer all 15 questions, but the more questions you answer, the more accurate'
                ' result you will get')
        st.write("1.What is your preferred condo total square feet")
        total_sqft = st.slider(label='sqft', min_value=200, max_value=8000, key=77)

        st.write(
            "2.what is your expect condo maintenance fee per month? e.g., the average maintenance fee is about $0.64 "
            "per-square-foot in toronto, i.e.,1000 sqft with maintenance fee 640")
        maint_fee = st.slider(label='maintenance fee', min_value=100, max_value=7000, key=55)

        st.write("3.your household income per year")
        avg_income_household_yr = st.slider(label='income', min_value=40000, max_value=500000, key=66)

        parking = st.selectbox("4.your expect number of parkings", (0, 1, 2, 3, 4, 5, 6, 7))

        st.write("5.What is your expect age of building?")
        age_of_building = st.slider(label='age of building', min_value=0, max_value=100, key=99)

        col6, col7 = st.columns(2)
        with col6:
            commute_transit = st.slider(label="6.How often you take public transportation?", min_value=0, max_value=100,
                                        key=6)
        with col7:
            commute_car = st.slider(label="7.How often you drive car?", min_value=0, max_value=100 - commute_transit,
                                    key=7)

        near_by_schools = st.selectbox("8.your expect nearby schools?", (0, 1, 2, 3, 4, 5, 6, 7))

        st.write("9.your expect bars or restaurant near you")
        near_by_bars = st.slider(label='bars & restaurant', min_value=0, max_value=100, key=100)

        st.write('10.Select your preferred ethnic group in your neighborhood')
        st.info('Please set first English slider to 0 if you want reset')
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            speak_English = st.slider(label='for English', min_value=0, max_value=100, key=5)
        with col2:
            speak_Europe_language = st.slider(label='for European', min_value=0, max_value=100 - speak_English, key=4)
        with col3:
            speak_Africa_language = st.slider(label='for African', min_value=0,
                                              max_value=100 - speak_English - speak_Europe_language, key=3)
        with col4:
            value1 = st.slider(label='for Asian', min_value=0,
                               max_value=100 - speak_English - speak_Europe_language - speak_Africa_language, key=2)
        with col5:
            value5 = st.slider(label='for Middle East', min_value=0,
                               max_value=100 - speak_English - speak_Europe_language - speak_Africa_language - value1,
                               key=0)

        st.write("11.How much would you like to live with below education background people?")
        col8, col9, col10 = st.columns(3)
        with col8:
            edu_coll_or_univ_cert = st.slider(label="own college or university certificate", min_value=0, max_value=100,
                                              key=8)
        with col9:
            edu_trade_cert = st.slider(label="own trade certificate", min_value=0,
                                       max_value=100 - edu_coll_or_univ_cert, key=9)
        with col10:
            edu_no_diploma = st.slider(label="no diploma", min_value=0,
                                       max_value=100 - edu_coll_or_univ_cert - edu_trade_cert, key=10)

        st.write("12.How willing are you to live with people aged 35 to 49?")
        population_35_49_years = st.slider(label='population age', min_value=0, max_value=100, key=101)

        st.write("13.How willing are you like to live in below property type?")
        col11, col12 = st.columns(2)
        with col11:
            tenancy_property_type_low_rise = st.slider(label="low rise", min_value=0, max_value=80, key=11)
        with col12:
            tenancy_property_type_semi_detached = st.slider(label="semi_detached", min_value=0,
                                                            max_value=80 - tenancy_property_type_low_rise, key=12)

        bedrooms_plus_field = st.selectbox("14.how many areas would you like to own as alternative bedrooms?",
                                           (0, 1, 2))

        st.write("15.How willing are you like to live in below region?")
        col13, col14, col15, col16, col17 = st.columns(5)
        with col13:
            downtown = st.slider(label="Downtown", min_value=0, max_value=100, key=13)
        with col14:
            north_york = st.slider(label="North York", min_value=0, max_value=100 - downtown, key=14)
        with col15:
            scarborough = st.slider(label="Scarborough", min_value=0, max_value=100 - downtown - north_york, key=15)
        with col16:
            midtown = st.slider(label="Midtown", min_value=0, max_value=100 - downtown - north_york - scarborough,
                                key=16)
        with col17:
            york_crosstown = st.slider(label="York Crosstown", min_value=0,
                                       max_value=100 - downtown - north_york - scarborough - midtown, key=17)

        col18, col19, col20, col21 = st.columns(4)
        with col18:
            east_end = st.slider(label="East End", min_value=0,
                                 max_value=100 - downtown - north_york - scarborough - midtown - york_crosstown, key=18)
        with col19:
            east_york = st.slider(label="East York", min_value=0,
                                  max_value=100 - downtown - north_york - scarborough - midtown - york_crosstown - east_end,
                                  key=19)
        with col20:
            etobicoke = st.slider(label="Etobicoke", min_value=0,
                                  max_value=100 - downtown - north_york - scarborough - midtown - york_crosstown - east_end - east_york,
                                  key=20)
        with col21:
            west_end = st.slider(label="West End", min_value=0,
                                 max_value=100 - downtown - north_york - scarborough - midtown - york_crosstown - east_end - east_york - etobicoke,
                                 key=21)
        # predict_dataset
        raw_instances = np.array([[maint_fee, avg_income_household_yr, total_sqft, parking,
                                   age_of_building, commute_transit, commute_car, near_by_schools, near_by_bars,
                                   speak_English, speak_Europe_language, speak_Africa_language, edu_no_diploma,
                                   edu_trade_cert,
                                   edu_coll_or_univ_cert, population_35_49_years, tenancy_property_type_semi_detached,
                                   tenancy_property_type_low_rise, bedrooms_plus_field, east_end, east_york, etobicoke,
                                   midtown]]).astype(np.float64)
        predict_df = pd.read_csv('predict_dataset.csv',
                                 index_col=[0])
        scaler = StandardScaler()
        X_train_Test = scaler.fit_transform(predict_df.values)
        scaled_instances = scaler.transform(raw_instances)
        input = np.array(scaled_instances).astype(np.float64)
        output = model.predict(input)
        # st.write(a)


        if st.button("Predict"):
            # output = a
            st.balloons()
            st.success('The condo price will be approximately {}:'.format(output))


    if __name__ == '__main__':
        main()

elif choose == "Condo clustering":
    # open pdf
#     with open('cluster_ppt.pdf', "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')
#     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="1100" height="1000" type="application/pdf"></iframe>'
#     st.markdown(pdf_display, unsafe_allow_html=True)
    st.image("cluster.png", width=1000)

elif choose == "Condo search engine":
    before_norm = pd.read_csv(('before_norm.csv'),
                              index_col=[0])
    st.sidebar.info('7 Mandatory Filters')
    st.sidebar.info('Please select values in each of following 7 mandatory filters, otherwise system will show error')
    condo_price_option = st.sidebar.checkbox('condo price')
    if condo_price_option:
        condo_price_choice = st.sidebar.slider('Select a range of values',
                                               289900, 12998000, (600000, 3000000))
    total_sqft_option = st.sidebar.checkbox('total square feet')
    if total_sqft_option:
        sqft_choice = st.sidebar.slider('Select a range of values',
                                        270, 5703, (500, 1500))
    bedroom_option = st.sidebar.checkbox('bedroom')
    if bedroom_option:
        bedrooms_number = np.sort(before_norm.bedrooms.unique())
        bedrooms_choice = st.sidebar.multiselect(
            'Choose your preferred number of bedrooms:', bedrooms_number, default=2)
    bath_option = st.sidebar.checkbox('bath')
    if bath_option:
        bath_number = np.sort(before_norm.bath.unique())
        bath_choice = st.sidebar.multiselect(
            'Choose your preferred number of baths:', bath_number, default=1)
    parking_option = st.sidebar.checkbox('parking')
    if parking_option:
        parking_number = np.sort(before_norm.parking.unique())
        parking_choice = st.sidebar.multiselect(
            'Choose your preferred number of parking:', parking_number, default=1)
    age_building_option = st.sidebar.checkbox('age of building')
    if age_building_option:
        age_of_building_choice = st.sidebar.slider('Select a range of values',
                                                   0.0, 100.0, (5.0, 35.0))
    region_option = st.sidebar.checkbox('toronto region')
    # if region_option:
    positions_number = before_norm.toronto_region.unique()
    position_choice = st.sidebar.multiselect(
        'Choose your preferred neighborhood region:', positions_number, default='Downtown')

    st.sidebar.info('Optional Advanced Filters')
    commute = ['public transit', 'drive car']
    commute_choice = st.sidebar.multiselect(
        'Choose your preferred commute way:', commute)
    nearby = ['near_by_schools', 'near_by_grocery_stores', 'near_by_bars&restaurants']
    nearby_choice = st.sidebar.multiselect(
        'Choose your preferred neighborhood nearby facilities:', nearby)

    language = ['Speak English person', 'European', 'African', 'Asian', 'Middle East']
    language_choice = st.sidebar.multiselect(
        'Choose your preferred neighborhood ethnic group:', language)
    education = ['no diploma', 'high school', 'trade certificate', 'college or university certificate',
                 'bachelor or master degree']
    education_choice = st.sidebar.multiselect(
        'Choose your preferred neighborhood education group:', education)
    population = ['population_0_14_years', 'population_15_19_years', 'population_20_34_years', 'population_35_49_years',
                  'population_50_64_years', 'population_65_and_more']
    population_choice = st.sidebar.multiselect(
        'Choose your preferred neighborhood population group:', population)
    property_type = ['owned', 'rented', 'detached', 'semi_detached', 'duplex', 'low_rise', 'high_rise']
    property_type_choice = st.sidebar.multiselect(
        'Choose your preferred property type:', property_type)
    household = ['single family', 'multi family', 'single person', 'multi person']
    household_compo_choice = st.sidebar.multiselect(
        'Choose your preferred neighborhood household composition:', household)
    criminal_options = st.sidebar.checkbox('criminal')
    criminal_records = st.sidebar.slider('criminal records in the last 10 years', 0, 175, (0, 175))
    # change user friendly name to data frame name
    for i in range(len(language)):
        if language[0] == 'Speak English person':
            language[0] = 'speak_English'
        elif language[1] == 'European':
            language[1] = 'speak_Europe_language'
        elif language[2] == 'African':
            language[2] = 'speak_Africa_language'
        elif language[3] == 'Asian':
            language[3] = 'speak_Asian_language'
        elif language[4] == 'Middle East':
            language[4] = 'speak_middleEast_language'

    for i in range(len(commute)):
        if commute[0] == 'public transit':
            commute[0] = 'commute_transit'
        elif commute[1] == 'drive car':
            commute[1] = 'commute_car'

    for i in range(len(education)):
        if education[0] == 'no diploma':
            education[0] = 'edu_no_diploma'
        elif education[1] == 'high school':
            education[1] = 'edu_high_school'
        elif education[2] == 'trade certificate':
            education[2] = 'edu_trade_cert'
        elif education[3] == 'college or university certificate':
            education[3] = 'edu_coll_or_univ_cert'
        elif education[4] == 'bachelor or master degree':
            education[4] = 'edu_ba_or_msc_cert'

    for i in range(len(property_type)):
        if property_type[0] == 'owned':
            property_type[0] = 'tenancy_property_type_owned'
        elif property_type[1] == 'rented':
            property_type[1] = 'tenancy_property_type_rented'
        elif property_type[2] == 'detached':
            property_type[2] = 'tenancy_property_type_detached'
        elif property_type[3] == 'semi_detached':
            property_type[3] = 'tenancy_property_type_semi_detached'
        elif property_type[4] == 'duplex':
            property_type[4] = 'tenancy_property_type_duplex'
        elif property_type[5] == 'low_rise':
            property_type[5] = 'tenancy_property_type_low_rise'
        elif property_type[6] == 'high_rise':
            property_type[6] = 'tenancy_property_type_high_rise'

    for i in range(len(household)):
        if household[0] == 'single family':
            household[0] = 'household_compo_singlefamily'
        elif household[1] == 'multi family':
            household[1] = 'household_compo_multifamily'
        elif household[2] == 'single person':
            household[2] = 'household_compo_singleperson'
        elif household[3] == 'multi person':
            household[3] = 'household_compo_multiperson'

    # header
    html_temp = """      
    <div style="background-color:#025246;padding:10px">
    <h2 style="color:white;text-align:center;">
    Your condo search result</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.info('Please check all your mandatory filters before submit')
    if st.button("Click Submit"):
        if not condo_price_option or not total_sqft_option or not bedroom_option or not bath_option or not \
                parking_option or not age_building_option or not region_option:
            st.error('Please select all 7 mandatory filters and then try again!')
        else:

            lists = [[condo_price_option, 'condo_price'], [total_sqft_option, 'total_sqft'],
                     [bedroom_option, 'bedrooms'], [bath_option, 'bath'], [parking_option, 'parking'],
                     [age_building_option, 'age_of_building'], [region_option, 'toronto_region']]
            appended_data = []
            for i in range(0, len(lists)):
                if lists[i][0]:
                    appended_data.append(before_norm[lists[i][1]])
            appended_data = pd.concat(appended_data, axis=1)
            if criminal_options:
                criminals = before_norm['criminal_records']
                advanced_filter = pd.concat([(before_norm[commute_choice + nearby_choice + language_choice +
                                                          education_choice + population_choice + property_type_choice
                                                          + household_compo_choice]), criminals], axis=1)
            else:
                advanced_filter = before_norm[commute_choice + nearby_choice + language_choice + education_choice +
                                              population_choice + property_type_choice + household_compo_choice]
            final_df = pd.concat([advanced_filter, appended_data], axis=1)
            # st.write(final_df)
            # filter
            filter_bedroom = (final_df['bedrooms'].isin(bedrooms_choice))
            filter_bath = (final_df['bath'].isin(bath_choice))
            filter_parking = (final_df['parking'].isin(parking_choice))
            filter_region = (final_df['toronto_region'].isin(position_choice))
            filter_age_building = ((age_of_building_choice[0] <= final_df['age_of_building']) & (
                    final_df['age_of_building'] <= age_of_building_choice[1]))
            filter_price = ((condo_price_choice[0] <= final_df['condo_price']) & (
                    final_df['condo_price'] <= condo_price_choice[1]))
            filter_sqft = ((sqft_choice[0] <= final_df['total_sqft']) & (final_df['total_sqft'] <= sqft_choice[1]))

            filter_lists = [[bedroom_option, filter_bedroom], [bath_choice, filter_bath],
                            [parking_choice, filter_parking], [position_choice, filter_region],
                            [age_of_building_choice, filter_age_building], [condo_price_choice, filter_price],
                            [sqft_choice, filter_sqft]]
            sample_arr = [True, True]
            # create a boolean array with size 3142 and all values in True
            bool_arr = np.random.choice(sample_arr, size=(1, 3142))

            for i in range(0, len(filter_lists)):
                if filter_lists[i][0]:
                    bool_arr = bool_arr & filter_lists[i][1]  # use & to vertical stack boolean columns
                elif not filter_lists[i][0]:
                    st.error('Please select your values in all mandatory filters')

            filter_criminal = ((criminal_records[0] <= before_norm['criminal_records']) & (
                    before_norm['criminal_records'] <= criminal_records[1]))
            if len(final_df[bool_arr]) == 0:
                st.info('There is no data in your selected values, please try again!')
            else:
                st.success('Successfully find condo from your customized filters!')
                st.balloons()
                if criminal_records:
                    st.write(final_df[bool_arr & filter_criminal])
                else:
                    st.write(final_df[bool_arr])

            api_df = final_df[bool_arr]
            api_df = api_df.copy()
            api_df['index'] = api_df.index.to_list()
            merge = api_df.merge(before_norm, how='left', on='index')
            geo_df = merge[['latitude', 'longitude']]
            # st.write(geo_df)
            st.map(geo_df)
                        # write  data frame api_df to google sheet
            scope = ['https://spreadsheets.google.com/feeds',
                     'https://www.googleapis.com/auth/drive']
            credentials = ServiceAccountCredentials.from_json_keyfile_name(
                'streamlit-353615-491ac72605bb.json', scope)
            gc = gspread.authorize(credentials)

            spreadsheet_key = '1_JuGM1m6oBy_gLJ2tLTp_KsB7fylbQCj8_wzXYmIap0'
            wks_name = 'Sheet1'
            d2g.upload(api_df, spreadsheet_key, wks_name, credentials=credentials, row_names=True)

            # write list to google sheet2
            region_selection = pd.DataFrame(position_choice, columns=['region'])
            d2g.upload(region_selection, spreadsheet_key, 'Sheet2', credentials=credentials, row_names=True)
            
elif choose == "Multi-criteria ranked condos":
    st.sidebar.info('Please finish part "Condo search engine" before you jump to this part!')
    # read google sheet data frame
    gc = gspread.service_account(filename='streamlit-353615-491ac72605bb.json')
    sh = gc.open_by_url(
        'https://docs.google.com/spreadsheets/d/1_JuGM1m6oBy_gLJ2tLTp_KsB7fylbQCj8_wzXYmIap0/edit#gid=0')
    ws = sh.worksheet('Sheet1')
    api_df = pd.DataFrame(ws.get_all_records()).iloc[:, 1:]
    # read region list from google sheet2
    # st.write(api_df)
    # ws_2 = sh.worksheet('Sheet2')
    # temp_region = pd.DataFrame(ws_2.get_all_records()).iloc[:, 1:]
    # region_list = temp_region.values.tolist()  # convert data frame to list
    # position_choice = list(chain(*region_list))  # merge 2 lists to 1 list
    # st.write(position_choice)

    # new way to read region list
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        'streamlit-353615-491ac72605bb.json', scope)
    service = discovery.build('sheets', 'v4', credentials=credentials)
    spreadsheet_id = '1_JuGM1m6oBy_gLJ2tLTp_KsB7fylbQCj8_wzXYmIap0'
    # ranges = 'Sheet1!A1:Z200'
    # request = service.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=ranges,
    #                                               ).execute()
    # sheet_values = request.get('values', [])
    # api_df = pd.DataFrame(sheet_values[1:], columns=sheet_values[0]).iloc[:, 1:]
    # st.write(api_df)

    request2 = service.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range='Sheet2!A1:B11').execute()
    sheet2_values = request2.get('values', [])
    temp_region = pd.DataFrame(sheet2_values[1:], columns=sheet2_values[0]).iloc[:, 1:]
    region_list = temp_region.values.tolist()  # convert data frame to list
    position_choice = list(chain(*region_list))  # merge 2 lists to 1 list
    # st.write(position_choice)
    
    # weight
    html_temp = """
    <div style="background-color:#025246;padding:8px">
    <h2 style="color:white;text-align:center;">
    Further Step: Rank condos by your customized Multi-Criteria Decision </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.info('Please be patient to make each of your choice slowly, since it is free version of google platform cloud '
        'which has limit speed to response to user, i.e. each of your choice needs running time')
    st.write('Please select the weight for each of your customized criteria')
    before_norm = pd.read_csv('before_norm.csv',
                              index_col=[0])
    column_list = api_df.columns.tolist()
    weight_list = []
    region_choose_list = []
    for i, x in enumerate(column_list[:-1]):
        if x == 'toronto_region':
            for j in range(0, len(position_choice)):
                weight_list.append(st.slider(label=position_choice[j], min_value=0, max_value=10, key=j))
                region_choose_list.append(position_choice[j])
        else:
            weight_list.append(st.slider(label=x, min_value=0, max_value=10, key=i + 20))
    # add info to choose weight
    if set(weight_list) != {0}:

        # need to change weight order to make sure region cols have same order with data table
        # 2 lists into dict
        dicts = dict(zip(region_choose_list, weight_list[-len(position_choice):]))
        # sort dict string keys in alphabet order
        sorted_dicts = dict(sorted(dicts.items(), key=lambda x: x[0]))
        # get new weights after sorted dict keys
        new_weight_region = [x for x in sorted_dicts.values()]
        # change region weights with new weights to ensure the same order with dataframe col
        weight_list[-len(position_choice):] = new_weight_region

        # transform categorical column toronto_region
        api_df = pd.concat([api_df, pd.get_dummies(api_df.toronto_region)], axis=1)
        api_df = api_df.drop(['toronto_region', 'index'], axis=1).copy()

        # st.write(api_df)

        # normalize matrix

        # get square root of all sum squares
        def replace_values(df, index):
            sums = 0
            for y in df[df.columns.tolist()[index]]:
                sums += y ** 2
                sqrt_sum = np.sqrt(sums)
            return sqrt_sum


        # every value divided by its column's square root of all values sum squares
        def normalize_df(df):
            for z in range(0, df.shape[1]):
                df[df.columns.tolist()[z]] = df[df.columns.tolist()[z]] / replace_values(df, z)


        normalize_df(api_df)

        # st.write(api_df)
        # introduce multi-criteria method
        final_moora = api_df.copy()

        # transfer weights
        weights = []
        for i in weight_list:
            weights.append(i / sum(weight_list))


        # multiply weights
        def weight_normalize_rate(df, weights):  # each column * weights
            m = df.shape[0]
            for i in range(0, m):
                df.iloc[i] = np.multiply(df.iloc[i], weights)
            return df


        final_moora = weight_normalize_rate(final_moora, weights)
        # st.write(final_moora)

        # rank
        column_name = final_moora.columns.tolist()
        for i in column_name:
            if 'criminal_records' in column_name:
                final_moora['non_beneficial'] = final_moora[['condo_price', 'age_of_building', 'criminal_records']].sum(
                    axis=1)
                final_moora['beneficial'] = final_moora[[x for i, x in enumerate(column_name) if
                                                         x not in (
                                                             'condo_price', 'age_of_building',
                                                             'criminal_records')]].sum(
                    axis=1)
            else:
                final_moora['non_beneficial'] = final_moora[['condo_price', 'age_of_building']].sum(axis=1)
                final_moora['beneficial'] = final_moora[[x for i, x in enumerate(column_name) if
                                                         x not in ('condo_price', 'age_of_building')]].sum(axis=1)
        # get moora col based on formula
        final_moora['moora'] = final_moora['beneficial'] - final_moora['non_beneficial']
        # get rank column, highest score rank 1, same score has same rank
        final_moora['rank'] = final_moora['moora'].rank(method='max', ascending=0)
        final_moora['index'] = pd.DataFrame(ws.get_all_records()).iloc[:, 1:]['index']
        # st.write(final_moora)

        # st.write(final_moora[final_moora['rank'] < 6])
        st.write('Check your ranked condo result based on your multi-criteria weights')
        final_rank_moora = final_moora[final_moora['rank'] < 6].copy()
        # final_rank_moora['index'] = final_rank_moora.index.to_list()
        # st.write(final_rank_moora)

        # merge with those before normalize data and also get street info
        merge = final_rank_moora.merge(before_norm, how='left', on='index')
        street_info = pd.read_csv('street.csv',
                                  index_col=[0])
        merge = merge.copy()
        merge = merge.merge(street_info, how='left', on='index')
        merge_col = [col for col in merge.columns if col.endswith("_y") or 'rank' in col
                     or 'latitude' in col or 'longitude' in col or 'street' in col]
        final_merge = merge[merge_col]

        # set button color and size
        m = st.markdown(""" <style> div.stButton > button:first-child { background-color: rgb(255, 255, 128); height: 4em;
        width: 10em; font-weight : bold ; box-sizing: border-box; } </style>""",
                        unsafe_allow_html=True)

        # TOPSIS
        final_topsis = api_df.copy()  # after normalize
        # multiply weight
        final_topsis = weight_normalize_rate(final_topsis, weights)
        # st.write(final_topsis)

        # get non-beneficial attributes index
        col_list = final_topsis.columns.tolist()
        for col in col_list:
            if 'criminal_records' in col_list:
                index = [api_df.columns.get_loc(col) for col in ["criminal_records", "condo_price", "age_of_building"]]
            else:
                index = [api_df.columns.get_loc(col) for col in ["condo_price", "age_of_building"]]
        non_benef_index_list = tuple(index)

        # PIS and NIS
        # calculate the max and min value for each column
        n = final_topsis.shape[1]
        PIS = np.zeros(n)  # build array-like with same column shape of TOPSIS
        NIS = np.zeros(n)

        for j in range(0, n):
            column = final_topsis.iloc[:, j]  # every column in dataframe
            max_val = np.max(column)  # get max and min for each column
            min_val = np.min(column)

            if j in non_benef_index_list:  # non-beneficial attributes index
                PIS[j] = min_val  # non-beneficial PIS is min, NIS is max
                NIS[j] = max_val
            else:
                PIS[j] = max_val  # beneficial PIS is max, NIS is min
                NIS[j] = min_val

        # get data frame of PIS and NIS
        df_PN = pd.DataFrame(data=[PIS, NIS], index=["PIS", "NIS"], columns=final_topsis.columns.tolist())
        # st.write(df_PN)

        # Calculate the euclidean distance from the PIS and NIS
        m = final_topsis.shape[0]
        sp = np.zeros(m)
        sn = np.zeros(m)
        cs = np.zeros(m)

        for i in range(m):
            diff_pos = final_topsis.iloc[i] - PIS  # each row minius PIS, they have same column shape
            diff_neg = final_topsis.iloc[i] - NIS
            sp[i] = np.sqrt(diff_pos @ diff_pos)  # @ operator for matrix multiplication
            sn[i] = np.sqrt(diff_neg @ diff_neg)
            cs[i] = sn[i] / (sp[i] + sn[i])  # performance formula

        performance_score = pd.DataFrame(data=zip(sp, sn, cs), index=final_topsis.index,
                                         columns=["PIS_Separation", "NIS_Separation", "Performance"])

        # concat 2 data frames
        final_TOPSIS_rank = pd.concat([final_topsis, performance_score], axis=1)
        final_TOPSIS_rank = final_TOPSIS_rank.copy()
        final_TOPSIS_rank['rank'] = final_TOPSIS_rank['Performance'].rank(method='max', ascending=0)
        final_TOPSIS_rank['index'] = pd.DataFrame(ws.get_all_records()).iloc[:, 1:]['index']
        # st.write(final_TOPSIS_rank)
        final_TOPSIS_rank = final_TOPSIS_rank[final_TOPSIS_rank['rank'] < 6]
        # final_TOPSIS_rank['index'] = final_TOPSIS_rank.index.to_list()
        # st.write(final_TOPSIS_rank)

        # merge with those before normalize data and also get street info
        merge_topsis = final_TOPSIS_rank.merge(before_norm, how='left', on='index')
        merge_topsis = merge_topsis.copy()
        merge_topsis = merge_topsis.merge(street_info, how='left', on='index')
        merge_col = [col for col in merge.columns if col.endswith("_y") or 'rank' in col
                     or 'latitude' in col or 'longitude' in col or 'street' in col]
        final_merge_topsis = merge_topsis[merge_col]
        # st.write(final_merge_topsis)

        # method options
        options = st.multiselect('Please select your multi-criteria method',
                                 ['MOORA: Multi-objective Optimization on the basis of Ratio Analysis',
                                  'TOPSIS: Technique for Order of Preference by Similarity to Ideal Solution'])

        if st.button('Click me: check your top condos!'):
            if options:
                for i in options:
                    if i == 'MOORA: Multi-objective Optimization on the basis of Ratio Analysis':
                        st.success('Congrats! Please check customized top condos for you.')
                        st.snow()
                        st.write(final_merge)
                        geo_moora = final_merge[['latitude', 'longitude']]
                        st.map(geo_moora)
                    elif i == 'TOPSIS: Technique for Order of Preference by Similarity to Ideal Solution':
                        st.success('Congrats! Please check customized top condos for you.')
                        st.snow()
                        st.write(final_merge_topsis)
                        geo_topsis = final_merge_topsis[['latitude', 'longitude']]
                        st.map(geo_topsis)
            elif not options:
                st.info('Please select your multi-criteria method!')
    else:
        st.error('Please choose weight for your customized multi-criteria!')

elif choose == "Talk to AI Libby":
    model = load_model('chatbot_model.h5')
    intents = json.loads(open('intents1.json').read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    lemmatizer = WordNetLemmatizer()

    def clean_up_sentence(sentence):
        # tokenize the pattern - split words into array
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word - create short form for word
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    def bow(sentence, words, show_details=True):
        # tokenize the pattern
        sentence_words = clean_up_sentence(sentence)
        # bag of words - matrix of N words, vocabulary matrix
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return (np.array(bag))
    def predict_class(sentence, model):
        # filter out predictions below a threshold
        p = bow(sentence, words, show_details=False)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        return return_list
    def getResponse(ints, intents_json):
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if (i['tag'] == tag):
                result = random.choice(i['responses'])
                break
        return result
    def chatbot_response(msg):
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
        return res
    def get_text():
        input_text = st.text_input("You: ", "Hello, how are you?", key="input")
        return input_text

    st.header("Chat with AI to know more about Libby")
    user_input = get_text()

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if user_input:
        output = chatbot_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

elif choose == "Algorithm theory and principle":

    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">The underlying theory and principle of project</p>', unsafe_allow_html=True)
    # open pdf
#     with open('theory_ppt.pdf', "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')
#     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="1100" height="1000" type="application/pdf"></iframe>'
#     st.markdown(pdf_display, unsafe_allow_html=True)
    st.image("theory.png", width=1000)
