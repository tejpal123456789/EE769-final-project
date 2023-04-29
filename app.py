import streamlit as st
import pickle
import pandas as pd

teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

# cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
#        'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
#        'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
#        'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
#        'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
#        'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('Artifacts/model.pkl','rb'))
preprocessor=pickle.load(open('Artifacts/proprocessor.pkl','rb'))
st.title('IPL Win Predictor')




batting_team = st.selectbox('Select the batting team',sorted(teams))
bowling_team = st.selectbox('Select the bowling team',sorted(teams))

#selected_city = st.selectbox('Select host city',sorted(cities))

target = st.number_input('Target')

score = st.number_input('Score')

overs = st.number_input('Overs completed')
wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left
    
    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})
    data_scaled=preprocessor.transform(input_df)
    print(input_df)

    result = pipe.predict_proba(data_scaled)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")