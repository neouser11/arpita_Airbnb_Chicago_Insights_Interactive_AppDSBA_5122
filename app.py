import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import zipfile
from st_btn_select import st_btn_select
import plotly.express as px
import squarify
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

#import sentimentanalysis
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")

@st.cache_data
def get_data():
  df = pd.read_csv('data/combinedlistings_cleanv2.zip',compression='zip')
  return df

st.title('Airbnb Chicago')
page = st_btn_select(
  # The different pages
  ('Interesting Trends','Insights','Top Hosts Listings','Sentiment Analyis of Reviews','Reliable and Secure Stay'),
  # Enable navbar
  nav=False
)

df = get_data()
 
 #Insights Page
if page == 'Insights':
  with st.expander('Interesting Insights on Airbnb Chicago'):
    st.write('This app lets the user visualize interesting insights in the Chicago Airbnb Market using Neighbourhood and Review score rating filters')


  neighbourhood = st.sidebar.selectbox('Choose a neighbourhood',df['neighbourhood_cleansed'].unique())

  rating_var = st.sidebar.slider("Review Scores Rating", float(df.review_scores_rating.min()), float(df.review_scores_rating.max()),(4.5, 5.0))

  #################

  st.markdown("Use the Slider for Review Score Rating to find the Neighbourhoods that has the **Maximum supply of Listings** and more **customer Reviews**")
    
  top = df.query(f"""review_scores_rating.between{rating_var}""")
  groupedDF = top.groupby( "neighbourhood_cleansed", as_index=False ).agg(Average_Number_of_Reviews=('number_of_reviews', \
                                                                    np.mean),CountOfListings=('id', np.size))  
  #st.table(groupedDF)
  test = alt.Chart(groupedDF,title=f"Neighbourhoods with Maximum Count of Listings and Customer Reviews between Review score Rating Range:{rating_var}").\
  mark_point().encode(
    x='Average_Number_of_Reviews',
    y='CountOfListings',    
    color=alt.Color('neighbourhood_cleansed', legend=None)
  ).interactive()
  st.altair_chart(test, use_container_width=True)

###########################

  st.markdown("Select Neighbourhood Filter to see the Average price per Roomtype")
  price=df.query(f"""neighbourhood_cleansed==@neighbourhood""")
  pricedf=price.groupby(['room_type'],as_index=False).agg(AveragePrice=('price',np.mean)).sort_values('AveragePrice', ascending=False, ignore_index= True)
  #st.table(pricedf)
  bars = alt.Chart(pricedf,title=f"Average Price by Room Type in **{neighbourhood}**").mark_bar().encode(
        x= alt.X('room_type:N', title='Room Type', sort = '-y' ),      
        y=alt.Y('AveragePrice:Q', title='Average Price')
        )
  st.altair_chart(bars, use_container_width=True)


  ############"#####
  st.markdown("Select Neighbourhood Filter to find the **Top Rated Hosts** in the area")

  top = df.query(f"""neighbourhood_cleansed==@neighbourhood""")
  topdf=top.groupby(['host_name'],as_index=False).agg(NumberOfReviews=('number_of_reviews',np.size))\
      .sort_values('NumberOfReviews',ascending=False,ignore_index=True)
  topdf= topdf.head(5)
  bars = alt.Chart(topdf,title=f"Top Rated Hosts in **{neighbourhood}**").mark_bar().encode(
        x= alt.Y('NumberOfReviews:Q', title='Number Of Reviews'),      
        y=alt.Y('host_name:N', title='Host',sort = '-x')
        )
  st.altair_chart(bars, use_container_width=True)

if page=='Interesting Trends':
  #st.markdown("Select Neighbourhood Filter to see the Average price per Roomtype")
  #price=df.query(f"""neighbourhood_cleansed==@neighbourhood""")
  df1=df.groupby(['neighbourhood_cleansed'],as_index=False).agg(AveragePrice=('price',np.mean)).sort_values('AveragePrice', ascending=False, ignore_index= True)
  df2= df1.head(5)#st.table(pricedf)
  df3=df.groupby(['neighbourhood_cleansed'],as_index=False).agg(AveragePrice=('price',np.mean)).sort_values('AveragePrice', ignore_index= True)
  df4=df3.head(5)
  bars = alt.Chart(df2,title="Top 5 and Bottom 5 neighbourhoods by Average Price").mark_bar(color='red').encode(
        y= alt.Y('neighbourhood_cleansed:N', title='Neighbourhood', sort = '-x' ),      
        x=alt.X('AveragePrice:Q', title='Average Price')
        )
  bars2=alt.Chart(df4).mark_bar(color='blue').encode(
        y= alt.Y('neighbourhood_cleansed:N', title='Neighbourhood', sort = 'x' ),      
        x=alt.X('AveragePrice:Q', title='Average Price')
        )

  st.altair_chart(bars+bars2, use_container_width=True)

if page=="Top Hosts Listings":
  neighbourhood = st.sidebar.selectbox('Choose Neighbourhood',df['neighbourhood_cleansed'].unique())
  tdf1 = df.query(f"""neighbourhood_cleansed==@neighbourhood""")
  tdf2=tdf1.groupby(['host_name'],as_index=False).agg(No_of_Listings=('id',np.size))\
      .sort_values('No_of_Listings',ascending=False,ignore_index=True)
  tdf3=tdf2.head(10)

  plt.rc('font', size=14)
  sizes= tdf3["No_of_Listings"]
  labels=tdf3["host_name"][:10]
  color_list = ['#0f7216', '#b2790c', '#ffe9a3',
              '#f9d4d4', '#d35158', '#ea3033']
  squarify.plot(sizes=sizes, label=labels,
              color=color_list, alpha=0.7).set(title='Treemap with Squarify')
  plt.axis('off')
  st.pyplot()

  st.markdown("Use the dropdown to select one of the top hosts and get more details of their listings and a link to the actual listing in airbnb")
  #Click your ideal choice to get all the details of the listings by the top hosts in the area along with a link to the actual listing in airbnb.
  #ophosts = st.selectbox("Choose one of the top hosts",tdf3['host_name'])
  
  price1 = st.sidebar.slider("Price Range($)", float(df.price.min()), float(df.price.clip(upper=10000.).max()),\
                      (100., 300.))
  reviewsd = st.sidebar.slider('Minimum Reviews', 0, 500, (100))
  #st.map(df.query(f"neighbourhood_cleansed==@neighbourhood and price.between{price1} and \
                  #number_of_reviews>={reviewsd}")[["latitude", "longitude"]].dropna(how="any"), zoom=10)
  #mapdf= df.query(f"neighbourhood_cleansed==@neighbourhood and price.between{price1} and \
                  #number_of_reviews>={reviewsd}")[["latitude", "longitude"]]
  tdfmap = df.query(f"""neighbourhood_cleansed==@neighbourhood and price.between{price1} and \
                  number_of_reviews>={reviewsd}""")
  tdf2=tdfmap[["latitude", "longitude", "neighbourhood_cleansed","host_name","room_type"]]
  map = folium.Map(location=[tdf2.latitude.mean(), tdf2.longitude.mean()], zoom_start=14,control_scale=True)
  for index, location_info in tdf2.iterrows():
    folium.Marker([location_info["latitude"], location_info["longitude"]], popup=location_info["neighbourhood_cleansed"],\
                  tooltip=location_info["neighbourhood_cleansed"]).add_to(map)

  st_folium(map, width=725, returned_objects=[])  

