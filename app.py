import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import zipfile
from st_btn_select import st_btn_select
import plotly.express as px
#matplotimport squarify
import matplotlib.pyplot as plt
#Interactive maps with detailed pop up using folium 
import folium
from streamlit_folium import st_folium
from PIL import Image


st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")

@st.cache_data
def get_data():
  df = pd.read_csv('data/combinedlistings_cleanv2.zip',compression='zip')
  return df

st.title('Airbnb Chicago Insights')
page = st_btn_select(
  # The different pages
  ('Interesting Trends','Listing By Top Hosts','SuperHosts vs Non-SuperHosts','Best Neighbourhoods+Listings'),
  # Enable navbar
  nav=False
)

df = get_data()

##################  PAGE=Interesting Trends #################################################################################
if page=='Interesting Trends':
  st.markdown("Settled along the banks of Lake Michigan, the Windy City of Chicago offers world-class dining, exciting architecture, a top performing arts scene, and plenty of excellent museums. With more than 200 neighborhoods calling this city home, there’s a unique aura found in each.Let us get an idea  which neighbourhoods have the **:red[highest]** and the **:blue[lowest]** average Price per night.")
  df1=df.groupby(['neighbourhood_cleansed'],as_index=False).agg(AveragePrice=('price',np.mean)).sort_values('AveragePrice', ascending=False, ignore_index= True)
  df2= df1.head(5)#st.table(pricedf)
  df3=df.groupby(['neighbourhood_cleansed'],as_index=False).agg(AveragePrice=('price',np.mean)).sort_values('AveragePrice', ignore_index= True)
  df4=df3.head(5)
  
  df5 =  pd.concat([df2, df4], ignore_index= True)
  top5bar = alt.Chart(df5,title="Top 5 and Bottom 5 neighbourhoods by Average Price").mark_bar().encode(
    y= alt.Y('neighbourhood_cleansed:N', title='Neighbourhood', sort = '-x' ),      
    x=alt.X('AveragePrice:Q', title='Average Price ($)'),
    color=alt.condition(alt.expr.datum['AveragePrice'] > df5["AveragePrice"].mean(),
                            alt.value('red'),
                            alt.value('blue'))
    )
  st.altair_chart(top5bar, use_container_width=True)

  st.markdown("Airbnb hosts can list entire homes/apartments, private, shared rooms, and more recently hotel rooms.This chart shows which property type have increased number of listings over the years.")
  from datetime import date as dt
  df['Date'] = pd.to_datetime(df['first_review'])
  df['Year']=df['Date'].dt.year
  #df
  topyear=df.groupby(['Year','room_type'],as_index=False).agg(NumberOfListings=('id',np.size))
  fig = px.line(topyear,title="Growth in Listings By Property Type Over Time", x = "Year", y = "NumberOfListings",
              color = "room_type").update_layout(yaxis_title="Number Of Listings",legend_title="Property Type")
  
  st.plotly_chart(fig, use_container_width= True )
  
######################################SUPERHOST VS NON SUPERHOST####################################################
if page=='SuperHosts vs Non-SuperHosts':
  
  neighbourhood = st.sidebar.selectbox('Select neighbourhood',df['neighbourhood_cleansed'].unique())
  image = Image.open('data/superhost.jpg')
  st.sidebar.image(image, use_column_width=True)
  st.markdown("Select Neighbourhood to analyze the **:red[Differences in Price ($)]** for Properties listed by Superhosts and Non-SuperHosts.")
  st.markdown("Are **:blue[Superhosts]** charging more compared to **:red[Non-Superhosts]** across different property types ?")
  #roomtype = st.sidebar.selectbox('Choose preferred Room Type',df['room_type'].unique())
  price=df.query(f"""neighbourhood_cleansed==@neighbourhood #and room_type==@roomtype""")

  def superhost(row_value):
      if "t" in str(row_value).lower():
        return 'SuperHost'
      else:
        return 'Non-SuperHost'
      
  price["Host Is Superhost"] =  price['host_is_superhost'].apply(superhost)
  superdf=price.groupby(['room_type','Host Is Superhost'],as_index=False).agg(AveragePrice=('price',np.mean))                  
 
  fig2 = px.bar(superdf, title=f"Average Price of Properties listed by Superhosts vs Non-SuperHosts in *{neighbourhood}*",\
             x="room_type",
              y="AveragePrice",      
              color="Host Is Superhost", 
              color_discrete_map={"SuperHost":"blue", "Non-SuperHost":"red"}, 
              #category_orders={"variable":["Revenue","Expenses"]}, 
              barmode = 'group',
              text="AveragePrice",        
              hover_data=['AveragePrice']).update_traces(textposition='outside',cliponaxis=False, texttemplate='$%{y:.2f}').update_layout(
    xaxis_title="Property Type", yaxis_title="Average Price ($)",xaxis_tickangle=-45)
    #fig.update_traces(textfont_size=12, textangle=0, textposition="outside")
  
  st.plotly_chart(fig2, use_container_width= True)



  st.markdown("Select Neighbourhood to analyze the differences in **:red[Number of listings]** by Superhosts and Non-SuperHosts.")
  st.markdown("Does Superhosts lists more number of properties compared to Non-Superhosts?")
  #roomtype = st.sidebar.selectbox('Choose preferred Room Type',df['room_type'].unique())
  price=df.query(f"""neighbourhood_cleansed==@neighbourhood #and room_type==@roomtype""")

  def superhost(row_value):
      if "t" in str(row_value).lower():
        return 'SuperHost'
      else:
        return 'Non-SuperHost'
      
  price["Host Is Superhost"] =  price['host_is_superhost'].apply(superhost)
  superdf1=price.groupby(['room_type','Host Is Superhost'],as_index=False).agg(NumberOfListings=('id',np.size))                  
 
  fig3 = px.bar(superdf1, title=f"Number of Properties Listed by Superhosts vs Non-SuperHosts in *{neighbourhood}*",\
             x="room_type",
              y="NumberOfListings",      
              color="Host Is Superhost", 
              color_discrete_map={"SuperHost":"blue", "Non-SuperHost":"red"}, 
              #category_orders={"variable":["Revenue","Expenses"]}, 
              barmode = 'group',
              text="NumberOfListings",        
              hover_data=['NumberOfListings']).update_traces(textposition='outside',cliponaxis=False).update_layout(
    xaxis_title="Property Type", yaxis_title="Number Of Listings",xaxis_tickangle=-45)
    #fig.update_traces(textfont_size=12, textangle=0, textposition="outside")
  
  st.plotly_chart(fig3, use_container_width= True)



  st.markdown("Select Neighbourhood to find who gets **:red[More Reviews]** from guests: Superhosts Or Non-SuperHosts? .")
  st.markdown("Does Superhosts gets more reviews compared to Non-Superhosts?")
  #roomtype = st.sidebar.selectbox('Choose preferred Room Type',df['room_type'].unique())
  price=df.query(f"""neighbourhood_cleansed==@neighbourhood #and room_type==@roomtype""")

  def superhost(row_value):
      if "t" in str(row_value).lower():
        return 'SuperHost'
      else:
        return 'Non-SuperHost'
      
  price["Host Is Superhost"] =  price['host_is_superhost'].apply(superhost)
  superdf2=price.groupby(['Host Is Superhost'],as_index=False).agg(Average_Number_of_Reviews=('number_of_reviews',np.mean))                  
 
  fig4 = px.bar(superdf2, title=f"Average Number of Reviews received by Superhosts vs Non-SuperHosts in *{neighbourhood}*",\
             x="Host Is Superhost",
              y="Average_Number_of_Reviews", 
                                    
              barmode = 'group',
              text="Average_Number_of_Reviews",        
              hover_data=['Average_Number_of_Reviews']).update_traces(textposition='outside',cliponaxis=False,texttemplate='%{y:.2f}').update_layout(
    xaxis_title="Type of Host", yaxis_title="Average Number of Reviews",xaxis_tickangle=-45)
    #fig.update_traces(textfont_size=12, textangle=0, textposition="outside")
  
  st.plotly_chart(fig4, use_container_width= True)


################################### Page=Best Neighbourhoods and Most Reliable Listings ######################################################
if page == 'Best Neighbourhoods+Reliable Listings':
  # with st.expander('Interesting Insights on Airbnb Chicago'):
  #   st.write('This app lets the user visualize interesting insights in the Chicago Airbnb Market using Neighbourhood and Review score rating filters')
  rating_var = st.sidebar.slider("Review Scores Rating", float(df.review_scores_rating.min()), float(df.review_scores_rating.max()),(4.5, 5.0))
  neighbourhood = st.sidebar.selectbox('Choose a neighbourhood',df['neighbourhood_cleansed'].unique())

  st.markdown("Are you looking for Neighbourhoods that have More Supply of listings and more Customer Reviews? Use the Slider for Review Score Rating to find those neighbourhoods.")
  top = df.query(f"""review_scores_rating.between{rating_var}""")
  groupedDF = top.groupby( "neighbourhood_cleansed", as_index=False ).agg(Average_Number_of_Reviews=('number_of_reviews', \
                                                                    np.mean),CountOfListings=('id', np.size))  
  #st.table(groupedDF)
  test = alt.Chart(groupedDF,title=f"Neighbourhoods with Maximum Supply of Listings and Customer Reviews between Review score Rating Range:{rating_var}").\
  mark_point().encode(
    x='Average_Number_of_Reviews',
    y='CountOfListings',    
    color=alt.Color('neighbourhood_cleansed', legend=None),
    size='neighbourhood_cleansed'
  ).interactive()
  
  st.altair_chart(test, use_container_width=True)
  st.markdown(":pencil: **:red[West Town, Lake View, Logan Square, Lincoln Park]** are your options.")
  st.write("#")

###########################How many listings are licenced in each room type#################################

  def getLicenseType(row_value):
    if pd.isnull(row_value) != True  and str(row_value) !='' :
      if "pending" in str(row_value).lower():
        return 'Pending'
      elif str(row_value) == "32+ Days Listing" or str(row_value) == "32+ days listing" or str(row_value) == "32+ Days of Listing" :
        return 'Exempt'
      else:
        return 'Licensed'
    else:
      return 'Unlicensed'
    
  def superhost(row_value):
      if "t" in str(row_value).lower():
        return 'SuperHost'
      else:
        return 'Non-SuperHost'

  st.markdown("Do you want to find the distribution of listings that are Licensed, Unlicensed, Pending or Exempt across different Room types in your chosen neighbourhood?")
  st.markdown("Drill down the chart by Superhost filter on the sidebar.")
  #st.markdown("Select Neighbourhood and SuperHost Filter to find the number of licensed or Unlicensed listings belonging to each Room Type.")
    #dfPrice=df.query(f"""neighbourhood_cleansed==@neighbourhood""")
  df["HostIsSuperhost"] =  df['host_is_superhost'].apply(superhost)
 
  superhost1 = st.sidebar.radio('Host Is Superhost?',df['HostIsSuperhost'].unique())
  
  dflicense=df.query(f"""neighbourhood_cleansed==@neighbourhood and HostIsSuperhost==@superhost1""")
  dflicense["License Type"] =  df['license'].apply(getLicenseType)
  licensedf=dflicense.groupby(['room_type','License Type' ],as_index=False).agg(CountOfListings=('id', np.size))
  #licensedf
  fig = px.bar(licensedf, title=f"Analysis of Listings by License Type under Different Room Types in **{neighbourhood}**",
              x="room_type",
              y="CountOfListings",
      
              color="License Type", 
              barmode = 'group',
              hover_data=['CountOfListings']).update_layout(
    xaxis_title="Room Type", yaxis_title="Count Of Listings")
  st.plotly_chart(fig, use_container_width= True )


  ################################PROACTIVE AND IDENTITY VERIFIED HOSTS##########################
  def getidentity_ver(row_value):
      if "t" in str(row_value).lower():
        return 'Verified'
      else:
        return 'Not Verified'
      
  st.markdown("Every customer in search of an airbnb looks out for properties from hosts whose identity is verified and who responds **pro-actively**.Let us analyze the number of Properties from the Most Trustworthy and Proactive Hosts in your preferred neighbourhood and Room Type.")
  roomtype = st.selectbox('Choose your preferred Room Type',df['room_type'].unique())
  dfsecure=df.query(f"""neighbourhood_cleansed==@neighbourhood and room_type==@roomtype""")
  dfsecure["Host Identity Verified"] =  dfsecure['host_identity_verified'].apply(getidentity_ver)
  securestaydf=dfsecure.groupby(['host_response_time','Host Identity Verified' ],as_index=False).agg(CountOfListings=('id', np.size))
  #securestaydf
  fig = px.bar(securestaydf, title=f"Properties from the Most Trustworthy and Proactive Hosts for **{roomtype}** in **{neighbourhood}** ",
              x="CountOfListings",
              y="host_response_time",
      
              color="Host Identity Verified", 
              color_discrete_map={"Verified": "blue", "Not Verified":"red"}, 
              #category_orders={"variable":["Revenue","Expenses"]}, 
              barmode = 'group',
              text="CountOfListings",        
              hover_data=['CountOfListings']).update_traces(textposition='outside').update_layout(
    xaxis_title="Number Of Listings", yaxis_title="Host Response Time"
)
  
  st.plotly_chart(fig, use_container_width= True )
  

###############################PAGE = LISTINGS BY TOP HOSTS######################################################################
if page=="Listing By Top Hosts":
  st.markdown("This page will help us to get details of the listings by Top Hosts within the selected Price Range and in the neighbourhood of your choice") 
  #st.text("Select Neighbourhood and Price Range to filter the top hosts in the area based on the number of properties listed.")
              
            
  neighbourhood = st.sidebar.selectbox('Choose Neighbourhood',df['neighbourhood_cleansed'].unique())
  price1 = st.sidebar.slider("Price Range($)", float(df.price.min()), float(df.price.clip(upper=10000.).max()),\
                     (100., 300.))
  #reviews = st.sidebar.slider('Minimum Reviews', 0, 1000, (100))


  tdf1 = df.query(f"""neighbourhood_cleansed==@neighbourhood and price.between{price1}""")
  tdf2=tdf1.groupby(['host_name'],as_index=False).agg(No_of_Listings=('id',np.size))\
      .sort_values('No_of_Listings',ascending=False,ignore_index=True)
  tdf3=tdf2.head(10)

  fig = px.treemap(tdf3,title=f"Top hosts in {neighbourhood} based on Number of listings.", path=['host_name'], values='No_of_Listings',
                  color='host_name', hover_data=['host_name'],
                  color_continuous_scale='RdBu'
                  )
  st.plotly_chart(fig, use_container_width= True )
  
  st.markdown("Use the dropdown to select one of the top 10 hosts and get details of their listings in the map.")
  st.markdown(":pencil: The map will display all the properties by that host and on click of a specific listing ,get a **Pop up details for\
              Room type, Price, Number of Reviews** and a :red[Link] to the :red[original listing page] of Airbnb.com")
  #Click your ideal choice to get all the details of the listings by the top hosts in the area along with a link to the actual listing in airbnb.
  #ophosts = st.selectbox("Choose one of the top hosts",tdf3['host_name'])
  hostname = st.selectbox('Choose Host',tdf3['host_name'].unique())
  tdf4 = tdf1.query(f"""host_name==@hostname""")

  # Define the center of the map and the initial zoom level
  zoom = 15

  # Create a folium map object
  m = folium.Map(location=[tdf4.latitude.mean(), df.longitude.mean()], zoom_start=zoom)

  sw = tdf4[['latitude', 'longitude']].min().values.tolist()
  ne = tdf4[['latitude', 'longitude']].max().values.tolist()
  m.fit_bounds([sw, ne])

  # Add the markers to the map
  for index,loc in tdf4.iterrows():    
      iframe = folium.IFrame("Id: <b>"+  str( int(loc["id"])) + "</b><br> Neighbourhood: <b>"+  str(loc["neighbourhood_cleansed"]) + "</b><br> Name: <b>"+  str(loc["name"])+ "</b><br> Room Type: <b>"+  str(loc["room_type"])+ "</b><br> Avg Price: <b>"+  str(loc["price"]) + "</b><br> Number of Reviews : <b>"+  str(loc["number_of_reviews"])+ "</b><br> Host Name: <b>"+  str(loc["host_name"] +"</b>")
                            + "<br><a href=https://www.airbnb.com/rooms/"+ str( int(loc["id"])) + " target=""_blank"">"+ "Get all details at one click for the listing-"+ str( int(loc["id"])) +"</a>")
      popup = folium.Popup(iframe, min_width=350, max_width=350)
      folium.Marker(location=[loc["latitude"], loc["longitude"]] , tooltip=loc["name"], popup=popup ).add_to(m)

  # Render the map in Streamlit
  st_data =  st_folium(m, width= 725)
