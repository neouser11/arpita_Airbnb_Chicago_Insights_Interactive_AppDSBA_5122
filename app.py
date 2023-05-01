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
#Sentiment Analysis Packages using Text Blob and Vader
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
import re
import seaborn as sns
from matplotlib import style            
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer


st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")

@st.cache_data
def get_data():
  df = pd.read_csv('data/combinedlistings_cleanv2.zip',compression='zip')
  return df

st.title('Airbnb Chicago Insights')
page = st_btn_select(
  # The different pages
  ('Interesting Trends','Listings By Top Hosts','Insights','Sentiment Analyis of Reviews'),
  # Enable navbar
  nav=False
)

df = get_data()
##################  PAGE=Interesting Trends #################################################################################
if page=='Interesting Trends':
  st.markdown("Settled along the banks of Lake Michigan, the Windy City of Chicago offers world-class dining, exciting architecture, a top performing arts scene, and plenty of excellent museums. With more than 200 neighborhoods calling this city home, there’s a unique aura found in each.Let us get an idea  which neighbourhoods have the highest and lowest average Price.")
  df1=df.groupby(['neighbourhood_cleansed'],as_index=False).agg(AveragePrice=('price',np.mean)).sort_values('AveragePrice', ascending=False, ignore_index= True)
  df2= df1.head(5)#st.table(pricedf)
  df3=df.groupby(['neighbourhood_cleansed'],as_index=False).agg(AveragePrice=('price',np.mean)).sort_values('AveragePrice', ignore_index= True)
  df4=df3.head(5)
  
  df5 =  pd.concat([df2, df4], ignore_index= True)
  top5bar = alt.Chart(df5,title="Top 5 and Bottom 5 neighbourhoods by Average Price").mark_bar().encode(
    y= alt.Y('neighbourhood_cleansed:N', title='Neighbourhood', sort = '-x' ),      
    x=alt.X('AveragePrice:Q', title='Average Price'),
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
 
 ###################################Insights Page######################################################
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
  groupedDF = groupedDF.head(10)
  #st.table(groupedDF)
  test = alt.Chart(groupedDF,title=f"Neighbourhoods with Maximum Count of Listings and Customer Reviews between Review score Rating Range:{rating_var}").\
  mark_point().encode(
    x='Average_Number_of_Reviews',
    y='CountOfListings',    
    color=alt.Color('neighbourhood_cleansed',title="Neighbourhood")
  ).interactive()
  st.altair_chart(test, use_container_width=True)

###########################

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

  st.markdown("Select Neighbourhood and SuperHost Filter on the sidebar to find the number of licensed or Unlicensed listings belonging to each Room Type and posted by Superhosts.")
    #dfPrice=df.query(f"""neighbourhood_cleansed==@neighbourhood""")
  df["HostIsSuperhost"] =  df['host_is_superhost'].apply(superhost)
 
  superhost1 = st.sidebar.radio('Host Is Superhost?',df['HostIsSuperhost'].unique())
  
  dflicense=df.query(f"""neighbourhood_cleansed==@neighbourhood and HostIsSuperhost==@superhost1""")
  dflicense["License Type"] =  df['license'].apply(getLicenseType)
  licensedf=dflicense.groupby(['room_type','License Type' ],as_index=False).agg(CountOfListings=('id', np.size))
  #licensedf
  fig = px.bar(licensedf, title=f"Listings belonging to Room Type and License Type in **{neighbourhood}**",
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
      
  st.markdown("Choose Properties from the Most Trustworthy and Proactive Hosts to have a Secure stay.")
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

###############################
  st.markdown("Select Neighbourhood ,Room type to find the Average price for Properties listed by Superhosts and Non-SuperHosts")
  roomtype = st.sidebar.selectbox('Choose preferred Room Type',df['room_type'].unique())
  price=df.query(f"""neighbourhood_cleansed==@neighbourhood #and room_type==@roomtype""")

  def superhost(row_value):
      if "t" in str(row_value).lower():
        return 'SuperHost'
      else:
        return 'Non-SuperHost'
      
  price["Host Is Superhost"] =  price['host_is_superhost'].apply(superhost)
  superdf=price.groupby(['room_type','Host Is Superhost'],as_index=False).agg(AveragePrice=('price',np.mean))                  
  # superdf=price.groupby(['room_type','HostIsSuperhost'],as_index=False).agg(AveragePrice=('price',np.mean)).\
  #                        sort_values('AveragePrice', ascending=False, ignore_index= True)
  #st.table(pricedf)
  fig = px.box(
    data_frame = superdf
    ,y = 'AveragePrice'
    ,x = 'room_type'
    ,color = 'Host Is Superhost'
    ,color_discrete_map={"SuperHost":"blue", "Non-SuperHost":"red"}
    ,category_orders={"Host Is Superhost":("SuperHost", "Non-SuperHost")}
  )
  st.plotly_chart(fig, use_container_width= True )

  fig2 = px.bar(superdf, title=f"Average Price of Properties listed by Superhosts and Non-SuperHosts in {neighbourhood}",\
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

  # bars = alt.Chart(pricedf,title=f"Average Price by Room Type in **{neighbourhood}**").mark_bar().encode(
  #       x= alt.X('room_type:N', title='Room Type', sort = '-y' ),      
  #       y=alt.Y('AveragePrice:Q', title='Average Price')
  #       )
  #st.altair_chart(fig, use_container_width=True)
  


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



###############################PAGE3######################################################################
if page=="Listings By Top Hosts":
  neighbourhood = st.sidebar.selectbox('Choose Neighbourhood',df['neighbourhood_cleansed'].unique())
  price1 = st.sidebar.slider("Price Range($)", float(df.price.min()), float(df.price.clip(upper=10000.).max()),\
                     (100., 300.))
  #reviews = st.sidebar.slider('Minimum Reviews', 0, 1000, (100))


  tdf1 = df.query(f"""neighbourhood_cleansed==@neighbourhood and price.between{price1}""")
  tdf2=tdf1.groupby(['host_name'],as_index=False).agg(No_of_Listings=('id',np.size))\
      .sort_values('No_of_Listings',ascending=False,ignore_index=True)
  tdf3=tdf2.head(10)

  fig = px.treemap(tdf3, path=['host_name'], values='No_of_Listings',
                  color='host_name', hover_data=['host_name'],
                  color_continuous_scale='RdBu'
                  )
  st.plotly_chart(fig, use_container_width= True )
  
  st.markdown("Use the dropdown to select one of the top hosts and get more details of their listings and a link to the actual listing in airbnb")
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

  ###########################SENTIMENT ANALYSIS #########################################################
nltk.download('vader_lexicon')

if page=="Sentiment Analyis of Reviews":
  
  @st.cache_data
  def get_data1():
      df_sa = pd.read_csv("data/reviews22.csv", encoding='latin-1')
      return df_sa

  #df = pd.read_csv("data/reviewsmarch22.csv")
  df_sa=get_data1()
  fdf=df_sa.dropna().reset_index(drop=True)
  df_1 = fdf[['listing_id','comments']]
  #st.write(df_1)


  def data_processing(text):
      text = text.lower()
      text = re.sub(r"https\S+|www\S+https\S+", '',text, flags=re.MULTILINE)
      text = re.sub(r'\@w+|\#','',text)
      text = re.sub(r'[^\w\s]','',text)
      text_tokens = word_tokenize(text)
      filtered_text = [w for w in text_tokens if not w in stop_words]
      return " ".join(filtered_text)

  df_1.comments = df_1.comments.apply(data_processing)

  stemmer = PorterStemmer()
  def stemming(data):
      text = [stemmer.stem(word) for word in data]
      return data

  df_1['comments'] = df_1['comments'].apply(lambda x: stemming(x))

  df_2=df_1
  df_2 = df_2[df_2.comments.str.len() > 3]

  #df_2.head()
  #Sentiment Analysis using VADER 
  analyzer = SentimentIntensityAnalyzer()


  def sentiment_scores(text):
    score = analyzer.polarity_scores(text)
    print("{:-<40} {}".format(text, str(score)))

  df_2['Scores'] = df_2['comments'].apply(sentiment_scores)
  

  df_2['Compound_Score'] = df_2['Scores'].apply(lambda score2: score2['compound'])
  
  def sentimentvader (score):
    if score >= 0.5:
        return 'Positive'
    if (score > 0) and (score < 0.5):
        return 'Neutral'
    if score <= 0:
        return 'Negative'
    
  
  df2['Vader_Sentiment'] = df2['Compound_score'].apply(sentimentvader)

  #Filter the positive comments
  pos_comments = df_2[df_2.Vader_Sentiment == 'Positive']
  pos_comments = pos_comments.sort_values(['Compound_score'], ascending= False)

  #WORD CLOUD OF MOST FREQUENT WORDS IN POSTITIVE REVIEWS

  comments = ' '.join([word for word in pos_comments['comments']])

  neg_com = df_2[df_2.Vader_Sentiment == 'Negative']
  neg_com = neg_com.sort_values(['Compound_score'], ascending= False)
  

  tab1,tab2=st.tabs(['Positive Reviews','Negative Reviews'])

  with tab1:
      #plt.figure(figsize=(20,15), facecolor='None')
      wordcloud = WordCloud(max_words=300, width=1400, height=700).generate(comments)
      plt.imshow(wordcloud, interpolation='bilinear')
      plt.axis("off")
      plt.title('Most frequent words in positive reviews', fontsize=15)
      st.image(wordcloud.to_array())

  with tab2:
     com1 = ' '.join([word for word in neg_com['comments']])
     
     wordcloud = WordCloud(max_words=300, width=1400, height=700).generate(com1)
     plt.imshow(wordcloud, interpolation='bilinear')
     plt.axis("off")
     plt.title('Most frequent words in negative reviews', fontsize=19)
     st.image(wordcloud.to_array())


  
  #Sentiment Analysis using Text Blob 

  # def polarity(text):
  #     return TextBlob(text).sentiment.polarity

  # df_2['Polarity'] = df_2['comments'].apply(polarity)
  # def sentiment(label):
  #     if label <0:
  #         return "Negative"
  #     elif label ==0:
  #         return "Neutral"
  #     elif label>0:
  #         return "Positive"

  # df_2['TB_sentiment'] = df_2['Polarity'].apply(sentiment)
  # #st.write(df_2.head(100))
  # #Filter the positive comments
  # pos_comments = df_2[df_2.TB_sentiment == 'Positive']
  # pos_comments = pos_comments.sort_values(['Polarity'], ascending= False)

  # #WORD CLOUD OF MOST FREQUENT WORDS IN POSTITIVE REVIEWS

  # comments = ' '.join([word for word in pos_comments['comments']])

  # neg_com = df_2[df_2.TB_sentiment == 'Negative']
  # neg_com = neg_com.sort_values(['Polarity'], ascending= False)
  

  # tab1,tab2=st.tabs(['Positive Reviews','Negative Reviews'])

  # with tab1:
  #     #plt.figure(figsize=(20,15), facecolor='None')
  #     wordcloud = WordCloud(max_words=300, width=1400, height=700).generate(comments)
  #     plt.imshow(wordcloud, interpolation='bilinear')
  #     plt.axis("off")
  #     plt.title('Most frequent words in positive reviews', fontsize=15)
  #     st.image(wordcloud.to_array())

  # with tab2:
  #    com1 = ' '.join([word for word in neg_com['comments']])
     
  #    wordcloud = WordCloud(max_words=300, width=1400, height=700).generate(com1)
  #    plt.imshow(wordcloud, interpolation='bilinear')
  #    plt.axis("off")
  #    plt.title('Most frequent words in negative reviews', fontsize=19)
  #    st.image(wordcloud.to_array())





   
