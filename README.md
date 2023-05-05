Airbnb Chicago Insights 
This application has been designed to provide useful insights to guests who are looking for an Airbnb in various neighborhoods of Chicago as well as for New hosts who want to Airbnb their properties and get the most benefit out of their investment.
 
Real time data is collected from http://insideairbnb.com/get-the-data 

The app provides information across four broad areas -Pricing and listing Trends over the years, Listing details by the top hosts, Super hosts vs non-Super hosts and Best Neighborhoods and Most Reliable and Trustworthy listings.

Data Preprocessing has been done using python in Jupyter Notebook: Steps include feature selection, feature engineering and data cleaning to transform variables like price, host response rate, host acceptance rate, etc. to the desired format for analysis and visualizations. Mean and mode Imputation for numerical features and One-Hot encoding for categorical variables have been performed as part of preprocessing.

The Interesting Trends page shows the Top 5 and Bottom 5 neighborhoods by Average Price out of more than 200 neighborhoods. Hosts can find which property type has increased number of listings over the years and can make a judgement call on which type of property to invest in and what price to quote.

The page for Listings by Top Hosts will help the guests to get details of the listings by Top Hosts within the selected Price Range and in the neighborhood of choice. Everyone wants properties listed by top hosts! To make this happen, Airbnb Guests can choose one of the top hosts displayed in the tree map and get details of their listings in the interactive map created using folium. The map displays all the properties by the selected top host and on click of a specific listing Pops up details for Room type, Price, Number of Reviews and a Link to the original listing page of Airbnb.com.

Superhost VS Non SuperHost page gives insights into the differences in price ,number of listings and average reviews between the two groups. Find out if Super hosts are charging more compared to Non-Superhosts across different property types in chosen neighborhoods ? Does Superhosts lists more number of properties compared to Non-Superhosts? Does Superhosts gets more reviews compared to Non-Superhosts? 

Best Neighbourhoods + Listings page can help the guests who are looking for Neighbourhoods that have More Supply of listings as well as more Customer Reviews. 
All guests look for licensed Airbnb properties but there might be some which are unlicensed or pending. The grouped bar chart using plotly shows the distribution of Licensed, Unlicensed, Pending or Exempt listings across different Room types and areas. 
Every customer in search of an airbnb looks out for properties from hosts whose identity is verified and who responds pro-actively.This page finds the listings from the Most Trustworthy and Proactive Hosts in your preferred neighbourhood and Room Type.

Streamlit app URL:



Future work: Perform sentiment analysis of the reviews using Vader sentiment analyzer to extract insights and use them as part of predictive modeling. This will give the hosts an idea of the areas where they are performing well as well as areas of improvement.
