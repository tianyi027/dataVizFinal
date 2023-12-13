#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 21:32:56 2023

@author: tianyiluo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
import plotly.express as px
#pio.renderers.default='browser'
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.figure_factory as ff
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler


#Import the dataset from the Kaggle, stored as imdb
imdb = pd.read_csv('/Users/tianyiluo/CST383/imdb_top_1000.csv')

#Data cleaning
#drop first column of the dataset, Poster_Link, which is extra information and does not contribute to
#our works
imdb_clean = imdb.drop('Poster_Link', axis=1)

#Eliminate the min on the Runtime column for cleaning purpose
imdb_clean['Runtime'] = imdb_clean['Runtime'].str.replace(' min', '', regex=False).astype(int)




# One error message from the original data, the released year was mislabeled as PG instead of
# a number. After checking it online to comfirm the year, 1939 should be the correct value here
# Find the row with the incorrect year
incorrect_row = imdb[imdb['Released_Year'] == 'PG']
imdb_clean.loc[imdb_clean['Released_Year'] == 'PG', 'Released_Year'] = '1939'

imdb_clean['Released_Year'] = imdb_clean['Released_Year'].astype(int)

#Create a new column called decade that can represent films in longer period
imdb_clean['decade'] = (imdb_clean['Released_Year'] // 10) * 10

# ????? consider redo this part, for now, drop all the missing values of the df
imdb_clean.dropna(how = 'all', axis = 1, inplace = True)

#Convert the Gross value from string to float without comma in between
imdb_clean['Gross'] = imdb_clean['Gross'].str.replace(',', '')
imdb_clean['Gross'] = imdb_clean['Gross'].astype(float)

#Calculate how much percent of the value of Gross (Box office) is missing value, which is 169 films.
imdb_clean['Gross'].isna().sum()

#Use negation symbol operates as a boolean value to select everything value without a missing value
imdb_clean2 = imdb_clean[~imdb_clean['Gross'].isna()]

#------------------------------------------\
    
    
    
    
    #Data Viz
    
    
    
    #Goal1: Stars + Gross/Rating relationship -> who(stars) should I pick if I want to watch a GOOD movie
    #Goal2: Runtime + genre + Gross/Rating relationship -> If one needs to shoot a file, for what length of the 
            # movie and the genre they should choose to make more profits
    #Goal3: Does good imdb scores transport to a good movie? Can we predict what revenue 
            # for future movies depend on genre/ length/ stars
            
    # Interactive dashboard and clear annotation for each figure, make sure to satisfiy all the rubric
    # Pictures if we have time



# Assuming your DataFrame is named 'imdb'
# Convert 'Released_Year' to integers if they are not already

# Calculate the decade

#find correlationship



# Calculate the correlation matrix
#correlation_matrix = selected_columns.corr()


#imdb_clean.corr(), seeking for the correlationship should be one goal

# b2 = sns.catplot(kind='bar', data= imdb_clean, x= 'decade', y = 'IMDB_Rating', ci = None)

#x1 = px.scatter(imdb_clean, x= 'decade', y = 'IMDB_Rating', size = 'Gross')
#x1.show()


# Data source; IMDB Dataset from Kaggle: https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows/
# any prerequisite data cleaning: 1. Create cleaned dataframe imdb by eliminating unnecessary columns: drop na values, generating features of decade, 
                                #    select genre with a matched way, impute missing values, convert datatype to numerical format, drop 
                                # 2. Selecting good python packages to do visualization: pyplot 
                                # 3. Dataframe manipulation on pandas

# questions                     
                                # Within Top 1000 Movies by IMDB Rating                       
#                               # 1. Are movies receving higher/lower ratings because they are produced in different decade?
                                # 2. What kind of genre have higher ratings/ higher gross?
                                # 3. Are famous/top directors produce filmwork with higher rating and higher gross
# a picture of the movie 'fight club'
                                # called finalproject.jpeg



x2 = px.scatter(imdb_clean, x= 'decade', y = 'IMDB_Rating', size = 'Runtime', color = 'Gross')

# x2.show()

#list comprehension to select relevant columns to display the hoverdata
hoverdata = [call for call in imdb_clean.columns if call not in ['decade', 'IMDB_Rating' , 'Overview',
                                                                 'Gross_clean', 'Meta_score',
                                                                 'Star2', 'Star3', 'Star4',
                                                                 'Certificate']]
#Calculate the mean value of all the films on the imdb list
film_importance = imdb_clean['IMDB_Rating'].mean()

#Calculate the 75th precentile line of imdb rating of all films
percent_75 = np.percentile(imdb_clean['IMDB_Rating'], 75)

#Create the plotly scatter plot for decade and imdb rating with hoverdata information
trace_line = px.scatter(imdb_clean2, 
                        x=imdb_clean2['Released_Year'], 
                        y=imdb_clean2['IMDB_Rating'],
                        color=imdb_clean2['IMDB_Rating'],
                        size=imdb_clean2['Gross'],
                        # hoverdata='text', 
                        template = 'plotly_white',
                        #animation_frame=()
                        hover_data=imdb_clean2[hoverdata])                                         
                             
#Create a second plotly object, a red line, represent the average value of all movies on the list                            
mean_line = go.Scatter(x = imdb_clean['decade'], y = [film_importance] * len(imdb_clean),
                       mode = 'lines', 
                       name = f'Average IMDB Rating: {round(film_importance, 3)}')

#Create the third plotly object, a green line, represnet the 75th percentile of all films on the list
line_75 = go.Scatter(x = imdb_clean['decade'], y = [percent_75] * len(imdb_clean), 
                     mode = 'lines',
                     line = dict(color = 'green'),
                     name = f'75th percentile of the IMDB Rating: {round(percent_75, 3)}')

#Add the second component, mean line, to the original scatter plot
trace_line.add_trace(mean_line)

#Add the third component, 75th percentile, to the scatter plot
trace_line.add_trace(line_75)

#Add the detailed extra layout information of title, x and y axis title and legend.                         
trace_line.update_layout(title='Movie IMDB rating on different Decades',
                         xaxis_title = 'Decade',
                         yaxis_title = 'Ratings on IMDB',
                         legend_title = 'Legend',
                         width = 1200,
                         height = 800,
                         legend = dict(x=1.5, y=1, xanchor='left', yanchor='top'))
                        
#imdb_clean['Genre'] = imdb_clean['Genre'].str.split(',', 1)

#Write a function to split the genre column after the first comma and keeps the first value,
#which helps me to categorize the primary genre of each film since many of them have mult-genre
def keepfirst (Genre):
    return Genre.split(',', 1)[0]

#Apply the keepfirst function as it is the best optimazation method 
imdb_clean['Genre_clean'] = imdb_clean['Genre'].apply(keepfirst)


#plot 2
#catplot, countplot, palette
plot2 = px.histogram(imdb_clean, 
                     x='IMDB_Rating', 
                     barmode='group', 
                     nbins=50)



# Update layout
plot2.update_layout(
    title='IMDB Rating: Histogram Plot',
    xaxis_title='IMDB Rating',
    yaxis_title='Probability'
)

def main():

    menu = ["Introduction","rating vs decade",'ratings by genre', 'does star matter?']

    choice = option_menu(
            menu_title="Main Menu", options= menu,
            icons=['view-list', 'vinyl', 'patch-exclamation', 'fast-forward'],
            menu_icon = "chevron-double-left", default_index = 0, orientation = "vertical")

    df_genre = imdb_clean.groupby('Genre_clean')['IMDB_Rating'].mean()

    highest_ratings = df_genre.sort_values(ascending= False)
    lowest_ratings = df_genre.sort_values(ascending= True)

    #st.write(df_genre.head())

    if choice == 'Introduction':
        st.image('finalproject.jpeg')
        st.write(''' Data source; IMDB Dataset from Kaggle: \n https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows/''')
                     
        st.write(''' Prerequisite data cleaning: 
                    \n1. Create cleaned dataframe imdb by eliminating unnecessary columns: drop na values, generating features of decade, 
                           select genre with a matched way, impute missing values, convert datatype to numerical format, drop 
                    \n2. Selecting good python packages to do visualization: pyplot 
                    \n3. Dataframe manipulation on pandas
                    ''')
                    
        st.write(''' Questions: Within Top 1000 Movies by IMDB Rating                                          
                    \n1. Are movies receving higher/lower ratings because they are produced in different decade?
                    \n2. What kind of genre have higher ratings/ higher gross?
                    \n3. Are famous/top directors produce filmwork with higher rating and higher gross
                 ''')
        #st.plotly_chart(x2)
        file_name = "plot1.html" 
       # fig.write_html(file_name)


    elif choice == 'rating vs decade':
        st.plotly_chart(trace_line)
        
        file_name = "plot1.html" 
        trace_line.write_html(file_name)
    
    elif choice == 'ratings by genre':

        col1, col2 = st.columns(2)
        with col1:
            st.write('Top 5 Genre by imdb rating')
            st.write(highest_ratings.head())
    
        with col2:
            st.write('Bottom 5 Genre by imdb rating')
            st.write(lowest_ratings.head())

        selection = st.radio('options', ['overall', 'by genre'])
        if selection == 'overall':

            plot2 = px.histogram(imdb_clean, 
                     x='IMDB_Rating', 
                     barmode='group', 
                     nbins=50)
        
            # Update layout
            plot2.update_layout(
                title='IMDB Rating: Histogram Plot',
                xaxis_title='IMDB Rating',
                yaxis_title='Frequency'
            )

            plot2_gross = px.bar(imdb_clean, 
                                  x= 'Genre_clean',
                                  y = 'Gross',
                                  color = 'Genre_clean'
                                
                                )
            
            plot2_gross.update_layout(showlegend=False)

            # Update layout
            plot2_gross.update_layout(
                title=f'Gross by Genre Type',
                xaxis_title='IMDB Rating',
                yaxis_title=f'Box office (dollar)'               
            )

            st.plotly_chart(plot2_gross)  

            st.plotly_chart(plot2)

            file_name = "plot2.html" 
            plot2.write_html(file_name)
           

        else:

            genre_choice = st.selectbox('Select Genre', imdb_clean['Genre_clean'].unique())

            plot_data = imdb_clean[imdb_clean['Genre_clean'] == genre_choice]
            

            #filtered_genre = imdb_clean['Genre_clean'].value_counts() < 30
            #imdb_clean[filtered_genre]

            unique_genres = plot_data['IMDB_Rating'].shape[0]
            plot2 = px.histogram(plot_data, 
                        x='IMDB_Rating', 
                        barmode='group', 
                        nbins=50)
            
            
            

            # Update layout
            plot2.update_layout(
                title=f'IMDB Rating: {genre_choice} Histogram Plot',
                xaxis_title='IMDB Rating',
                yaxis_title=f'Frequency'
            )
            st.plotly_chart(plot2)

            plot2_gross = px.line(plot_data, 
                        x= 'Genre_clean',
                        y = 'Gross'
                        )

            # Update layout
            plot2_gross.update_layout(
                title=f'IMDB Rating: {genre_choice} Histogram Plot',
                xaxis_title='IMDB Rating',
                yaxis_title=f'Frequency'
            )

            file_name = "plot2.html" 
            plot2.write_html(file_name)
    
    else:
        #director_choice = st.selectbox('select director choice', 
        #                               imdb_clean['Director'].unique())
        
        #Query can alos do it: imdb_clean.query(director == director_choice)
        #director_dataset = imdb_clean[imdb_clean['Director']==director_choice][['No_of_Votes', 'Gross']]

        #directorplot = px.imshow(director_dataset[['Gross']], text_auto=True)
        #st.plotly_chart(directorplot)
        #heatmap = go.Heatmap(z = director_dataset.values, x=director_dataset.columns, y = director_dataset.index)
        #fig = go.Figure(data=[heatmap])
        #st.plotly_chart(fig)

        top_10_d = imdb_clean['Director'].value_counts()[:10]

    #---
    # Step 1: Bin 'IMDB_Rating'

        select_director = imdb_clean[imdb_clean['Director'].isin(top_10_d.index)]
        
        rating_bins = [7, 7.5, 8, 8.5, 9, 9.5]  # Adjust these values as needed
        rating_labels = ['7-7.5', '7.5-8', '8-8.5', '8.5-9', '9.5']
        select_director['rating_bin'] = pd.cut(select_director['IMDB_Rating'], bins=rating_bins, labels=rating_labels)

        # Step 2: Aggregate Data
        # Calculate the average (or median) Gross for each Director within each Rating bin
        heatmap_data = select_director.groupby(['Director', 'rating_bin']).Gross.mean().reset_index()

        # Step 3: Create Heatmap
        # Adjust the x and y axis to reflect the change in variable positions
        fig = px.density_heatmap(heatmap_data, x='rating_bin', y='Director', z='Gross', 
                                color_continuous_scale='Viridis')

        # Customize the layout
        fig.update_layout(
            title='Average Gross Revenue by Director and IMDB Rating Bins',
            xaxis_title='IMDB Rating Bin',
            yaxis_title='Director'
        )

        
        st.plotly_chart(fig)
        file_name = "plot3.html" 
        fig.write_html(file_name)

        if st.button('Director fun facts: '):
            st.markdown('* :balloon: Spielberg made the highest grossing movie ever 3 times!')
            st.markdown('* :balloon: Nolan is the first British director to have a film make one billion dollars at the box office with The Dark Knight')


    #---

        #top_10_plot = px.imshow(top_10_d[['Gross']], text_auto=True)
        #st.plotly_chart(top_10_plot)


        #directors, box office, rating, voting of each film -> of a director
            
        #ctrl c, stops; streamlit run finalproject.py -> run the file



#which genre has the highest rates on average, which is the lowest. 

# 1, show the average line of those on the overall plot

# 2, highest box office and lowest box office 

# so, show 4 lines (or less than 4) on the main plot

# add some annotation to show why its popular as side information. 


# directors vs. box offic; vs. ratings; vs. voting. 



main()


#file_name = "plotly_heatmap.html" 
#fig.write_html(file_name)







