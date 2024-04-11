# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import ast
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Film Pre-Production Analyis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


################ movie recommendation from the script ##################
movies=pd.read_csv("movie_data.csv")
movies.dropna(inplace=True)
movies.duplicated().sum()


def convert(obj):
  L= []
  for i in ast.literal_eval(obj):
    L.append(i["name"])
  return L

def convert3(obj):
  L=[]
  counter=0
  for i in ast.literal_eval(obj):
    if counter !=3:
      L.append(i['name'])
      counter+=1
    else:
      break
  return L

def fetch_director(obj):
  L= []
  for i in ast.literal_eval(obj):
    if i['job']=='Director':
      L.append(i["name"])
      break
  return L

def stem(text):
  y=[]

  for i in text.split():
    y.append(ps.stem(i))

  return "".join(y)

def movie_re():
	movies['genres']=movies['genres'].apply(convert)
	# movies['keywords']=movies['keywords'].apply(convert)
	# movies['cast'].apply(convert3)
	# movies['crew'].apply(fetch_director)
	movies['overview'].apply(lambda x:x.split())
	
	movies['genres']=movies['genres'].apply(lambda x:[i.replace("","")for i in x])
	# movies['keywords']=movies['keywords'].apply(lambda x:[i.replace("","")for i in x])
	# movies['cast']=movies['cast'].apply(lambda x:[i.replace("","")for i in x])
	# movies['crew']=movies['crew'].apply(lambda x:[i.replace("","")for i in x])
	
	movies['tags'] = movies['overview'].astype(str) + ' ' + \
	                movies['genres'].astype(str)
	
	movies['tags']= movies['overview'].astype(str) + movies['genres'].astype(str)
	
	new_df=movies[['title','tags','release_date','revenue','runtime','budget']]
	
	new_df['tags']=new_df['tags'].apply(lambda x:"".join(x))
	
	new_df['tags']=new_df['tags'].apply(stem)
	
	cv.fit_transform(new_df['tags']).toarray().shape
	
	vectors = cv.fit_transform(new_df['tags']).toarray()
	
	similarity=cosine_similarity(vectors)



def recommend(movie):
	dic = {}
	movie_index = new_df[new_df['title'] == movie].index[0]
	distances=similarity[movie_index]
	movies_list= sorted(list(enumerate(distances)),reverse=True,key = lambda x: x[1])[1:6]
	
	print('Similar Movie-Script Movies and their Details are')
	
	for i in movies_list:
	     # print(f'Movie Title: {new_df.iloc[i[0]].title}')
	     # print(f'Movie Release Date : {new_df.iloc[i[0]].release_date}')
	     # print(f'Movie Budget: {new_df.iloc[i[0]].budget}')
	     # print(f'Movie Revenue: {new_df.iloc[i[0]].revenue}')
	     # print(f'Movie Runtime (min): {new_df.iloc[i[0]].runtime}')
	     # print('---------------------------------------------------')
	     
	     dic = {'Movie Title': new_df.iloc[i[0]].title,
	     'Movie Release Date' : new_df.iloc[i[0]].release_date,
	     'Movie Budget': new_df.iloc[i[0]].budget,
	     'Movie Revenue': new_df.iloc[i[0]].revenue,
	     'Movie Runtime (min)': new_df.iloc[i[0]].runtime}
	     if df1:
		     df.loc[len(df1)] = dic
	     else:
		     df1 = pd.DataFrame(dic)
	return st.write(df1)


################ movie recommendation from the script-end ##################

def load_data(file):
    # Assuming the file is a CSV for the sake of this example
    df = pd.read_csv(file)
    return df

def visualize_data(df):
    # Replace this with your own visualization logic
    st.write(df)


import pandas as pd

import re

##########----SEGMENTING THE FILM SCRIPTS INTO DIFFERENT SCENES SEGMENTS------#######
##########################################################################
############ EXTRACT OUT SCENES AND CHARACTERS ##########

def extract_scene_characters(filename):

    # read the data into a list (each row is one list element)
    with open(filename, "r", encoding='utf-8', errors='ignore') as f:
        data = [row for row in f]

    dat = []
    for x in data:
        x = re.sub(r'\(.*\)', '', x)
        x = re.sub(r'\-|\#\d+', '', x)
        #x = re.sub(r"[^a-zA-Z0-9.,?'\n ]+", '', x)
        x = re.sub(r"POINT OF VIEW", 'Point of view', x)
        x = re.sub(r"TEXT", 'Text', x)
        x = re.sub(r"NEXT", 'Next', x)
        dat.append(x.replace('\t', ' ').lstrip(" "))

    scenes = []
    for l in dat:
        match = re.search(r'(((INT\.|EXT\.)\s[A-Z]+.*)|((INT\.|EXT\.)\s+[A-Z]+.*)|((INT\.|EXT\.)\s[A-Z]+)|((INT\.|EXT\.)\s[0-9]+.*)|\
        ((INT\./EXT\.|EXT\./INT\.)\s[A-Z]+.*)|((INT\.|EXT\.)\s[0-9]+)|((INT\./EXT\.|EXT\./INT\.)\s[0-9]+.*)|(INT\.\s+.*|EXT\.\s+.*)\
        |((INT\.|EXT\.)\s[A-Z]+\W+.+)|((INT|EXT)\s[A-Z]+.*)|((INT|EXT)\s+[A-Z]+.*)|((INT|EXT)\s[A-Z]+)|((INT|EXT)\s[0-9]+.*)\
        |((INT/EXT|EXT/INT)\s[A-Z]+.*)|((INT|EXT)\s[0-9]+)|((INT/EXT|EXT/INT)\s[0-9]+.*)|((I/E\.|E/I\.)\s+[A-Z].*)\
        |((INT|EXT)\s[A-Z]+\W+.+)|((I/E\.|E/I\.)\s+.*))\n', l)
        if match:
            scenes.append(match.group(1))
    #scenes = [x.strip(" ") for x in scenes]



    characters = []
    for x in dat:
        xters = re.findall(r'(^[A-Z]+[A-Z]+\n)|(^[A-Z]+[A-Z]+\s+\n)|(^[A-Z]+\.\s+[A-Z]+\n)|(^[A-Z]+[A-Z]+\s+[A-Z]+[A-Z]+\s\n)\
        |(^[A-Z]+[A-Z]+\s+[A-Z]+[A-Z]+\s+[A-Z]+[A-Z]+\n)|(^[A-Z]+[A-Z]+\s+[A-Z]+[A-Z]+\n)|(^[A-Z]+[A-Z]+\'S\s+[A-Z]+[A-Z]+\s+[A-Z]+[A-Z]+\n)\
        |(^[A-Z]+[A-Z]+\'S\s+[A-Z]+[A-Z]+\n)|(^[A-Z]+[A-Z]+\'S\s+[A-Z]+[A-Z]+\s+\n)|(^MR\s+[A-Z]+[A-Z]+|MRS\s+[A-Z]+[A-Z]+\n)\
        |(^[A-Z]+[A-Z]+\s+\&\s+[A-Z]+[A-Z]+\n)|(^MR\s+[A-Z]+[A-Z]+|MRS\s+[A-Z]+[A-Z]+\s+\n)', x)
        characters.append(xters)

    characters = [x for x in characters if x != []]
    refined_characters = []
    for c in characters:
        cc = [tuple(filter(None, i)) for i in c]
        refined_characters.append(cc)
    refined_xters = [x[0][0] for x in refined_characters]

    best_ = ['BEST DIRECTOR', 'BEST ADAPTED SCREENPLAY', 'BROADCASTING STATUS', 'BEST COSTUME DESIGN', 'TWENTIETH CENTURY FOX', 'BEST ORIGINAL SCORE', 'BEST ACTOR', 'BEST SUPPORTING ACTOR', 'BEST CINEMATOGRAPHY', 'BEST PRODUCTION DESIGN', 'BEST FILM EDITING', 'BEST SOUND MIXING', 'BEST SOUND EDITING', 'BEST VISUAL EFFECTS']
    transitions = ['RAPID CUT TO:', 'TITLE CARD', 'FINAL SHOOTING SCRIPT', 'CUT TO BLACK', 'CUT TO:', 'SUBTITLE:', 'SMASH TO:', 'BACK TO:', 'FADE OUT:', 'END', 'CUT BACK:', 'CUT BACK', 'DISSOLVE TO:', 'CONTINUED', 'RAPID CUT', 'RAPID CUT TO', 'FADE TO:', \
                   'FADE IN:', 'FADE OUT:', 'FADES TO BLACK', 'FADE TO', 'CUT TO', 'FADE TO BLACK', 'FADE UP:', 'BEAT', 'CONTINUED:', 'FADE IN', \
                   'TO:', 'CLOSE-UP','WIDE ANGLE', 'WIDE ON LANDING', 'THE END', 'FADE OUT','CONTINUED:', 'TITLE:', 'FADE IN','DISSOLVE TO','CUT-TO','CUT TO', 'CUT TO BLACK',\
                   'INTERCUT', 'INSERT','CLOSE UP', 'CLOSE', 'ON THE ROOF', 'BLACK', 'BACK IN SILENCE', 'TIMECUT', 'BACK TO SCENE',\
                   'REVISED', 'PERIOD', 'PROLOGUE', 'TITLE', 'SPLITSCREEN.', 'BLACK.',\
                   'FADE OUT', 'CUT HARD TO:', 'OMITTED', 'DISSOLVE', 'WIDE SHOT', 'NEW ANGLE']
    movie_characters = []
    for x in refined_xters:
        x = re.sub(r'INT\..*|EXT\..*', '', x)
        x = re.sub(r'ANGLE.*', '', x)
        trans = re.compile("({})+".format("|".join(re.escape(c) for c in transitions)))
        x = trans.sub(r'', x)
        best = re.compile("({})+".format("|".join(re.escape(c) for c in best_)))
        x = best.sub(r'', x)
        movie_characters.append(x.replace('\n', '').strip())
    movie_characters = [x.strip() for x in movie_characters if x]

    return scenes, movie_characters




#######################################################################
######## CLEAN THE FILM SCRIPT NOT THOROUGHLY THOUGH ##############

def clean_text(filename):
    """
    Applies some pre-processing on the given text.
    """
    with open(filename, "r", encoding='utf-8', errors='ignore') as r:
        text = [row for row in r]

    #REMOVE TRANSITIONS OR CAMERA ANGLES
    transitions = ['SMASH CUT TO:', 'FINAL SHOOTING SCRIPT', 'CUT TO BLACK', 'SMASH TO:', 'RAPID CUT TO:', 'BACK TO:', 'BLACK SCREEN', 'FADE OUT TO WHITE LIGHT', 'CUT TO:', 'CUT BACK:', 'CUT BACK', 'DISSOLVE TO:', 'CONTINUED', 'RAPID CUT', 'RAPID CUT TO', 'FADE TO:', \
                   'FADE IN:', 'FADES TO BLACK', 'FADE TO', 'CUT TO', 'FADE UP:', 'BEAT', 'AFTERNOON', 'EVENING', 'CONTINUED:', 'FADE IN', \
                   'TO:', 'CLOSE-UP','WIDE ANGLE','CONTINUED:', 'TITLE:', 'FADE IN','DISSOLVE TO','CUT-TO','CUT TO', 'CUT TO BLACK',\
                   'INTERCUT', 'INSERT', 'CLOSE UP', 'TITLE CARD', 'PAUSE', 'SOUND', 'SONG CONTINUES OVER', 'BACK TO SCENE',\
                   'CUT', 'WATCH', 'CU WATCH', 'BLACK', 'BACK IN SILENCE', 'SUBTITLE:', 'CLOSE', 'ON THE ROOF','CUT HARD TO:',\
                   'THE SCREEN', 'TITLE', 'PROLOGUE', 'SPLITSCREEN.', 'OMITTED', 'BLACK.',\
                   'FADE OUT:', 'FADE OUT.', 'FADE OUT', 'DISSOLVE', 'NEW ANGLE', 'WIDE SHOT']
    # remove directors or the film production company
    best_ = ['BEST DIRECTOR', 'BEST ADAPTED SCREENPLAY', 'SENTENCE', 'BROADCASTING STATUS', 'BEST COSTUME DESIGN', 'TWENTIETH CENTURY FOX', 'BEST ORIGINAL SCORE', 'BEST ACTOR', 'BEST SUPPORTING ACTOR', 'BEST CINEMATOGRAPHY', 'BEST PRODUCTION DESIGN', 'BEST FILM EDITING', 'BEST SOUND MIXING', 'BEST SOUND EDITING', 'BEST VISUAL EFFECTS']
    #text = re.sub('\d+', '', text)
    tex = []
    for x in text:
        tx = x.replace('\t', ' ').lstrip(" ")
        tx = re.sub(r'^\d+\n', r'', tx)
        tx = re.sub(r'\(.*\)', r'', tx)
        tx = re.sub(r'\#\d+', r'', tx)
        #tx = tx.replace('\n', '')
        #tx = re.sub(r'\d+', r'', tx)
        tx = re.sub(r'(((INT\.|EXT\.)\s[A-Z]+.*)|((INT\.|EXT\.)\s+[A-Z]+.*)|((INT\.|EXT\.)\s[A-Z]+)|((INT\.|EXT\.)\s[0-9]+.*)|\
        ((INT\./EXT\.|EXT\./INT\.)\s[A-Z]+.*)|((INT\.|EXT\.)\s[0-9]+)|((INT\./EXT\.|EXT\./INT\.)\s[0-9]+.*)|(INT\.\s+.*|EXT\.\s+.*)\
        |((INT\.|EXT\.)\s+[A-Z]+\W+.+)|((INT|EXT)\s+[A-Z]+.*)|((INT|EXT)\s+[A-Z]+.*)|((INT|EXT)\s[A-Z]+)|((INT|EXT)\s[0-9]+.*)|\
        ((INT/EXT|EXT/INT)\s+[A-Z]+.*)|((INT|EXT)\s+[0-9]+)|((INT/EXT|EXT/INT)\s+[0-9]+.*)|((I/E\.|E/I\.)\s+[A-Z].*)\
        |((INT|EXT)\s+[A-Z]+\W+.+)|((I/E\.|E/I\.)\s+.*))', 'SCC', tx)
        tx = re.sub(r'(^\d+\w+\.\s\n)|(^\d+\.\s\n)|(^\d+\.\n)', r'', tx)
        tx = re.sub(r'^\W+', r'', tx)
        tx = re.sub(r'^\d+\.', r'', tx)
        tx = re.sub(r'^\d+/\d+/\d+', r'', tx)
        tx = re.sub(r'ANGLE.*', '', tx)
        tx = re.sub(r'(\'m|\’m)', r' am', tx)
        tx = re.sub(r'(\'ll|\’l)', r' will', tx)
        tx = re.sub(r'(\'re|\’re)', r' are', tx)
        tx = re.sub(r'(\'d|\’d)', r' had', tx)
        tx = re.sub(r'(\'ve|\’ve)', r' have', tx)
        tx = re.sub(r'SEQ\.\s+\d+', r'', tx)
        #tx = re.sub(r'Final\s+\d+\.', r'', tx)
        tx = re.sub(r'Goldenrod\s+\-\s+\d+\.\d+\.\d+\s+\d+\.', r'', tx)
        tx = re.sub(r'(^\d+\s+\d+\s+\d+\s+\-\sRev\.\s\d+/\d+/\d+\s+\d+[A-Z])|(^\d+\s+\d+\s+\d+\s+\-\sRev\.\s\d+/\d+/\d+\s+\d+)', '', tx)
        tx = re.sub(r'([A-Z]+[A-Z]+\sREV\s\d+\-\d+\-\d+\s\d+\.)|([A-Z]+[A-Z]+\sREV\s\d+\-\d+\-\d+\s\d+[A-Z]\.)|(DBL\.\s[A-Z]+[A-Z]+\sREV\s\d+\-\d+\-\d+\s\d+\.)', '', tx)
        #tx = re.sub(r'^TITLE:\n', '', tx)
        #end = re.compile(r'THE END.*|FADE OUT.*', re.MULTILINE)
        #tx = end.sub(r'', tx)
        trans = re.compile("({})+".format("|".join(re.escape(c) for c in transitions)))
        tx = trans.sub(r'', tx)
        #tx = re.sub(r'[A-Z]+\'S', '', tx)
        #tx = tx.replace('[^a-zA-Z]', '')
        #tx = tx.replace('', '')
        #tx = tx.strip()
        #tx = re.sub(r'\d+', r'', tx)
        tx = re.sub(r"[^a-zA-Z0-9.,?'&\n ]+", '', tx)
        #tx = re.sub(r'\W+', ' ', tx)
        tex.append(tx)
    txt = "".join([s for s in tex if s.strip()])
    txt = re.sub(r'\nTHE END\n(.|\n)*', '', txt)

    return txt


######################################################################
######## EXTRACT CHARACTER'S DIALOGUE, SCENE CONTENT AND SCENE ACTION ################

def characters_dialogue_action(text, scenes, movie_characters):

    scene_action = []
    scene_xters = []
    scene_dialogue = []
    xters_list = []
    xters_dialogue = []

    scene_data = text.split('SCC')[1:]

#     scs = re.compile("({})+".format("|".join(re.escape(c) for c in scenes))).split(text)[1:]
#     scene_data = [x for x in scs if x not in scenes]


    for x in scene_data:
        scene_text = re.compile('(\n[A-Z]+[A-Z]+\n)|(\n[A-Z]+[A-Z]+\s+\n)|(\n[A-Z]+\.\s+[A-Z]+\n)|(\n[A-Z]+[A-Z]+\s+[A-Z]+[A-Z]+\s+\n)\
        |(\n[A-Z]+[A-Z]+\s+[A-Z]+[A-Z]+\s+[A-Z]+[A-Z]+\n)|(\n[A-Z]+[A-Z]+\s+[A-Z]+[A-Z]+\n)|(\nMR\s+[A-Z]+[A-Z]+|MRS\s+[A-Z]+[A-Z]+\n)\
        |(\n[A-Z]+[A-Z]+\'S\s+[A-Z]+[A-Z]+\n)|(\n[A-Z]+[A-Z]+\'S\s+[A-Z]+[A-Z]+\s+\n)|(\nMR\s+[A-Z]+[A-Z]+|MRS\s+[A-Z]+[A-Z]+\s+\n)\
        |(\n[A-Z]+[A-Z]+\s+\&\s+[A-Z]+[A-Z]+\n)').split(x)
        scene_text = [x for x in scene_text if x != None]
        xter_split = re.findall('(\n[A-Z]+[A-Z]+\n)|(\n[A-Z]+[A-Z]+\s+\n)|(\n[A-Z]+\.\s+[A-Z]+\n)|(\n[A-Z]+[A-Z]+\s+[A-Z]+[A-Z]+\s+\n)\
        |(\n[A-Z]+[A-Z]+\s+[A-Z]+[A-Z]+\s+[A-Z]+[A-Z]+\n)|(\n[A-Z]+[A-Z]+\s+[A-Z]+[A-Z]+\n)|(\n[A-Z]+[A-Z]+\'S\s+[A-Z]+[A-Z]+\s+[A-Z]+[A-Z]+\n)\
        |(\n[A-Z]+[A-Z]+\'S\s+[A-Z]+[A-Z]+\n)|(\n[A-Z]+[A-Z]+\'S\s+[A-Z]+[A-Z]+\s+\n)|(\nMR\s+[A-Z]+[A-Z]+|MRS\s+[A-Z]+[A-Z]+\s+\n)\
        |(\n[A-Z]+[A-Z]+\s+\&\s+[A-Z]+[A-Z]+\n)|(\nMR\s+[A-Z]+[A-Z]+|MRS\s+[A-Z]+[A-Z]+\n)', x)
        split_ = []
        for c in xter_split:
            cc = tuple(filter(None, c))
            split_.append(cc)
        xters_ = [x[0] for x in split_]
        sc_xters = []
        for r in xters_:
            tr = r.replace('\n', '').strip()
            if tr in movie_characters:
                sc_xters.append(r)
        if scene_text[0] not in sc_xters:
            scene_action.append(scene_text[0])
        else:
            scene_action.append(None)
        if len(sc_xters) >= 1:
            dialogue = []
            xhrs = []
            for z in range(0, len(scene_text),1):
                if scene_text[z] in sc_xters:
                    xters_list.append(scene_text[z])
                    xhrs.append(scene_text[z])
                    xters_dialogue.append(scene_text[z+1])
                    dialogue.append(scene_text[z+1])
                    xters_list = [re.sub(r'\n', r' ', y).strip() for y in xters_list]
                    xhrs = [re.sub(r'\n', r' ', y).strip() for y in xhrs]
                    xters_dialogue = [re.sub(r'\n', r' ', y) for y in xters_dialogue]
                    dialogue = [re.sub(r'\n', r' ', y) for y in dialogue]
            scene_dialogue.append(dialogue)
            scene_xters.append(xhrs)
        else:
            scene_dialogue.append(None)
            scene_xters.append(None)

    scene_data = [re.sub(r'\n', r' ', x) for x in scene_data]
    scene_action = [re.sub(r'\n', r' ', x).strip() for x in scene_action]

    data_tuples = list(zip(scenes, scene_action, scene_xters, scene_dialogue, scene_data))
    df_scene = pd.DataFrame(data_tuples, columns=['Scene_Names','Scene_action', 'Scene_Characters', 'Scene_Dialogue', 'Contents'])

    data_tuples2 = list(zip(xters_list, xters_dialogue))
    df_dialogue = pd.DataFrame(data_tuples2, columns=['characters','Character_dialogue'])
    df_xters = pd.DataFrame(movie_characters, columns = ['characters'])

    return df_scene, df_xters, df_dialogue

def script_breakdown(input):
  sc, movie_xters = extract_scene_characters(input)

  if len(sc) > 1:
      sc_text = clean_text(input)
      df_sc, df_xtrs, df_dia = characters_dialogue_action(sc_text, sc, movie_xters)

      return df_sc




################# page organization ###################
# Define all your pages
def home_page():
    st.title("Home Page")
    st.write("Welcome to our movie analysis app!")

def script_breakdown():
	st.title('🎥 Script Breakdown')
	st.sidebar.header('User Input Features')
	# uploaded_file = st.sidebar.file_uploader("Upload your input file", type=["txt"])
	# Create a file uploader component
	uploaded_files = st.file_uploader("Upload multiple files", accept_multiple_files=True)
	
	# If files are uploaded, display their filenames
	if uploaded_files:
		for uploaded_file in uploaded_files:
			st.write("Filename:", uploaded_file.name)
			output=script_breakdown(uploaded_file.name)
			st.write(output)
		    
	if st.button('download csv'):
		output.to_csv("output.csv")

def market_analysis():
	# Define a list of genres
	genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'Western']
	title = st.text_input("Enter title of your script")
	# Create a select box widget
	selected_genre = st.selectbox('Select your favorite genre:', genres)
	s_plot = st.text_input("please write a short movie plot of your script")
	new_entry = {'title': title, 'genres': selected_genre, 'overview': s_plot}
	df.loc[len(movies)] = new_entry
	movie_re()
	st.title("Similar movie based on the provided script are as follows")
	recommend(title )

def genre_prediction():
    st.title("Genre Prediction")
    st.write("This is where we'll predict the genre of a movie.")

# Create a dictionary of pages
pages = {
    "Home": home_page,
    "Script Breakdown": script_breakdown(),
    "Market Analysis": market_analysis(),
    "Genre Prediction": genre_prediction()
}

# Use radio button on sidebar for navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Display the selected page with the help of our dictionary
# pagesselection

# Add a theme
st.markdown("""
<style>
body {
    color: #fff;
    background-color: #4F8BF9;
}
</style>
    """, unsafe_allow_html=True)



