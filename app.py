
# carregando as bibliotecas
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import streamlit as st
import re
import nltk
import pickle

#import streamlit.components.v1 as components

from PIL import Image
from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk import word_tokenize
from nltk.stem import RSLPStemmer
#from nltk.tokenize import word_tokenize
from nltk import FreqDist

import mglearn

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


st.set_page_config(
    page_title="Koalas",
    page_icon="🐨",
    layout="centered",
    initial_sidebar_state='auto',
    menu_items=None)


paginas = ['Home', 'Business Intelligence', 'Engenharia de Dados', 'Equipe Koalas']

col1, col2, col3 = st.sidebar.columns([1, 3, 1])
with col2:
    st.image("logoKoalas.png", width=120)
col1, col2, col3 = st.sidebar.columns([1.5, 3, 1])

pagina = st.sidebar.selectbox("Navegação", paginas)


if pagina == 'Home':
    
    st.subheader("Análise de sentimentos")
    col1,col2,col3 = st.columns([1,2,3])
    st.markdown("Vamos analisar o sentimento dos clientes?")

    col1,col2,col3 = st.columns([1,2,3])
    uploaded_file = st.file_uploader("escolha um arquivo *.csv")
    if uploaded_file is not None:
        df2 = pd.read_csv(uploaded_file)
        df2=df2['review'].to_list()
        print(df2) # checar a saída no terminal

        # Funções
        def re_breakline(text_list):
            return [re.sub('[\n\r]', ' ', r) for r in text_list]
        reviews = df2
        reviews_breakline = re_breakline(reviews)
        
        def re_hiperlinks(text_list):
            pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            return [re.sub(pattern, ' link ', r) for r in text_list]
        reviews_hiperlinks = re_hiperlinks(reviews_breakline)
        
        def re_dates(text_list):
            pattern = '([0-2][0-9]|(3)[0-1])(\/|\.)(((0)[0-9])|((1)[0-2]))(\/|\.)\d{2,4}'
            return [re.sub(pattern, ' data ', r) for r in text_list]
        reviews_dates = re_dates(reviews_breakline)

        def re_money(text_list):
            pattern = '[R]{0,1}\$[ ]{0,}\d+(,|\.)\d+'
            return [re.sub(pattern, ' dinheiro ', r) for r in text_list]
        reviews_money = re_money(reviews_dates)

        def re_numbers(text_list):
            return [re.sub('[0-9]+', ' numero ', r) for r in text_list]
        reviews_numbers = re_numbers(reviews_money)

        def re_negation(text_list):
            return [re.sub('([nN][ãÃaA][oO]|[ñÑ]| [nN] )', ' negação ', r) for r in text_list]
        reviews_negation = re_negation(reviews_numbers)

        def re_special_chars(text_list):
            return [re.sub('\W', ' ', r) for r in text_list]
        reviews_special_chars = re_special_chars(reviews_negation)

        def re_whitespaces(text_list):
            white_spaces = [re.sub('\s+', ' ', r) for r in text_list]
            white_spaces_end = [re.sub('[ \t]+$', '', r) for r in white_spaces]
            return white_spaces_end
        reviews_whitespaces = re_whitespaces(reviews_special_chars)

        def stopwords_removal(text, cached_stopwords=stopwords.words('portuguese')):
            return [c.lower() for c in text.split() if c.lower() not in cached_stopwords]
        reviews_stopwords = [' '.join(stopwords_removal(review)) for review in reviews_whitespaces]

        def stemming_process(text, stemmer=RSLPStemmer()):
            return [stemmer.stem(c) for c in text.split()]
        reviews_stemmer = [' '.join(stemming_process(review)) for review in reviews_stopwords]
        print(reviews_stemmer)

        #carregando o modelo de predição
        modelo = pickle.load(open('3modelo20220127.pkl','rb'))
        y_pred = modelo.predict(reviews_stemmer)
        # for c in y_pred:
        #     print(c)


        total = len(y_pred)
        #print(total)
        st.write("Comentários analisados:")
        st.write("Total: ", total)

        negativo = (y_pred ==0).sum()
        print(negativo)
        positivo = (y_pred ==1).sum()
        print(positivo)
        porc_positiva = (positivo/total)*100
        porc_negativa= (negativo/total)*100

        st.write("Positivos (%): ", round(porc_positiva,2))
        st.write("Negativos(%): ", round(porc_negativa,2))
        col1,col2,col3 = st.columns([1,2,3])
        col1,col2,col3 = st.columns([1,1,4])







if pagina == 'Business Intelligence':
    st.subheader("Business Intelligence")

    # src = "https://minerandodados.com.br/como-salvar-um-modelo-de-machine-learning-em-disco/"
    # st.components.v1.iframe(src, width=1000, height=None, scrolling=False)

import streamlit as st
import streamlit.components.v1 as components

# # bootstrap 4 collapse example
# components.html(
#     """
#     <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
#     <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
#     <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
#     <div id="accordion">
#       <div class="card">
#         <div class="card-header" id="headingOne">
#           <h5 class="mb-0">
#             <button class="btn btn-link" data-toggle="collapse" data-target="#collapseOne" aria-expanded="true" aria-controls="collapseOne">
#             Collapsible Group Item #1
#             </button>
#           </h5>
#         </div>
#         <div id="collapseOne" class="collapse show" aria-labelledby="headingOne" data-parent="#accordion">
#           <div class="card-body">
#             Collapsible Group Item #1 content
#           </div>
#         </div>
#       </div>
#       <div class="card">
#         <div class="card-header" id="headingTwo">
#           <h5 class="mb-0">
#             <button class="btn btn-link collapsed" data-toggle="collapse" data-target="#collapseTwo" aria-expanded="false" aria-controls="collapseTwo">
#             Collapsible Group Item #2
#             </button>
#           </h5>
#         </div>
#         <div id="collapseTwo" class="collapse" aria-labelledby="headingTwo" data-parent="#accordion">
#           <div class="card-body">
#             Collapsible Group Item #2 content
#           </div>
#         </div>
#       </div>
#     </div>
#     """,
#     height=600,
# )








if pagina=="Engenharia de Dados":
    st.subheader('Engenharia de Dados')

    st.markdown("A squad Koalas optou desenvolver o projeto simulando o cotidiano de um profissional senior, buscando um diferencial no tratamento dos dados.")
    st.markdown("Desafiamo-nos a utilizar o Databricks, uma ferramenta pouco conhecida pela squad, para orquestrar a nossa engenharia de dados por ser uma solução em cloud baseado em Apache Spark, pois permite  que profissionais de diversas áreas possam trabalhar de forma colaborativa em um único lugar.")
    col1,col2,col3 = st.columns([1,2,3])
    col1,col2,col3 = st.columns([1,2,3])
    # col1,col2 = st.columns(2,2)
    # with col2:
    st.image("databricks.png", width=500) 
            # col3.markdown('**Pedir para aumentar a arte**')
            
    col1,col2,col3 = st.columns([1,2,3])
    st.markdown("A figura acima demonstra o nosso roadmap, que distribuídos todo  o processo em um cluster foi dividido em três fases:")
    st.markdown("- Landing: Recebimento dos dados brutos. É um pequeno pré-processamento e transformação dos arquivos em parquet.")
    st.markdown("- Processing: Todo o trabalho de ETL, normalização dos dados, análise exploratória, pré-processamento e machine learning.")
    st.markdown("- Curated: É o deploy do projeto.")
    




if pagina== "Equipe Koalas":
    st.subheader('A Equipe')
    col1,col2,col3 = st.columns([1,2,3])
    col1,col2,col3 = st.columns([1,2,3])

# membro1
    col1,col2,col3 = st.columns([1,2,3])
    with col1:
            st.image("Clarice.png", width=100)
            col2.markdown('**Clarice Satiko Aoto**')
            col2.write("Data Scientist Jr. | UX Designer Jr.")
            col2.write("[Linkedin](https://www.linkedin.com/in/claricesatikoaoto-bi-python-ux/)")
            
    col1,col2,col3 = st.columns([1,2,3])
    col1,col2,col3 = st.columns([1,2,3])

# membro2
    col1,col2,col3 = st.columns([1,2,3])
    with col1:
            st.image("Isis.png", width=100)
            col2.markdown('**Isis Karina de Souza**')
            col2.write("Data Engineer Jr.")
            col2.write("[Linkedin](https://www.linkedin.com/in/isiskarina/)")
            
    col1,col2,col3 = st.columns([1,2,3])
    col1,col2,col3 = st.columns([1,2,3])

# membro3
    col1,col2,col3 = st.columns([1.2,3,3])
    with col1:
            st.image("marcos.png", width=100)
            col2.markdown('**Marcos Costa**')
            col2.write("Data Analytics")
            col2.write("[Linkedin](https://www.linkedin.com/in/mcosta7/)")
            
    col1,col2,col3 = st.columns([1,2,3])
    col1,col2,col3 = st.columns([1,2,3])

# membro4
    col1,col2,col3 = st.columns([1.2,3,3])
    with col1:
            st.image("Octavio.png", width=100)
            col2.markdown('**Octávio Oliveira**')
            col2.write("Data Scientist | Business Analytics")    
            col2.write("[Linkedin](https://www.linkedin.com/in/octavio-oliveira-56974178/)")
            
    col1,col2,col3 = st.columns([1,2,3])
    col1,col2,col3 = st.columns([1,2,3])

# membro5
    col1,col2,col3 = st.columns([1,2,3])
    with col1:
            st.image("Peterson.png", width=100)
            col2.markdown('**Peterson Silva**')
            col2.write("Data Scientist Senior")  
            col2.write("[Linkedin](https://www.linkedin.com/in/peterson-rosa-silva/)")
            
    # components.html(
    #    """
    #    <style>
    #    h3{
    #        padding-button: -30px;
    #    }
    #    </style>
    
    #    """
    #  )
