
# carregando as bibliotecas
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import streamlit as st
import re
import nltk
import pickle

#import streamlit.components.v1 as components
import PIL
from PIL import Image
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('rslp')
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


paginas = ['Home', 'Análise Exploratória', "Análise de Sentimentos",'Roadmap do projeto', 'Equipe Koalas']

###### SIDE BAR ######
col1, col2, col3 = st.sidebar.columns([1, 3, 1])
with col2:
    image1 = Image.open('LogoKoalas.png')
    st.image(image1, width=120)
#col1, col2, col3 = st.sidebar.columns([1.5, 3, 1])

    pagina = st.sidebar.selectbox("Navegação", paginas)

###### PAGINA INICIAL ######
if pagina == 'Home':
    
    st.subheader("Análise de sentimentos")
    col1,col2,col3 = st.columns([1,2,3])
    st.write("""
    
    Em meio a recuperação tímida da economia brasileira, o ano de 2018 foi marcado com a greve dos caminhoneiros em protesto contra o aumento diário nos preços do diesel, o dólar cambial e a bolsa de valores sofreram movimentações importantes, motivados pela eleição presidencial e a guerra comercial entre os Estados Unidos da América e a China.

    E a empresa Olist preocupada com a satisfação dos consumidores finais do serviço prestado pelas empresas parceiras  e para a execução de um possível planejamento estratégico, demandou-nos para solucionarmos o problema dela.

    Na análise exploratória, hipóteses diversas foram levantadas para iniciarmos o projeto e após validações e descartes, percebemos padrões de comportamento dos consumidores em relação ao nível de satisfação sobre a aquisição do produto.

    O resultado do projeto poderá ser conferido navegando pelo menu lateral, através das abas “Análise exploratória”, “Análise de sentimento” e “Caminhos do projeto”.
        
    """)
    st.write("Todos os arquivos utilizados poderão ser acessados no repositório do [GitHub](https://github.com/petersonrs/projetostack.git).")






###### BI ######
if pagina == 'Análise Exploratória':
    st.subheader("Análise Exploratória")






###### NLP ######
if pagina == 'Análise de Sentimentos':
    st.markdown("Você poderá analisar o sentimento dos seus clientes carregando um arquivo do tipo csv contendo os comentários.")
    #st.markdown("Estamos implementando a análise de comentários individuais para de teste de usuário.")
    #st.markdown("---")
    col1,col2,col3 = st.columns([1,2,3])
    col1,col2,col3 = st.columns([1,2,3])
    # col1,col2,col3 = st.columns([1,2,3])
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
        total = len(y_pred)
        st.write("Comentários analisados:")
        st.write("Total: ", total)

        negativo = (y_pred ==0).sum()
        print(negativo)
        positivo = (y_pred ==1).sum()
        print(positivo)
        porc_positiva = (positivo/total)*100
        porc_negativa= (negativo/total)*100

        st.write("Positivos (%)*: ", round(porc_positiva,2))
        st.write("Negativos(%)*: ", round(porc_negativa,2))
        st.markdown("*poderá haver um erro de margem de 10pts para cima ou para baixo.")
        col1,col2,col3 = st.columns([1,2,3])
        col1,col2,col3 = st.columns([1,1,4])

    #Inserindo dado novo
    col1,col2,col3= st.columns(3)
    col1,col2,col3= st.columns(3)
    col1,col2,col3= st.columns(3)
    Dado_novo= st.text_input("ou cole/digite um comentário", key="dado_novo")
    Dado_novo = Dado_novo.split(',')
    print(Dado_novo)
    if Dado_novo is not None:

    #### funções ####
        def re_breakline(text_list):
            return [re.sub('[\n\r]', ' ', r) for r in text_list]
        reviews = Dado_novo
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

        #predição do modelo
        y_pred = modelo.predict(reviews_stemmer)
        print(y_pred)
        total = len(y_pred)

        #st.write(y_pred)
        unique, counts = np.unique(y_pred, return_counts= True)
        result = np.column_stack((unique, counts)) 
        print (result)


        negativo = (y_pred ==0).sum()
        positivo = (y_pred ==1).sum()
        porc_positiva = (positivo/total)*100
        porc_negativa= (negativo/total)*100

        
        st.write("Possibilidade de ser positivo: ", round(porc_positiva,2), "%")
        st.write("Possibilidade de ser negativo: ", round(porc_negativa,2), "%")




        
###### ENG. DADOS ######
if pagina=="Roadmap do projeto":
    st.subheader('Nosso trajeto')

    st.markdown("Optamos desenvolver o projeto simulando o cotidiano de um profissional senior, buscando um diferencial no tratamento dos dados.")
    st.markdown("Desafiamo-nos a utilizar o Databricks, uma ferramenta pouco conhecida pela equipe, para orquestrar a nossa engenharia de dados por ser uma solução em cloud baseado em Apache Spark, pois permite  que profissionais de diversas áreas possam trabalhar de forma colaborativa em um único lugar.")
    col1,col2,col3 = st.columns([1,2,3])
    col1,col2,col3 = st.columns([1,2,3])
    col1,col2,col3 = st.columns([0.4,2,1])
    with col2:
            image2 = Image.open("databricks.png")
            st.image(image2, caption='roadmap da engenharia de dados', width=500) 
            
            
    col1,col2,col3 = st.columns([1,2,3])
    st.markdown("A figura acima demonstra o nosso roadmap, que distribuídos todo  o processo em um cluster foi dividido em três fases:")
    st.markdown("- Landing: Recebimento dos dados brutos. É um pequeno pré-processamento e transformação dos arquivos em parquet.")
    st.markdown("- Processing: Todo o trabalho de ETL, normalização dos dados, análise exploratória, pré-processamento e machine learning.")
    st.markdown("- Curated: É o deploy do projeto.")
    



###### EQUIPE ######
if pagina== "Equipe Koalas":
    st.subheader('A Squad')
    col1,col2,col3 = st.columns([1,2,3])
    col1,col2,col3 = st.columns([1,2,3])

# membro1
    col1,col2,col3 = st.columns([1,2,3])
    with col1:
            image3 = Image.open("Clarice.png")
            st.image(image3, width=100)
            col2.markdown('**Clarice Satiko Aoto**')
            col2.write("Data Scientist Jr. | UX Designer Jr.")
            col2.write("[Linkedin](https://www.linkedin.com/in/claricesatikoaoto-bi-python-ux/)")
            
    col1,col2,col3 = st.columns([1,2,3])
    col1,col2,col3 = st.columns([1,2,3])

# membro2
    col1,col2,col3 = st.columns([1,2,3])
    with col1:
            image4 = Image.open("Isis.png")
            st.image(image4, width=100)
            col2.markdown('**Isis Karina de Souza**')
            col2.write("Data Engineer Jr.")
            col2.write("[Linkedin](https://www.linkedin.com/in/isiskarina/)")
            
    col1,col2,col3 = st.columns([1,2,3])
    col1,col2,col3 = st.columns([1,2,3])

# membro3
    col1,col2,col3 = st.columns([1.2,3,3])
    with col1:
            image5 = Image.open("marcos.png")
            st.image(image5, width=100)
            col2.markdown('**Marcos Costa**')
            col2.write("Data Analytics")
            col2.write("[Linkedin](https://www.linkedin.com/in/mcosta7/)")
            
    col1,col2,col3 = st.columns([1,2,3])
    col1,col2,col3 = st.columns([1,2,3])

# membro4
    col1,col2,col3 = st.columns([1.2,3,3])
    with col1:
            image6 = Image.open("Octavio.png")
            st.image(image6, width=100)
            col2.markdown('**Octávio Oliveira**')
            col2.write("Data Scientist | Business Analytics")    
            col2.write("[Linkedin](https://www.linkedin.com/in/octavio-oliveira-56974178/)")
            
    col1,col2,col3 = st.columns([1,2,3])
    col1,col2,col3 = st.columns([1,2,3])

# membro5
    col1,col2,col3 = st.columns([1,2,3])
    with col1:
            image7 = Image.open("Peterson.png")
            st.image(image7, width=100)        
            #st.image("Peterson.png", width=100)
            col2.markdown('**Peterson Silva**')
            col2.write("Data Scientist Senior")  
            col2.write("[Linkedin](https://www.linkedin.com/in/peterson-rosa-silva/)")
            

