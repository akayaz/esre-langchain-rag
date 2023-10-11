import streamlit as st

st.set_page_config(
    page_title="Elasticsearch Retrieval Engine (ESRE) LangChain openAI chatbot",
    layout="centered",
    page_icon="👋",
    initial_sidebar_state="expanded",
    menu_items={
        'About': 'This App shows you how to use Elasticsearch with LLM'
    }
)

# creating sessions state


st.image(
    'https://images.contentstack.io/v3/assets/bltefdd0b53724fa2ce/blt601c406b0b5af740/620577381692951393fdf8d6/elastic-logo-cluster.svg',
    width=75)

st.header("Chat with your Elasticsearch data using Large Language Models (LLMs)")


st.sidebar.header('Choose a demo')


with st.sidebar:
    st.subheader('Choix du LLM et des paramètres')
    st.session_state.llm_model = st.sidebar.selectbox('Choisissez votre LLM', ['gpt-3.5-turbo-16k', 'gpt-4', 'vertex-ai'], key='selected_model', help='Choisissez votre LLM')
    st.session_state.llm_temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value=0.0, step=0.1, key='llm_temp', help='Controller la créativité du modèle')

