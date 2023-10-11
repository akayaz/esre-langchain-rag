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
    st.session_state.llm_model = st.sidebar.selectbox('Choisissez votre LLM', ['gpt-3.5-turbo-16k', 'gpt-4', 'vertex-ai', 'llama2-70b'], key='selected_model', help='Choisissez votre LLM')
    st.session_state.llm_temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value=0.1, step=0.1, key='llm_temp', help='Controller la créativité du modèle')

st.markdown(
    """
   Build cost-effective and secure AI apps that are accurate and relevant using Elastic’s vector database, out of the box semantic search, and transformer model flexibility. 
   The future is possible today with Elastic..
    **👈 Select a demo from the sidebar** to see some examples of what ESRE can do!
    ### Want to learn more?
    - Check out [Demystifying ChatGPT](https://www.elastic.co/blog/demystifying-chatgpt-methods-building-ai-search)
    - Jump into our [documentation](https://www.elastic.co/guide/index.html)
    - Ask a question in our [community
        forums](https://discuss.elastic.co)
    ### See more Interesting blog Posts on Generative AI
    - Transforming observability with [AI Assistant, OTel standardization, continuous profiling, and enhanced log analytics](https://www.elastic.co/blog/transforming-observability-ai-assistant-otel-standardization-continuous-profiling-log-analytics)
"""
)
