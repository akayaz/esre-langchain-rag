# import modules
from dotenv import load_dotenv
from langchain.vectorstores import ElasticsearchStore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI, ChatVertexAI
import streamlit as st
import os
import vertexai
from langchain.prompts import PromptTemplate
from elasticsearch import Elasticsearch
from utils import StreamHandler
import time

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Service Public QA",
    page_icon="🔍"
)

# set elastic cloud id and password
CLOUD_ID = os.getenv('CLOUD_ID')
CLOUD_USERNAME = os.getenv('CLOUD_USERNAME')
CLOUD_PASSWORD = os.getenv('CLOUD_PASSWORD')
# ES_VECTOR_INDEX = os.getenv('ES_VECTOR_INDEX')
ES_VECTOR_INDEX = os.getenv('HOSTED_VECTOR_INDEX')

# set vertex ai init params
PROJECT_ID = os.getenv('PROJECT_ID')
REGION = os.getenv('REGION')
VERTEX_AI_MODEL_ID = os.getenv('VERTEX_AI_MODEL')
vertexai.init(project=PROJECT_ID, location=REGION)

# connect to es
es = Elasticsearch(
    cloud_id=CLOUD_ID,
    basic_auth=(CLOUD_USERNAME, CLOUD_PASSWORD)
)
es.info()

# set OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vector_store = ElasticsearchStore(
    es_cloud_id=CLOUD_ID,
    es_user=CLOUD_USERNAME,
    es_password=CLOUD_PASSWORD,
    index_name=ES_VECTOR_INDEX,
    query_field='text_field',
    vector_query_field='vector_query_field.predicted_value',
    distance_strategy="DOT_PRODUCT",
    strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=True,
                                                        query_model_id="sentence-transformers__all-minilm-l6-v2", )
    # embedding=embeddings
)

# Init retriever
retriever = vector_store.as_retriever()



# Build prompt
template = """
    Utilise uniquement les informations suivantes de contexte pour répondre à la question. 
    Si tu ne connais pas la réponse dis: Je ne dispose malheureusement pas d'informations pertinentes pour répondre à cette question, 
    n'essaie pas d'inventer une réponse. Utilise au maximum 10 phrases. 
    Sois clair, concis et précis dans tes réponses. 
    Termine tes réponses par "J'espere avoir répondu à votre question." sauf lorsque tu ne connais pas la réponse. 
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


def init_qa_chain(llm_model, llm_temperature):
    llm = None
    if st.session_state.llm_model == 'gpt-3.5-turbo-16k' or 'gpt4':
        llm = ChatOpenAI(
            model_name=st.session_state.llm_model,
            temperature=st.session_state.llm_temperature,
            openai_api_key=OPENAI_API_KEY)
    if st.session_state.llm_model == 'vertex-ai':
        llm = ChatVertexAI(
            model_name="chat-bison",
            max_output_tokens=256,
            temperature=st.session_state.llm_temperature,
            top_p=0.8,
            top_k=40,
            verbose=True,
        )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

st.title("Service Public QA Demo")

# chat interface
with st.container():
    st.write("Paramètres actuel du Chatbot")
    st.write("Modèle LLM: ", st.session_state.llm_model)
    st.write("Température: ", st.session_state.llm_temperature)

if question := st.text_input("Posez votre question"):
    qa = init_qa_chain(llm_model=st.session_state.llm_model, llm_temperature=st.session_state.llm_temperature)
    with st.chat_message("assistant"):
        st.spinner("Recherche de la réponse...")
        message_placeholder = st.empty()
        full_response = ""
        #stream_handler = StreamHandler(st.empty())
        response = qa({"query": question})
        for chunk in response['result'].split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.write(full_response)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        st.markdown(""" ##### Sources: """)
        for docs_source in response['source_documents']:
            #st.markdown(" %s " %docs_source.metadata['url'])
            link = f'<a href="{docs_source.metadata["url"]}" target="_blank">{docs_source.metadata["title"]}</a>'
            st.markdown(link, unsafe_allow_html=True)
        # st.markdown(""" ###### LLM: """ + st.session_state.llm_model)

