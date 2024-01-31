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
from langchain.embeddings import HuggingFaceEmbeddings
import time
import elasticapm
import boto3
from langchain.llms import Bedrock
from langchain.llms import Ollama
from langchain.chat_models import AzureChatOpenAI

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Stellantis Assistant",
    page_icon=":car:"
)
st.image(
    #'https://images.contentstack.io/v3/assets/bltefdd0b53724fa2ce/blt601c406b0b5af740/620577381692951393fdf8d6/elastic-logo-cluster.svg',
    "/Users/yakadiri/Downloads/aws-elastic.png",
    width=250)
st.title("Assistant Stellantis ESRE Demo")
st.sidebar.header('Configuration du Chatbot')
with st.sidebar:
    st.subheader('Choix du LLM et des paramètres')
    st.write("Paramètres du Chatbot")
    st.session_state.llm_model = st.sidebar.selectbox('Choisissez votre LLM',
                                                      ["azure-openai", 'Ollama', 'bedrock',], key='selected_model',
                                                      help='Choisissez votre LLM')
    st.session_state.llm_temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value=0.1,
                                                         step=0.1, key='llm_temp',
                                                         help='Controller la précision du modèle')

@st.cache_resource
def initAPM():
    apm_client = elasticapm.Client(
        service_name=os.getenv('ELASTIC_APM_SERVICE_NAME'),
        server_url=os.getenv('ELASTIC_APM_SERVER_URL'),
        secret_token=os.getenv('ELASTIC_APM_SECRET_TOKEN'),
    )
    elasticapm.instrument()
    return apm_client

apm_client= initAPM()

# set elastic cloud id and password
CLOUD_ID = os.getenv('VECTOR_CLOUD_ID')
CLOUD_USERNAME = os.getenv('VECTOR_CLOUD_USERNAME')
CLOUD_PASSWORD = os.getenv('VECTOR_CLOUD_PASSWORD')
#ES_VECTOR_INDEX = os.getenv('STELLANTIS_VECTOR_INDEX')
ES_VECTOR_INDEX = "stellantis-minilm-embeddings-hosted"
TRANSFORMER_MODEL_ID = os.getenv('MS_MARCO_TRANSFORMER_MODEL_ID')

# set vertex ai init params
PROJECT_ID = os.getenv('PROJECT_ID')
REGION = os.getenv('REGION')
VERTEX_AI_MODEL_ID = os.getenv('VERTEX_AI_MODEL')
vertexai.init(project=PROJECT_ID, location=REGION)

# Init bedrock client
session = boto3.Session(profile_name="default")
bedrock_client = session.client(service_name="bedrock-runtime")


# connect to es
es = Elasticsearch(
    cloud_id=CLOUD_ID,
    basic_auth=(CLOUD_USERNAME, CLOUD_PASSWORD)
)
es.info()

# set OpenAI API key
#OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
#embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vector_store = ElasticsearchStore(
    es_cloud_id=CLOUD_ID,
    es_user=CLOUD_USERNAME,
    es_password=CLOUD_PASSWORD,
    index_name=ES_VECTOR_INDEX,
    query_field='text_field',
    vector_query_field="vector_query_field.predicted_value",
    #embedding=embeddings,
    distance_strategy="DOT_PRODUCT",
    strategy=ElasticsearchStore.ApproxRetrievalStrategy(query_model_id=TRANSFORMER_MODEL_ID,
                                                        hybrid=True)
)

# Init retriever
retriever = vector_store.as_retriever()

# Build prompt for stuff chain
stuff_template= """
    Utilise uniquement les informations suivantes de contexte pour répondre à la question.  
    n'essaie pas d'inventer une réponse. Utilise au maximum 10 phrases. 
    Sois clair et détaillé dans tes réponses. 
    Termine tes réponses par "J'espere avoir répondu à votre question." sauf lorsque tu ne connais pas la réponse. 
{context}
Question: {question}
Helpful Answer:"""
stuff_prompt = PromptTemplate.from_template(stuff_template)

# Prompt for refine chain
template = """
    Utilise uniquement les informations suivantes de contexte pour répondre à la question. 
    n'essaie pas d'inventer une réponse. Utilise au maximum 10 phrases. 
    Sois clair et détaillé dans tes réponses. 
    Termine tes réponses par "J'espere avoir répondu à votre question." sauf lorsque tu ne connais pas la réponse. 
{context_str}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

refine_template = ("The original question is as follows: {question}\n"
                   "We have provided an existing answer: {existing_answer}\n"
                   "We have the opportunity to refine the existing answer(only if needed) with some more context below.\n"
                   "------------\n{context_str}\n------------\n"
                   "Given the new context, refine the original answer to better answer the question. "
                   "If the context isn't useful, always return the original answer."
                    "If you don't know the answer, say 'I don't know the answer to this question.'\n"
                   )

refine_prompt = PromptTemplate.from_template(refine_template)

def init_qa_chain(llm_model, llm_temperature, chain_type):
    llm = None
    if st.session_state.llm_model == 'gpt-3.5-turbo-16k' or 'gpt4':
        # set OpenAI API key
        llm = ChatOpenAI(
            model_name=st.session_state.llm_model,
            temperature=st.session_state.llm_temperature,
            model_kwargs={"engine": st.session_state.llm_model},
            openai_api_key=os.getenv('OPENAI_API_KEY'))

    if st.session_state.llm_model == 'azure-openai':
        os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('AZURE_OPENAI_API_KEY')
        os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv('AZURE_OPENAI_ENDPOINT')
        llm = AzureChatOpenAI(
            temperature=st.session_state.llm_temperature,
            openai_api_version="2023-07-01-preview",
            azure_deployment="yazid-gpt4",
            #azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        )
    if st.session_state.llm_model == 'vertex-ai':
        llm = ChatVertexAI(
            model_name="chat-bison",
            max_output_tokens=256,
            temperature=st.session_state.llm_temperature,
            verbose=True,
        )

    # aws bedrock
    if st.session_state.llm_model == 'bedrock':
        default_model_id = "anthropic.claude-v2"
        AWS_MODEL_ID = default_model_id
        llm = Bedrock(
            client=bedrock_client,
            model_id=AWS_MODEL_ID
        )

    if st.session_state.llm_model == 'Ollama':
        llm = Ollama(
            model="mistral"
        )
    if chain_type == "stuff":
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": stuff_prompt},
            verbose=True
        )
    if chain_type == "refine":
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="refine",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "question_prompt": QA_CHAIN_PROMPT,
                "refine_prompt": refine_prompt},
            verbose=True
        )

def main():
    if question := st.text_input("Posez votre question"):
        # start APM Transaction
        apm_client.begin_transaction("request")
        try:
            elasticapm.label(query=question, llm=st.session_state.llm_model)
            qa = init_qa_chain(llm_model=st.session_state.llm_model, llm_temperature=st.session_state.llm_temperature, chain_type="stuff")

            with st.chat_message("assistant"):
                st.spinner("Recherche de la réponse...")
                message_placeholder = st.empty()
                full_response = ""
                #stream_handler = StreamHandler(st.empty())
                with elasticapm.capture_span("qa-"+st.session_state.llm_model, "qa"):
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
            elasticapm.set_transaction_outcome("success")
            apm_client.end_transaction("user-query-"+ st.session_state.llm_model)
        except Exception as e:
            apm_client.capture_exception()
            elasticapm.set_transaction_outcome("failure")
            apm_client.end_transaction("user-query-"+ st.session_state.llm_model)
            print(e)
            st.error(e)

if __name__ == '__main__':
    main()

