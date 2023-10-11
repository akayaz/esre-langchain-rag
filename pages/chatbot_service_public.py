import os

import streamlit as st
import vertexai
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatVertexAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import VertexAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.vectorstores import ElasticsearchStore
from utils import StreamHandler
from google.cloud import aiplatform

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# set elastic cloud id and password
CLOUD_ID = os.getenv('CLOUD_ID')
CLOUD_USERNAME = os.getenv('CLOUD_USERNAME')
CLOUD_PASSWORD = os.getenv('CLOUD_PASSWORD')
ES_VECTOR_INDEX = os.getenv('HOSTED_VECTOR_INDEX')

# set vertex ai init params
PROJECT_ID = os.getenv('PROJECT_ID')
REGION = os.getenv('REGION')
VERTEX_AI_MODEL_ID = os.getenv('VERTEX_AI_MODEL')
#ES_VECTOR_INDEX = os.getenv('ES_VECTOR_INDEX')



st.set_page_config(page_title="Chatbot avec contexte", page_icon="⭐")
st.image(
    'https://images.contentstack.io/v3/assets/bltefdd0b53724fa2ce/blt601c406b0b5af740/620577381692951393fdf8d6/elastic-logo-cluster.svg',
    width=50)
st.title("Service-Public.fr Chatbot")
st.write('Assistant Virtuel sur les actualités de Service-Public.fr')

template = """
                    Tu es un assistant virtuel qui aide les citoyens à trouver des informations sur le site service-public.fr.
                    Utilise uniquement les informations suivantes de contexte pour répondre à la question. 
                    Si tu ne connais pas la réponse, dis juste 'Désolé, Je ne dispose malheureusement pas des informations nécessaires pour répondre à cette question, n'essaie pas d'inventer une réponse. 
                    Sois clair, concis et précis dans tes réponses.
                    Si la question ne concerne pas Service-Public.fr, reponds poliment que tu es un assistant dédié au secteur public

                    {context}
                    Question: {question}
                    Helpful Answer:"""


def init_chat_model(selected_model, llm_temp):
    try:
        if selected_model == 'gpt-3.5-turbo-16k' or 'gpt4':
            return ChatOpenAI(
                model_name=selected_model,
                temperature=llm_temp,
                openai_api_key=OPENAI_API_KEY,
                streaming=True
            )
        if selected_model == 'vertex-ai':
            vertexai.init(project=PROJECT_ID, location=REGION)
            llm = VertexAI(
                model_name="text-bison@001",
                max_output_tokens=256,
                temperature=llm_temp,
                top_p=0.8,
                top_k=40,
                verbose=True,
            )
            return llm
    except Exception as e:
        st.write("Oops! Une erreur est survenue. Veuillez réessayer.")
        print(e.__cause__)
        return


class ContextChatbot:
    def __init__(self):
        
        self.es = Elasticsearch(
            cloud_id=CLOUD_ID,
            basic_auth=(CLOUD_USERNAME, CLOUD_PASSWORD)
        )
        self.es.info()
        self.vector_store = ElasticsearchStore(
            es_cloud_id=CLOUD_ID,
            es_user=CLOUD_USERNAME,
            es_password=CLOUD_PASSWORD,
            index_name=ES_VECTOR_INDEX,
            # embedding=self.embeddings,
            query_field='text_field',
            vector_query_field='vector_query_field.predicted_value',
            distance_strategy="DOT_PRODUCT",
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=True,
                                                                query_model_id="sentence-transformers__all-minilm-l6-v2",)
        )
        self.openai_api_key = OPENAI_API_KEY
        # self.retriever = self.vector_store.as_retriever()
        # test with similarity score threshold
        self.retriever = self.vector_store.as_retriever(type='similarity_search_threshold', similarity_threshold=0.8)

    def setup_llm_chain(self, prompt_template, selected_model, llm_temperature):
        msgs = StreamlitChatMessageHistory()
        memory = ConversationBufferMemory(memory_key="chat_history",
                                          chat_memory=msgs,
                                          return_messages=True,
                                          max_history=20,
                                          output_key="answer")

        # OpenAI LLM
        chat = init_chat_model(selected_model, llm_temperature)

        qa_chain_prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

        chain = ConversationalRetrievalChain.from_llm(
            llm=chat,
            memory=memory,
            verbose=True,
            retriever=self.retriever,
            combine_docs_chain_kwargs={"prompt": qa_chain_prompt},
            return_source_documents=True,
        )
        return chain

    def main(self):
        chain = self.setup_llm_chain(template, st.session_state.llm_model, st.session_state.llm_temperature)
        msgs = StreamlitChatMessageHistory()
        documents = self.vector_store
        if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
            msgs.clear()
            msgs.add_ai_message("Bonjour! En quoi puis-je vous aider aujourd'hui? :slightly_smiling_face:")

        avatars = {"human": "user", "ai": "assistant"}
        for msg in msgs.messages:
            st.chat_message(avatars[msg.type]).write(msg.content)

        if user_query := st.chat_input(placeholder="Que souhaitez vous savoir?"):
            st.chat_message("user").write(user_query)
            print("model selected: ", chain)
            with st.chat_message("assistant"):
                stream_handler = StreamHandler(st.empty())
                try:
                    response = chain({"question": user_query, "chat_history": msgs}, callbacks=[stream_handler])

                    st.markdown(""" ##### Sources: """)
                    for docs_source in response['source_documents']:
                        link = f'<a href="{docs_source.metadata["url"]}" target="_blank">{docs_source.metadata["title"]}</a>'
                        st.markdown(link, unsafe_allow_html=True)
                    st.markdown(""" ###### LLM: """ + st.session_state.llm_model)
                except Exception as e:
                    st.write("Oops! Une erreur est survenue. Veuillez réessayer.")
                    print(e)
                    return



if __name__ == "__main__":
    obj = ContextChatbot()
    obj.main()
