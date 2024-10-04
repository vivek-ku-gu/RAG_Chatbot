import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
import json
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from dotenv import load_dotenv
from langchain_chroma import Chroma
from Utils import get_folder_names
import datetime
load_dotenv()
st.set_page_config(
    page_title="RAG with GPT",
    layout="wide")

st.title("RAG with GPT")

dbexist = False
history = []
filename_for_json = ""
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0,max_tokens=None,timeout=None)

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)
# model = ChatGoogleGenerativeAI(model="gemini-1.0-pro",convert_system_message_to_human=True)
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question "
    "If you don't know the answer, say that you don't know."
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

retriever_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history,"
    "formulate a standalone question which can be understood without the chat history."
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", retriever_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),

    ]
)


chat_history = []
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)




temp = None
filename = None
data = None
sub_for_db = None
rag = False
chain = None

def clear_cache():
    keys = list(st.session_state.keys())
    for key in keys:
        st.session_state.pop(key)
def save_chat():
    directory_path = "Chats"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    if filename_for_json:
        Keys = (st.session_state.messages)
        json_data = json.dumps(list(Keys))
        json_name = filename_for_json + ".json"
        file_name = os.path.join(directory_path, json_name)
        if os.path.isfile(file_name):
            st.warning("chat already exist")
        else:
            with open(file_name, 'w') as f:
                f.write(json_data)
    else:
        Keys = (st.session_state.messages)
        json_data = json.dumps(list(Keys))
        x = datetime.datetime.now()
        data_time = f"{x.strftime('%m_%d_%Y_%H_%M_%S')}"
        json_name = data_time + ".json"
        file_name = os.path.join(directory_path, json_name)
        with open(file_name, 'w') as f:
            f.write(json_data)

with st.sidebar:
    rag = st.toggle('RAG')

    if not os.path.exists("Databases"):
        os.makedirs("Databases")
    directories = [entry.name for entry in os.scandir("Databases") if entry.is_dir()]
    database = st.selectbox("Please select the database for RAG", (directories))

    temp = st.slider(label='Temperature', min_value=0.0, max_value=1.0, value=0.5, help="The Temperature parameter controls the diversity of the generated text.")
    top_p = st.slider(label='Top_P', min_value=0.0, max_value=1.0, value=0.45, help="The Top p parameter controls the diversity of the generated text.")

    filename_for_json = st.text_input("Enter the chat name here")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.button('Save Chats', on_click=save_chat, type='primary')
        with col2:
            st.button('Clear Chats', on_click=clear_cache)
    if not os.path.exists("Chats"):
        os.makedirs("Chats")
    directories_json = [entry.name for entry in os.scandir("Chats") if entry.is_file()]

    chat = st.selectbox("Select Your chats from the below ",
                        (directories_json),
                        index=None,
                        placeholder="Select Chats here...",)
    import_clicked = st.button('Import Chats', type='primary')

    if import_clicked:
        chat_dir = os.path.join("Chats", chat)
        with open(chat_dir) as file:
        # Load the JSON data
            chat_data = json.load(file)
        # chat_data = list(chat)
        for c in chat_data:
            st.session_state.messages.append(c)
# if rag:
#     vectordb = get_folder_names('Databases')

if os.path.exists('Databases'):
        dbexist = True
        print(f"db is existing : {dbexist}")

if dbexist and rag:
        directory_for_db = os.path.join("Databases", database)
        vectordb = Chroma(persist_directory=directory_for_db, embedding_function=embeddings)

def create_chain(vectordb, llm):
    retriever = vectordb.as_retriever()
    # retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"score_threshold": 0.7})

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the provided context.
    Think step by step before providing a detailed answer.
    prompt

    if asked for code you provide the complete code importing all the required libraries framework coding standards
    with help of the context and with proper comments were ever needed.

    <context>
    {context}
    </context>

    Question: {input}""")
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def create_history_chain(vectordb,model):
    retriever = vectordb.as_retriever()
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

def run_my_rag_with_history(chain,query):
    result = chain.invoke({"input": query, "chat_history": chat_history})

    chat_history.extend(
        [
            HumanMessage(content=query),
            AIMessage(content=result["answer"]),
        ]
    )
    return result
def run_my_rag(chain, query):

    result = chain.invoke({"input": query})
    return result
counter = True
fullresponse =""
if dbexist and rag:
        chain = create_chain(vectordb, llm)
        # chain = create_history_chain(vectordb, model)
        if "messages" not in st.session_state.keys():
                st.session_state.messages = [
                    {"role": "assistant", "content": "Welcome, You can put your questions below"}]

            # Display chat messages
        for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            # User-provided prompt
        if (len(st.session_state.messages) > 5):
            st.session_state.messages.pop(0)
        # chat_his = ""
        #
        # for message in enumerate(st.session_state.messages):
        #     if(message[0] == 0):
        #         continue
        #     chat_his += (message[1]["content"])
        # print(chat_his)

        input = st.chat_input()



        if input:
            st.session_state.messages.append({"role": "user", "content": input})
            with st.chat_message("user"):
                st.write(input)

            # Generate a new response if last message is not from assistant
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Please wait, Getting your answer from RAG"):
                    try:

                        ans = run_my_rag(chain,input)

                    except Exception as e:
                        st.warning("token limit have been crossed"+ e)
                    response = ans['answer']
                    st.write(response)

            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
else:
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Welcome, You can put your questions below "}]
        print(type(st.session_state.messages))

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    if (len(st.session_state.messages) > 5):
        st.session_state.messages.pop(0)
    input = st.chat_input()

    if input:
        st.session_state.messages.append({"role": "user", "content": input})
        with st.chat_message("user"):
            st.write(input)
    response = ""

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Please wait, Getting your answer"):
                chat_session = model.start_chat(
                    history= []
                )
                completion = chat_session.send_message(input)
                response = completion.text
                # st.session_state.messages.append(message)
                st.write(response)

        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)