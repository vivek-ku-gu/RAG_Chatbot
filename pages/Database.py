import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from Utils import documentation, delete_file_if_exists, get_folder_names
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
import time

documents = []

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# def process_in_batches(documents, batch_size, delay_time, dbname, embeddings):
#     for i in range(0, len(documents), batch_size):
#         batch = documents[i:i + batch_size]
#         ids = [str(j) for j in range(i, i+batch_size)]
#         try:
#             Chroma.from_documents(batch, embeddings, ids=ids, persist_directory=dbname)
#             #FAISS.from_documents(batch, embeddings)
#         except Exception as error:
#             print(f"Handling request: {error}")
#         print(i)
#         exit()

def create_db(documents, chunking_size, chunking_overlap, database_name):
    st.write(documents)
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunking_size, chunk_overlap=chunking_overlap)
    all_splits = text_splitter.split_documents(documents)

    directory = os.path.join("Databases", database_name)
    # batch_size = 100
    # delay_time = 2
    #
    # for i in range(0, len(documents), batch_size):
    #     batch = documents[i:i + batch_size]
    #     ids = [str(j) for j in range(i, i + batch_size)]
    #
    #     try:
    #         vectordb = Chroma.from_documents(documents=filter_complex_metadata(batch), ids=ids,
    #                                          embedding=embeddings, persist_directory=directory)
    #         # FAISS.from_documents(batch, embeddings)
    #     except Exception as error:
    #         print(f"Handling request: {error}")
    #         exit()
    #
    #     print(i)
    #     time.sleep(delay_time)
    vectordb = Chroma.from_documents(documents=filter_complex_metadata(all_splits), embedding=embeddings,
                                     persist_directory=directory)
    return vectordb

st.title("You can create and delete database from here")
database_name = ""
chunking_size = int(st.text_input(label='Chunking Size', value=5000))
chunking_overlap = int(st.text_input(label='Chunking Overlap', value=500))
database_name = st.text_input("Enter name for the database you want to create")

path = st.file_uploader("Choose files", accept_multiple_files=True)
print("printing in path")

print(path)
len_of_files = len(path)

for uploaded_file in path:
    st.write(uploaded_file)
    if uploaded_file:
        if not os.path.exists("Files"):
            os.mkdir("Files")
        filePath = os.path.join("Files", uploaded_file.name)
        with open(filePath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)

        st.write(len_of_files)
        docs = documentation(filePath)
        documents.extend(docs)

parent_directory= "Databases"
directories = [entry.name for entry in os.scandir(parent_directory) if entry.is_dir()]

sub_for_db = st.button(label = 'Create DB', type='primary')


clear_db = st.selectbox("Please Select the database name you want to delete", directories, index=None, placeholder="Select database here...")
clear_for_db = st.button(label='Clear DB', type='secondary')
if clear_for_db:
    delete_dir = os.path.join("Databases", clear_db)
    delete_file_if_exists(delete_dir)



if sub_for_db:
    if len(database_name) > 3:
        create_db(documents, chunking_size, chunking_overlap, database_name)
    else:
        st.warning("Name of the database must be of atleast 3 characters")
