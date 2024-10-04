from langchain_community.document_loaders import TextLoader, JSONLoader, PyPDFLoader, CSVLoader, UnstructuredExcelLoader, Docx2txtLoader
import os
import shutil
import time
import streamlit as st

def documentation(filename):
    if filename.endswith('.txt') or filename.endswith('.md') or filename.endswith('dat'):
        loader = TextLoader(filename, encoding="utf-8")
    elif filename.endswith('.json'):
        loader = JSONLoader(filename)
    elif filename.endswith('.pdf'):
        loader = PyPDFLoader(filename)
    elif filename.endswith('.csv'):
        loader = CSVLoader(filename)
    elif filename.endswith('.xlsx') or filename.endswith('.xls'):
        loader = UnstructuredExcelLoader(filename)
    elif filename.endswith('.docx'):
        loader = Docx2txtLoader(filename)
    else:
        print("Unsupported file format")
        return
    try:
        data = loader.load()
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")

def delete_file_if_exists(folder_path):
    try:
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            print(f"Deleted folder {folder_path}")
        else:
            print(f"Folder {folder_path} does not exist")
    except Exception as e:
        print(f"Error deleting folder {folder_path}: {e}")

def get_folder_names(directory):
    try:
        folder_names = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
        return folder_names
    except:
        st.warning("NO Database is found for RAG, pl")