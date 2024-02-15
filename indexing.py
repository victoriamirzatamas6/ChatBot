import os
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

emb_model = HuggingFaceBgeEmbeddings(
    model_name = 'bge-large-en-v1.5',
    model_kwargs = {'device':'cpu'}, #specifying to use the CPU for computations
    encode_kwargs = {'normalize_embeddings': True}, #specifying to compute cosine similarity,
    query_instruction = "Represent this sentence for searching relevant passages:"
)

def load_vector_store():
    chunk_size = 512
    chunk_overlap = 0
    file_name = 'data/UWB_TuningTool_Manual_clean_FINAL.pdf'

    persist_directory=f'cdb-{chunk_size}-{chunk_overlap}_{os.path.splitext(file_name)[0]}'

    if os.path.exists(persist_directory):
        print(f'Vector store found at {persist_directory}!\nLoading from disk...')
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=emb_model )
        print('Loaded vector store from disk!')

    else:

        print(f'Creating and indexing ChromaDB vector database from {file_name}...')
        loader = PyPDFLoader(file_name)
        document = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
                        separators=["\n \n \n \n", "\n \n \n"],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        keep_separator=False)
        
        splits = text_splitter.split_documents(documents=document)

        for i in range(len(splits)):
            splits[i].page_content = splits[i].page_content.strip()

        vector_store = Chroma.from_documents(splits, emb_model, persist_directory=persist_directory)

        txt_path = os.path.join(persist_directory, 'chunks.txt')
        
        print(f'Writing chunks to txt file : {txt_path}')
        with open(txt_path, 'w', encoding="utf-8") as file:
            for i in range(len(splits)):
                file.write(f"\n {i} split  -------------- \n{splits[i].page_content}\n")
        print(f"Succesfully wrote chunks to txt!")


    print('Successfully created vector store!')

    return vector_store