import re
import json
import boto3
import warnings
import random
import pandas as pd
from sklearn.metrics import accuracy_score
from IPython.display import Markdown
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import (RecursiveCharacterTextSplitter, CharacterTextSplitter,)
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Use Bedrock for inference to test everything works as expected
def load_bedrock_runtime():
    # Define the bedrock-runtime client that will be used for inference
    bedrock_runtime = boto3.client(service_name="bedrock-runtime")
    # Define the model name and parameters for inference
    bedrock_model_id = "anthropic.claude-instant-v1"
    # each model has a different set of inference parameters
    inference_modifier = {
        "max_tokens_to_sample": 512, 
        "temperature": 0.0,
        "stop_sequences": [
        "\\n\\nHuman:"
        ]
    }
    # Load the Bedrock langchain module with the selected bedrock model
    llm = Bedrock(
        model_id=bedrock_model_id, client=bedrock_runtime, model_kwargs=inference_modifier
    )
    # Quick test
    #reply = str(llm("\n\nHuman: Are you ready to answer some questions? Just say 'yes sir' if you are ready.\n\nAssistant:"))
    #print(reply)
    print("+++++++++++++++++++++++++++++++++")
    return(llm)

# provide a prompt response without RAG
def prompting_without_retriever(llm, question):
    template = """

    Human: Answer the question below.
    Keep your response as precise as possible and limit it to a few words. 
    If you don't know the answer, respond "I don't know".

    Here is the question: 
    {question}

    Assistant:"""

    prompt_message = PromptTemplate.from_template(template).format(
        question=question
    )
    answer = llm(prompt_message)
    return answer.strip()

# document loader
def rag_document_loader(urls):
    # Define the URL Loader
    loader = UnstructuredURLLoader(urls=urls)
    # Load the data
    data = loader.load()
    # Pre-process the content for prettier display
    data[0].page_content = re.sub("\n{3,}", "\n", data[0].page_content)
    data[0].page_content = re.sub(" {2,}", " ", data[0].page_content)
    #print(data[0].page_content[231:2100])
    #print()
    return(data)

# Split documents into chunks
def split_documents(data):
    # Use the recursive character splitter
    recur_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        is_separator_regex=True,
    )

    # Perform the splits using the splitter
    data_splits = recur_splitter.split_documents(data)

    # Print a random chunk
    #print(random.choice(data_splits).page_content)
    return(data_splits)

# Embeddings and vector databases
def embeddings_vector(data_splits):
    bedrock_runtime = boto3.client(service_name="bedrock-runtime")
    bedrock_embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1", client=bedrock_runtime
    )

    # Create a vector DB from documents retrieved from the URL and split with the RecursiveCharacterTextSplitter
    vectorstore_faiss = FAISS.from_documents(
        data_splits,
        bedrock_embeddings,
    )
    
    return(vectorstore_faiss)

# Prompting with RAG
def prompting_with_retriever(llm, vectorstore_faiss, question):
    # Supress warnings
    warnings.filterwarnings("ignore")

    context_template = """

    Human: Answer the question below.
    Use the given context to answer the question. 
    If you don't know the answer, respond "I don't know".
    Keep your response as precise as possible and limit it to a few words. 

    Here is the context:
    {context}

    Here is the question: 
    {question}

    Assistant:"""

    # Define the prompt template for Q&A
    #context_prompt_template = PromptTemplate.from_template(context_template).format(question=question)
    context_prompt_template = PromptTemplate(template = context_template, input_variables=["context", "question"])

    # Define the RetrievalQ&A chain
    # We pass the llm and the FAISS vector store, retrieving the k most relevant documents
    rag_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": context_prompt_template},
    )

    answer = rag_chain({"query": question})["result"].strip()

    return(answer)

def lambda_handler(event, context):
    print("Lambda query function starting.")

    # Load the bedrock runtime
    llm_instance = load_bedrock_runtime()

    # Query without RAG
    question = "How many people attended re:Invent 2022 in person?"
    answer = prompting_without_retriever(llm_instance, question)
    print(answer)

    # Load the document data from URLs
    urls = ["https://aws.amazon.com/blogs/security/three-key-security-themes-from-aws-reinvent-2022/"]
    data = rag_document_loader(urls)

    # Split the data
    data_splits = split_documents(data)

    # Embeddings and vector databases
    vectorstore_faiss = embeddings_vector(data_splits)

    # Prompting with RAG
    #question = "How many years has AWS re:Invent been running?"
    question = "Who is the AWS CEO?"
    answer = prompting_with_retriever(llm_instance, vectorstore_faiss, question)
    print(answer)

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
