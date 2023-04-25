import langchain
import os
import pdfminer3 as pdfminer
from langchain.llms import OpenAI
from langchain import HuggingFaceHub
from langchain import PromptTemplate
from langchain import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

import logging
# huggingface langchain token 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
OPENAI_API_KEY="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"

#openai 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

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

llm = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":64})


#llm("translate English to German: How old are you?")
question = """Question: Tell me why Sir Isaac Newton did not speak to Leibnitz"""
template = """Question: {question}

Let's think step by step.

Answer: """


prompt = PromptTemplate(template=template, input_variables=["question"])
prompt.format(question=question)

llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.run(question))
print("------")
#---- loader this is from abonia blog
# Load documents from the specified directory using a DirectoryLoader object
FILE_DIR="C:\\Users\\ravi\\Amazon Drive\\Amazon Drive\\Books"
loader = DirectoryLoader(FILE_DIR, glob='*.pdf')
documents = loader.load()

# split the text to chuncks of of size 1000
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# Split the documents into chunks of size 1000 using a CharacterTextSplitter object
texts = text_splitter.split_documents(documents)

# Create a vector store from the chunks using an OpenAIEmbeddings object and a Chroma object
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
docsearch = Chroma.from_documents(texts, embeddings)


"""" seems to do asomething 
from langchain import OpenAI, ConversationChain

#llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)
conversation.predict(input="Can we talk about AI?")
"""