from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
import dotenv
import os
import chromadb

dotenv.load_dotenv('./.env')

def file_filter(file_path) -> bool:
    return file_path.endswith('.mdx')

# clone_url = 'https://github.com/langchain-ai/langchain'

# if os.path.exists('langchain'):
#   clone_url = None
# 
# loader = GitLoader(
#     clone_url=clone_url,
#     repo_path='./langchain',
#     branch='master',
#     file_filter=file_filter
# )
# 
# documents = loader.load()
# print(len(documents))

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
client = chromadb.PersistentClient(path='./chroma_db')
db = Chroma(
  collection_name='langchain',
  embedding_function=embeddings,
  client=client
)

retriever = db.as_retriever()

prompt = ChatPromptTemplate.from_template('''\
以下の文脈を踏まえて質問に回答してください。
文脈:"""
{context}
"""
質問:
{question}
''')

llm = ChatOpenAI(model='gpt-4o-mini')

chain = (
  {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
)

out = chain.invoke('langchainの概要を教えてください')
print(out)