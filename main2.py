from operator import itemgetter
import dotenv
import os
dotenv.load_dotenv('./.env')
from pprint import pprint

# print(os.getenv("LANGCHAIN_TRACING_V2"))
# print(os.getenv("LANGCHAIN_ENDPOINT"))
# print(os.getenv("LANGCHAIN_API_KEY"))
# print(os.getenv("LANGCHAIN_PROJECT"))
# print(os.getenv("OPENAI_API_KEY"))


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4o-mini')


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template('''\
以下の文脈を踏まえて質問に回答してください。
文脈:"""
{context}
"""
質問:
{question}
''')

from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.runnables import RunnablePassthrough
retriever = TavilySearchAPIRetriever(k=3)

chain = (
  {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
)

out = chain.invoke('東京の今日の天気を詳しくおしえてください')
pprint(out)