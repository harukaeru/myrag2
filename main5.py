from typing import List, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import GitLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import CohereRerank
import dotenv
import os

from pydantic import BaseModel, Field
from sympy import per

dotenv.load_dotenv('./.env')

def file_filter(file_path) -> bool:
    return file_path.endswith('.mdx')

# loader = DirectoryLoader('./langchain', glob="**/*.md")
# loader = GitLoader(
#     clone_url='https://github.com/langchain-ai/langchain',
#     repo_path='./langchain',
#     branch='master',
#     file_filter=file_filter
# )
# 
# documents = loader.load()
# print(len(documents))

# def reciprocal_rank_fusion(retriever_outputs: List[List[Document]], k: int = 60) -> List[str]:
#     content_score_mapping = {}
#     for docs in retriever_outputs:
#         for rank, doc in enumerate(docs):
#             content = doc.page_content
# 
#             if content not in content_score_mapping:
#                 content_score_mapping[content] = 0
# 
#             content_score_mapping[content] += 1 / (rank + 1)
#     ranked = sorted(content_score_mapping.items(), key=lambda x: x[1], reverse=True)
#     return [content for content, _ in ranked[:k]]

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
# db = Chroma.from_documents(documents, embeddings, persist_directory='./chroma_db')
db = Chroma(
    persist_directory='./chroma_db',
    embedding_function=embeddings
)

retriever = db.as_retriever()

class QueryGenerationOutput(BaseModel):
    queries: List[str] = Field(description='3つの異なる検索クエリ')

query_generation_prompt = ChatPromptTemplate.from_template('''\
質問に対してベクターデータベースから関連文書を検索するために、
3つの異なる検索クエリを生成してください。
距離ベースの類似性検索の限界を克服するために、
ユーザーの質問に対して複数の視点を提供することが目標です。

質問:
{question}
''')




hyde_prompt = ChatPromptTemplate.from_template('''\
次の質問に回答する一文を書いてください。

質問:
{question}
''')

prompt = ChatPromptTemplate.from_template('''\
以下の文脈を踏まえて質問に回答してください。
文脈:"""
{context}
"""
質問:
{question}
''')

llm = ChatOpenAI(model='gpt-4o-mini')

query_generation_chain = query_generation_prompt | llm.with_structured_output(QueryGenerationOutput) | (lambda x: x.queries)

def rerank(inp: dict[str, Any], top_n: int = 3) -> List[Document]:
    question = inp['question']
    documents = inp['documents']

    cohere_reranker = CohereRerank(model='rerank-multilingual-v3.0', top_n=top_n)
    reranked = cohere_reranker.compress_documents(documents=documents, query=question)
    return reranked
# hyde_chain = hyde_prompt | llm | StrOutputParser()

chain = (
  {"documents":  retriever, "question": RunnablePassthrough()} | RunnablePassthrough.assign(context=rerank) | prompt | llm | StrOutputParser()
)

out = chain.invoke('langchainの概要を教えてください')
print(out)