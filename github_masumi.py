import os

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator

from langchain.document_loaders import GitLoader

clone_url = "https://github.com/hwchase17/langchain"
branch = "master"
repo_path = "./temp/"
filter_ext = ".py"

if os.path.exists(repo_path):
    clone_url = None

loader = GitLoader(
    clone_url=clone_url,
    branch=branch,
    repo_path=repo_path,
    file_filter=lambda file_path: file_path.endswith(filter_ext),
)

index = VectorstoreIndexCreator(
    vectorstore_cls=Chroma, # Default
    embedding=OpenAIEmbeddings(disallowed_special=()), # Default
).from_loaders([loader])

query = "LangChainには、どんな種類のDocument Loadersが用意されていますか？"

answer = index.query(query)
print(answer)
