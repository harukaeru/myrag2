import dotenv
import os
dotenv.load_dotenv('./.env')

print(os.getenv('REDDIT_NAME'))
print(os.getenv("LANGCHAIN_TRACING_V2"))
print(os.getenv("LANGCHAIN_ENDPOINT"))
print(os.getenv("LANGCHAIN_API_KEY"))
print(os.getenv("LANGCHAIN_PROJECT"))
