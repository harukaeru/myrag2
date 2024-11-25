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
from langchain_core.runnables import RunnableParallel

optimistic_prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは楽観主義者です。ユーザーの入力に対して楽観的な回答をしてください。"),
    ("user", "{topic}"),
])

pessimistic_prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは悲観主義者です。ユーザーの入力に対して悲観的な回答をしてください。"),
    ("user", "{topic}"),
])


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

output_parser = StrOutputParser()

optimistic_chain = optimistic_prompt | llm | output_parser
pessimistic_chain = pessimistic_prompt | llm | output_parser

parallel_chain = RunnableParallel({
  'optimistic_opinion': optimistic_chain, 
  'pessimistic_opinion': pessimistic_chain,
  'topic': itemgetter('topic')
})

synthesize_prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは客観的AIです。{topic}についての2つの意見をまとめてください"),
    ("user", "楽観的意見:{optimistic_opinion}\n悲観的意見:{pessimistic_opinion}"),
])

synthesize_chain = parallel_chain | synthesize_prompt | llm | output_parser

out = synthesize_chain.invoke({"topic": "生成AIの進化について"})
pprint(out)
