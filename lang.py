import dotenv
import operator
from typing import Annotated, Any

from langchain_core.pydantic_v1 import BaseModel, Field

dotenv.load_dotenv('./.env')

ROLES = {
  '1': {
    'name': '一般知識エキスパート',
    'description': '幅広い分野の一般的な質問に答える',
    'details': '幅広い分野の一般的な質問に対して、正確でわかりやすい回答を提供してください。'
  },
  '2': {
    'name': '生成AI製品エキスパート',
    'description': 'LangChainやClaudeなどの生成AI製品に関する質問に答える',
    'details': 'LangChainやClaudeなどの生成AI製品に関する質問に対して、最新の情報とや深い洞察を提供してください。'
  },
  '3': {
    'name': 'カウンセラー',
    'description': '個人的な悩みや心理的な問題に対してサポートを提供する',
    'details': '個人的な悩みや心理的な問題に対して、共感的で支援的な回答を提供し、可能ならば適切なアドバイスを提供してください。'
  }
}

class State(BaseModel):
  query: str = Field(description="ユーザーからの質問")

  current_role: str = Field(defualt="", description="選定された回答ロール")

  messages: Annotated[list[dict], operator.add] = Field(default_factory=list, description="回答履歴")

  current_judge: bool = Field(default=False, description="回答の品質評価")

  judgement_reason: str = Field(default="", description="回答の品質評価の判定理由")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm = llm.configurable_fields(max_tokens=1000)

def selection_role(state: State) -> dict[str, Any]:
  query = state.query

  role_options = '\n'.join([f'{key}. {ROLES[key]["name"]}: {ROLES[key]["description"]}' for key in ROLES.keys()])

  prompt = ChatPromptTemplate.from_template('''
  質問を分析し、最も適切な
                                            )

  return {"current_role": "1"}

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

workflow = StateGraph(initial_state=State())

def answering_node(state: State) -> dict[str, Any]:
  query = state.query
  role = state.current_role

  generated_message = ''

  return {"messages": [generated_message]}