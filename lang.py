import operator
from typing import Annotated

from langchain_core.pydantic_v1 import BaseModel, Field

class State(BaseModel):
  query: str = Field(description="ユーザーからの質問")

  current_role: str = Field(defualt="", description="選定された回答ロール")

  messages: Annotated[list[dict], operator.add] = Field(default_factory=list, description="回答履歴")

  current_judge: bool = Field(default=False, description="回答の品質評価")

  judgement_reason: str = Field(default="", description="回答の品質評価の判定理由")

  from langgraph.graph import StateGraph

  workflow = StateGraph(initial_state=State())