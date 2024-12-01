import dotenv
import operator
from typing import Annotated, Any

from langchain_core.pydantic_v1 import BaseModel, Field

dotenv.load_dotenv('./.env')

from langgraph.graph import StateGraph
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

# LLMの設定
llm = OpenAI(model="text-davinci-003", temperature=0.7)

# 各ノードの定義
start_prompt = PromptTemplate(input_variables=[], template="こんにちは！どのようなサポートが必要ですか？")
chatbot_prompt = PromptTemplate(input_variables=["user_input"], template="あなたの入力: {user_input} を処理中です。どうぞお待ちください。")
end_prompt = PromptTemplate(input_variables=[], template="ありがとうございました！またのご利用をお待ちしています。")

# チェーンの作成
start_chain = LLMChain(prompt=start_prompt, llm=llm)
chatbot_chain = LLMChain(prompt=chatbot_prompt, llm=llm)
end_chain = LLMChain(prompt=end_prompt, llm=llm)

# グラフの構築
graph = StateGraph()

# ノードを追加
graph.add_node("start", start_chain)
graph.add_node("chatbot", chatbot_chain)
graph.add_node("end", end_chain)

# ノード間の遷移を設定
graph.add_edge("start", "chatbot")
graph.add_edge("chatbot", "end")

# グラフの実行
def run_langgraph():
    print("LangGraphフローを開始します。\n")

    # 初期ノード
    current_node = "start"

    while current_node:
        # 現在のノードの実行
        print(f"現在のノード: {current_node}")
        node = graph.get_node(current_node)

        # ユーザーの入力を取得
        if current_node == "start":
            result = node.run()
            print(result)
            current_node = "chatbot"
        elif current_node == "chatbot":
            user_input = input("ユーザー入力をどうぞ: ")
            result = node.run({"user_input": user_input})
            print(result)
            current_node = "end"
        elif current_node == "end":
            result = node.run()
            print(result)
            current_node = None

    print("LangGraphフローが終了しました。")


if __name__ == "__main__":
    run_langgraph()
