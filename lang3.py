import dotenv
import operator
from typing import Annotated, Any

from langchain_core.pydantic_v1 import BaseModel, Field

dotenv.load_dotenv('./.env')

from typing import Dict, TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
import operator

# State definitions
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]

# Channel definitions
class GraphState(TypedDict):
    agent: AgentState

# Initialize LLM
llm = ChatOpenAI(
    temperature=0.7,
    model="gpt-4o-mini"
)

# Define the chat function
def chat_function(state: AgentState) -> AgentState:
    """Basic chat function that processes messages and generates responses."""
    messages = state.get("messages", [''])
    response = llm.invoke(messages)
    return {"messages": messages + [response]}

# Build the graph
def build_graph() -> StateGraph:
    """Constructs the workflow graph for the chatbot."""
    
    # Create new graph
    workflow = StateGraph()

    # Add chat node
    workflow.add_node("chat", chat_function)

    # Create edges
    workflow.set_entry_point("chat")
    
    # Conditional check if we should continue
    def should_continue(state: AgentState) -> bool:
        """Determine if the conversation should continue."""
        # You can add custom logic here
        return False

    # Add chat node
    workflow.add_node("should_continue", should_continue)

    # Add edge back to chat if conversation should continue
    workflow.add_edge("chat", 'should_continue')

    # Compile the graph
    return workflow.compile()

# Create app interface
def create_chatbot():
    """Creates and initializes the chatbot."""
    
    # Build the graph
    graph = build_graph()
    
    def chat(message: str, chat_history: list = None) -> dict:
        """Handle incoming chat messages."""
        if chat_history is None:
            chat_history = []
            
        # Create message history
        messages = chat_history + [HumanMessage(content=message)]
        
        # Initialize state
        state = {"agent": {"messages": messages}}
        
        # Run the graph
        result = graph.invoke(state)

        print('result' ,result)
        
        # Return final state
        return result["agent"]

    return chat

# Example usage
if __name__ == "__main__":
    # Initialize chatbot
    chatbot = create_chatbot()
    
    # Example conversation
    response = chatbot("こんにちは！")
    print(response["messages"][-1].content)  # 最後のメッセージを表示
    
    # 会話を継続
    response = chatbot("今日の天気はどうですか？", response["messages"])
    print(response["messages"][-1].content)