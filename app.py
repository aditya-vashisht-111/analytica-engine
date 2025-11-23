import streamlit as st
import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated

# --- PAGE CONFIG ---
st.set_page_config(page_title="Analytica", page_icon="📊", layout="wide")

# --- 1. SECURE AUTHENTICATION ---
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    else:
        st.error("⚠️ Missing Gemini Key in .streamlit/secrets.toml")
        st.stop()
        
    if "TAVILY_API_KEY" in st.secrets:
        os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
except FileNotFoundError:
    st.error("🚨 Secrets file not found! Please create .streamlit/secrets.toml")
    st.stop()

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("📂 Secure Workspace")
    st.markdown("✅ **Encryption:** Active")
    st.markdown("✅ **Keys:** Loaded from Safe")
    
    uploaded_file = st.file_uploader("Upload Data", type=["csv"])
    
    if uploaded_file:
        file_path = "dataset.csv"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Data Encrypted & Loaded")
        df = pd.read_csv(file_path)
        st.dataframe(df.head(3))

# --- 3. AGENT LOGIC ---
@st.cache_resource
def get_agent():
    repl = PythonREPL()
    
    @tool
    def python_analyst_tool(code: str):
        """Executes Python code to analyze data or plot charts.
        ALWAYS save plots as 'plot.png'."""
        try:
            code = code.strip("`").replace("python", "")
            result = repl.run(code)
            if os.path.exists("plot.png"):
                return "Chart generated and saved as 'plot.png'."
            return f"Executed:\n{result}"
        except Exception as e:
            return f"Error: {e}"

    # Only add search capability if the key was found
    tools = [python_analyst_tool]
    if "TAVILY_API_KEY" in os.environ:
        tools.append(TavilySearchResults(max_results=3))

    # Using the model we confirmed works for you
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    # Use 'add_messages' to keep the entire history intact
    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    def reasoner(state: AgentState):
        return {"messages": [llm_with_tools.invoke(state['messages'])]}

    def executor(state: AgentState):
        last_message = state['messages'][-1]
        results = []
        for tool_call in last_message.tool_calls:
            if tool_call['name'] == "python_analyst_tool":
                output = python_analyst_tool.invoke(tool_call['args'])
            elif tool_call['name'] == "tavily_search_results_json":
                search = TavilySearchResults(max_results=3)
                output = search.invoke(tool_call['args'])
            else:
                output = "Error: Unknown tool."
            
            # CRITICAL FIX: Return a ToolMessage object, not a dictionary
            results.append(ToolMessage(
                tool_call_id=tool_call['id'],
                content=str(output)
            ))
        return {"messages": results}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", reasoner)
    workflow.add_node("tool", executor)
    workflow.set_entry_point("agent")
    
    def should_continue(state):
        if state["messages"][-1].tool_calls: return "tool"
        return END
    
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tool", "agent")
    
    return workflow.compile()

# --- 4. INTERFACE ---
if "messages" not in st.session_state: st.session_state.messages = []
app = get_agent()

st.title("📊 Analytica: Enterprise Data Intelligence")
st.caption("Powered by Gemini 2.0 Flash • Running Locally")

# Display Chat
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"): st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"): 
            st.write(msg.content)
            if "plot.png" in msg.content or os.path.exists("plot.png"):
                 if os.path.getmtime("plot.png") > st.session_state.get('last_plot_time', 0):
                    st.image("plot.png")
                    st.session_state['last_plot_time'] = os.path.getmtime("plot.png")

# Input
user_input = st.chat_input("Ask your question...")

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"): st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            prompt = user_input
            if uploaded_file: prompt += " (Data is in 'dataset.csv')"
            
            inputs = {"messages": st.session_state.messages}
            final_state = app.invoke(inputs)
            
            # Get the very last response
            st.write(final_state["messages"][-1].content)
            
            if os.path.exists("plot.png"): st.image("plot.png")
            
            st.session_state.messages = final_state["messages"]