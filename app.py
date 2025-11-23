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
        st.error("⚠️ Missing Gemini Key in Secrets.")
        st.stop()
    if "TAVILY_API_KEY" in st.secrets:
        os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
except FileNotFoundError:
    st.error("🚨 Secrets file not found!")
    st.stop()

# --- 2. SIDEBAR & FILE HANDLING ---
with st.sidebar:
    st.header("📊 Analytica Workspace")
    
    # A. File Uploader
    uploaded_file = st.file_uploader("Upload Data", type=["csv"])
    if uploaded_file:
        file_path = "dataset.csv"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Dataset Loaded")
    
    st.divider()
    
    # B. THE DOWNLOAD ZONE (NEW!) 📥
    st.subheader("📂 Agent Deliverables")
    st.caption("Files created by the agent will appear here.")
    
    # Check the folder for any new files
    for filename in os.listdir("."):
        if filename.endswith((".csv", ".png", ".txt", ".pdf")) and filename not in ["dataset.csv", "requirements.txt", "app.py", "launcher.bat"]:
            with open(filename, "rb") as f:
                st.download_button(
                    label=f"⬇️ Download {filename}",
                    data=f,
                    file_name=filename,
                    mime="application/octet-stream"
                )

# --- 3. AGENT LOGIC ---
@st.cache_resource
def get_agent():
    repl = PythonREPL()
    
    @tool
    def python_analyst_tool(code: str):
        """Executes Python code. 
        Use this to analyze data, save CSVs, or save plots as 'plot.png'."""
        try:
            code = code.strip("`").replace("python", "")
            result = repl.run(code)
            return f"Executed:\n{result}"
        except Exception as e:
            return f"Error: {e}"

    tools = [python_analyst_tool]
    if "TAVILY_API_KEY" in os.environ:
        tools.append(TavilySearchResults(max_results=3))

    # Using Gemini 2.0 Flash
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

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

st.title("📊 Analytica: Enterprise Intelligence")

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"): st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"): 
            st.write(msg.content)
            
            # Display Plot if it exists
            if os.path.exists("plot.png"):
                 try:
                     if os.path.getmtime("plot.png") > st.session_state.get('last_plot_time', 0):
                        st.image("plot.png")
                        st.session_state['last_plot_time'] = os.path.getmtime("plot.png")
                 except OSError:
                     pass

user_input = st.chat_input("Ask the agent...")

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"): st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing & Saving Files..."):
            prompt = user_input
            if uploaded_file: prompt += " (Data is in 'dataset.csv')"
            inputs = {"messages": st.session_state.messages}
            
            # High recursion limit for complex tasks
            final_state = app.invoke(inputs, config={"recursion_limit": 100})
            
            st.write(final_state["messages"][-1].content)
            
            if os.path.exists("plot.png"): st.image("plot.png")
            st.session_state.messages = final_state["messages"]
            
            # Force Sidebar Refresh to show new download buttons
            st.rerun()
