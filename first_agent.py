import streamlit as st
from dotenv import load_dotenv
import os
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.github import GithubTools


load_dotenv()
GITHUB_TOKEN=os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    st.error("GITHUB_TOKEN environment variable not set.")
    st.stop()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY environment variable not set.")
    st.stop()


model = Groq(id="llama3-70b-8192", api_key=api_key)
github_tools = GithubTools()


agent = Agent(
    model=model,
    instructions=[
        "Use your tools to answer questions about the repo: garvkhurana/mlops_.",
        "Do not create any issues or pull requests unless explicitly asked to do so.",
        "Use markdown to format your answers.",
    ],
    tools=[github_tools],
    show_tool_calls=True,
    markdown=True,
    verbose=True,
)


st.set_page_config(page_title="GitHub MLOps Chatbot", page_icon="🤖")
st.title(" MLOps GitHub Chatbot")
st.markdown("Ask anything about the GitHub repo: **`garvkhurana/mlops_`**")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Ask a question...")
if user_input:
    
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            response = agent.run(user_input)
            if hasattr(response, "content"):
                reply = response.content
            else:
                reply = str(response)  
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
