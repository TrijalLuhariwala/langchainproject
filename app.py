import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from urllib.parse import urlparse
from typing import Dict, List, Any
from typing_extensions import TypedDict

# LangChain & LangGraph imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import GithubFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langgraph.graph import StateGraph, END

# llm = None


def build_llm(provider: str, api_key: str):
    provider = provider.lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, google_api_key=api_key)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.7, api_key=api_key)
    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model="mistral-large-latest", temperature=0.7, api_key=api_key)
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model="openai/gpt-oss-120b", temperature=0.1, api_key=api_key)
    raise ValueError("Unsupported provider selection")
# =====================================================================
# 1. DEFINE LANGGRAPH AGENT STATE
# =====================================================================
class AgentState(TypedDict):
    repo_url: str
    repo_full_name: str
    md_content: str
    py_files: List[Dict[str, str]]
    md_summary: str
    md_score: str
    chunk_summaries: List[str]
    current_chunk_idx: int
    final_py_score: str
    suggestions: str

# Helper to extract 'owner/repo' from hyperlink
def parse_github_url(url: str) -> str:
    clean_url = url.strip().rstrip("/")
    parsed = urlparse(clean_url)
    path_parts = parsed.path.split('/')
    if len(path_parts) < 3:
        raise ValueError("Invalid GitHub repository link format.")
    return f"{path_parts[1]}/{path_parts[2]}"


# =====================================================================
# 2. DEFINE THE GRAPH NODES (FUNCTIONALITIES)
# =====================================================================

def scrape_github_node(state: AgentState) -> Dict[str, Any]:
    """Node: Connects to GitHub API in-memory and filters files."""
    st.info("🤖 Agent Node: Scraping repository data into memory...")
    repo_full_name = parse_github_url(state["repo_url"])
    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    
    # Load markdown files
    md_loader = GithubFileLoader(
        repo=repo_full_name,
        access_token=token,
        github_api_url="https://api.github.com",
        file_filter=lambda p: p.lower().endswith("readme.md")
    )
    md_docs = md_loader.load()
    md_content = md_docs[0].page_content if md_docs else "No README.md found."
    
    # Load Python files
    py_loader = GithubFileLoader(
        repo=repo_full_name,
        access_token=token,
        github_api_url="https://api.github.com",
        file_filter=lambda p: p.endswith(".py")
    )
    py_docs = py_loader.load()
    py_files = [{"path": doc.metadata.get("path", "code.py"), "content": doc.page_content} for doc in py_docs]
    
    return {
        "repo_full_name": repo_full_name,
        "md_content": md_content,
        "py_files": py_files,
        "chunk_summaries": [],
        "current_chunk_idx": 0
    }

def analyze_readme_node(state: AgentState) -> Dict[str, Any]:
    """Node: Processes MD files through Groq to rank concept novelty."""
    st.info("🤖 Agent Node: Analyzing project concept and README...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert venture capitalist and project auditor."),
        ("human", """Analyze this repository README documentation:
        ---
        {readme}
        ---
        Extract and summarize:
        1. Problem Statement
        2. Proposed Solution
        
        Provide an explicit score out of 10 for:
        - Originality of idea
        - Novelty
        - Viability of the problem and respective solution

        Summarise this score to return the average overall score out of 10 for the idea.
        Show all the scores along with final overall idea score. 
        
        Format output strictly with clear markdown headers and an obvious 'FINAL SCORES' block.""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"readme": state["md_content"]})
    
    return {
        "md_score": response.content,
        "md_summary": response.content  # Acts as the foundation context layer
    }

def process_py_chunks_node(state: AgentState) -> Dict[str, Any]:
    """Node: Incremental structure splitting to avoid context limit."""
    py_files = state["py_files"]
    current_idx = state["current_chunk_idx"]
    summaries = state["chunk_summaries"]
    
    if not py_files:
        return {"current_chunk_idx": -1}
        
    st.info(f"🤖 Agent Node: Memory Chunking - Processing script {current_idx + 1} of {len(py_files)}...")
    
    current_file = py_files[current_idx]
    
    # Split code structurally via Python syntax logic
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=1500,
        chunk_overlap=150
    )
    code_chunks = splitter.split_text(current_file["content"])
    
    file_summary_accumulator = []
    
    # Accumulate recursive understanding per script chunk
    for chunk in code_chunks:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an advanced software engineer tracking execution logic in state registers."),
            ("human", """Given the foundational project overview context:
            {readme_summary}
            
            Analyze this localized code snippet from the file '{filename}':
            ---
            {code}
            ---
            Summarize what these declarations/operations achieve and tools they utilise to achieve so dynamically to preserve execution memory.""")
        ])
        chain = prompt | llm
        res = chain.invoke({
            "readme_summary": state["md_summary"], 
            "filename": current_file["path"], 
            "code": chunk
        })
        file_summary_accumulator.append(f"[{current_file['path']}]: {res.content}")
        
    summaries.extend(file_summary_accumulator)
    
    return {
        "chunk_summaries": summaries,
        "current_chunk_idx": current_idx + 1
    }

def finalize_evaluation_node(state: AgentState) -> Dict[str, Any]:
    """Node: Synthesizes memories to grade technical engineering quality."""
    st.info("🤖 Agent Node: Executing codebase evaluation & grading...")
    
    accumulated_memory = "\n".join(state["chunk_summaries"])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Senior Principal Code Auditor."),
        ("human", """Review the summarized structural execution memory collected across this entire application repository:
        ---
        {code_memory}
        ---
        
        Comparing this execution against the initial foundation concept:
        ---
        {readme_summary}
        ---
        
        Grade the execution output from 1 to 10 on the following criteria:
        1. Execution Quality
        2. Code Correctness
        3. Relevance to the stated problem and solution
        4. Application structural mechanics (Mandatory use of concept of tools, Multi query retriever, and memory integrated with Langchain or Langgraph to carry out the execution)
        
        Use scores from these to give average overall marks for execution out of 10.
        Show all the scores and the final average score.

        Generate a detailed list of actionable suggestions for architectural improvements or missing components.""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "code_memory": accumulated_memory,
        "readme_summary": state["md_summary"]
    })
    
    # Split evaluation text from improvement tips for output layout sanitation
    content = response.content
    return {
        "final_py_score": content,
        "suggestions": content
    }


# =====================================================================
# 3. BUILD THE LANGGRAPH SYSTEM ROUTING WORKFLOW
# =====================================================================

def chunk_router_condition(state: AgentState):
    """Conditional Edge Router verifying loop conditions."""
    idx = state["current_chunk_idx"]
    total = len(state["py_files"])
    
    if idx == -1 or idx >= total:
        return "finalize"
    return "loop"

# Assemble workflow architecture
workflow = StateGraph(AgentState)

# Add operational states
workflow.add_node("scrape", scrape_github_node)
workflow.add_node("analyze_md", analyze_readme_node)
workflow.add_node("process_py", process_py_chunks_node)
workflow.add_node("finalize", finalize_evaluation_node)

# Set path routing rules
workflow.set_entry_point("scrape")
workflow.add_edge("scrape", "analyze_md")
workflow.add_edge("analyze_md", "process_py")

# Router determines whether to process next file chunk or break out
workflow.add_conditional_edges(
    "process_py",
    chunk_router_condition,
    {
        "loop": "process_py",
        "finalize": "finalize"
    }
)
workflow.add_edge("finalize", END)

# Compile into executable AI runtime module
agent_app = workflow.compile()
# from IPython.display import display, Image
# display(Image(agent_app.get_graph().draw_mermaid_png()))

# =====================================================================
# 4. STREAMLIT INTERACTION FRONTEND INTERFACE
# =====================================================================
st.set_page_config(page_title="GitAgent Repo Auditor", layout="wide")

st.title("🛡️ GitAgent: Repository Concept & Architecture Auditor")
st.markdown("Analyze GitHub codebases structurally without local cloning footprints.")

# Sidebar Configuration keys
with st.sidebar:
    st.header("🔑 Configurations")
    github_token = st.text_input("GitHub Classic Token (No scopes required for public)", type="password")
    agent_provider = st.selectbox("Agent Provider", ["OpenAI", "Gemini", "Anthropic", "Mistral", "Groq"])
    agent_key = st.text_input(f"{agent_provider} API Key", type="password")

# User Input Target Repository URL
repo_link = st.text_input("GitHub Hyperlink Target", placeholder="https://github.com/owner/repository")

if st.button("🚀 Execute Repository Audit", use_container_width=True):
    if not repo_link:
        st.error("Please provide a valid repository hyperlink target address.")
    elif not github_token or not agent_key:
        st.error("Please supply both the GitHub token and the selected agent API key.")
    else:
        # Dynamic context binding safely at frame invocation
        os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"] = github_token
        os.environ["OPENAI_API_KEY"] = agent_key if agent_provider == "OpenAI" else os.environ.get("OPENAI_API_KEY", "")
        os.environ["GOOGLE_API_KEY"] = agent_key if agent_provider == "Gemini" else os.environ.get("GOOGLE_API_KEY", "")
        os.environ["ANTHROPIC_API_KEY"] = agent_key if agent_provider == "Anthropic" else os.environ.get("ANTHROPIC_API_KEY", "")
        os.environ["MISTRAL_API_KEY"] = agent_key if agent_provider == "Mistral" else os.environ.get("MISTRAL_API_KEY", "")
        os.environ["GROQ_API_KEY"] = agent_key if agent_provider == "Groq" else os.environ.get("GROQ_API_KEY", "")

        try:
            global llm
            llm = build_llm(agent_provider, agent_key)
        except Exception as e:
            st.error(f"Unable to initialize {agent_provider} agent: {e}")
            st.stop()
        
        # Initialize LangGraph AgentState inputs
        initial_inputs = {
            "repo_url": repo_link,
            "repo_full_name": "",
            "md_content": "",
            "py_files": [],
            "md_summary": "",
            "md_score": "",
            "chunk_summaries": [],
            "current_chunk_idx": 0,
            "final_py_score": "",
            "suggestions": ""
        }
        
        with st.spinner("Agent state machine running..."):
            try:
                # Execute compiled application framework graph state transitions
                final_output = agent_app.invoke(initial_inputs)
                
                st.success("🎉 Audit Operations Complete!")
                
                # Render Split Results Interface Columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.header("📝 1. Concept & Ideation Review")
                    st.markdown(final_output["md_score"])
                    
                with col2:
                    st.header("💻 2. Execution & Architecture Review")
                    st.markdown(final_output["final_py_score"])

                report_markdown = f"""# GitAgent Repository Audit Report

## Repository
{repo_link}

## 1. Concept & Ideation Review
{final_output['md_score']}

## 2. Execution & Architecture Review
{final_output['final_py_score']}
"""
                st.download_button(
                    "⬇️ Download Markdown Report",
                    report_markdown,
                    file_name="repo_audit_report.md",
                    mime="text/markdown",
                )
                    
            except Exception as e:
                st.error(f"Runtime Operational failure during graph transition: {e}")