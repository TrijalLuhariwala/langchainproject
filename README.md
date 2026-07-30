# 🛡️ GitAgent: Repository Concept & Architecture Auditor

GitAgent is an autonomous AI agent engineered to perform comprehensive, multi-stage code reviews and project conceptual evaluations on remote GitHub repositories. Built using **LangGraph** for deterministic state management, **LangChain** for document handling, and **ChatGroq (Llama-3)** for rapid infrastructure analysis, GitAgent provides detailed analytical scoring and structural critiques completely **in-memory**, with a zero local-disk footprint.

---

## 📌 Problem Statement

Reviewing open-source or submitted project code repositories thoroughly is incredibly time-consuming for hackathon judges, technical recruiters, and senior architects. The manual workflow suffers from three major points of friction:
1. **Context Fragmentation:** Reviewers look at a `README.md` file to understand what the project *claims* to do, but quickly lose track of that baseline premise when diving into hundreds of lines of disjointed raw source code.
2. **Local Machine Clutter:** Traditional evaluation workflows require performing a local `git clone` or downloading zip files, scattering untrusted repositories across the auditor's filesystem.
3. **Context Window Limitations:** Modern LLMs break down when an entire codebase is dumped into a single prompt, resulting in high token costs, broken code context, and missed architectural flaws.

---

## 💡 The Solution: Hybrid State Machine Chunking

GitAgent solves these vulnerabilities by creating a structured **StateGraph Agent Workflow** combined with **Dynamic Text Chunking** and **Ephemeral In-Memory Pipelines**:
* **In-Memory Streaming:** Utilizing specialized API-level loaders, the agent streams the repository files straight into RAM using GitHub's REST architecture—ensuring **zero local storage footprint**.
* **Conceptual Anchoring:** The `README.md` is extracted and audited *first* to map out the core problem and solution framework, establishing a programmatic memory profile. 
* **Grammar-Aware Splitters:** Instead of slicing raw code arbitrarily, the agent splits code by prioritizing semantic indentation and syntax lines, ensuring logical units stay intact.
* **Recursive Ephemeral Memory Loop:** The agent steps through code files one by one, continuously updating its understanding based on both the original README intent and previous chunk reviews.

---

## ⚙️ Architecture & Project Flow

The following graph maps out the precise, automated loop execution enforced by the LangGraph engine:
+-----------------------+
|   Input: GitHub URL   |
+-----------+-----------+
            |
            v
 +---------------------+
 |  [1] Scrape GitHub  |  <--- Over the API (In-Memory Only)
 +-----------+---------+
             |
             v
 +---------------------+
 | [2] Analyze README  |  ---> Computes Ideation & Novelty Scores
 +-----------+---------+
             |
             v
 +---------------------+
 |  [3] Process Chunks |  ---> Iterates script-by-script via Python logic
 +-----------+---------+
             |
             v
 +---------------------+
 |    Router Status    |
 +-----+-----------+---+
            | 
            v
+---------------------+
|  [4] Finalize Audit | ---> Grades Execution & Structural Mechanics
+----------+----------+
           |
           v
+---------------------+
| Streamlit UI Output | ---> Dual Scoring & System Suggestions
+---------------------+

### Detailed Node Execution Sequence

#### 1. Data Ingestion Node (`scrape`)
* Captures the input hyperlink and extracts the `owner/repository` string safely.
* Triggers a LangChain custom `@tool` backed by `GithubFileLoader`. 
* Streams down all `.md` and `.py` source documents as separate byte packages straight into the application `AgentState`.

#### 2. Ideation Engine Node (`analyze_md`)
* Feeds the pure documentation payload to `ChatGroq`.
* Isolates the project's Core Problem Statement and Proposed Solution.
* Scores the project out of 10 based on **Originality**, **Novelty**, and **Viability**, saving this summary directly into the agent's foundation context layer.

#### 3. Ephemeral Memory Loop Node (`process_py`)
* Employs the `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)`.
* Breaks down individual python scripts sequentially into manageable 1,500-character logical chunks with 150-character contextual bridges.
* For each chunk, it passes the code to the LLM alongside the *Readme Summary Memory*. The LLM evaluates the snippet's purpose and stores this analysis inside an accumulator stack (`chunk_summaries`).
* A conditional router edge calculates `current_chunk_idx`. If additional unread `.py` arrays remain, it loops back to analyze the next block.

#### 4. Architecture Synthesis Node (`finalize`)
* Once the loop terminates, the compiled accumulator memory string represents the entire application's flow.
* The agent reads the composite memory bank to score the code out of 10 based on **Execution Quality**, **Syntax Correctness**, **Relevance**, and **Structural Mechanics** (specifically noting implementation parameters like agents, chains, or external frameworks).
* Spits out an automated, actionable "Suggestions for Improvement" ledger.

---

## 🛠️ Tech Stack & Core Libraries

* **State Machine Logic:** `langgraph` (Enforces deterministic processing states, state variables, and conditional edge routing rules).
* **LLM Orchestration:** `langchain-groq` (Utilizing highly accelerated `llama3-70b` models for massive throughput speed).
* **Document Ingestion:** `langchain-community` (`GithubFileLoader` streaming protocols).
* **Grammar Chunking Engine:** `langchain-text-splitters` (`RecursiveCharacterTextSplitter` matching `Language.PYTHON` specifications).
* **Interface Layer:** `streamlit` (A hyper-clean, minimal web user interface).
