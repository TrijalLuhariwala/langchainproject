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