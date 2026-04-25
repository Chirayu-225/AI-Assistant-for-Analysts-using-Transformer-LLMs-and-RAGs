# AI-Assistant-for-Analysts-using-Transformer-LLMs-and-RAGs

Analyst Assistant is a fully autonomous, air-gapped, multi-agent AI analytics platform. It empowers users to interrogate both structured datasets (CSVs) and unstructured documents using natural language, all while guaranteeing 100% data privacy and zero cloud dependency.

Unlike standard LLM chatbots that hallucinate math, this platform utilizes a Dual-Engine Architecture. It routes tabular queries to an Agentic Code Execution sandbox for deterministic accuracy, and text queries to a Vector RAG pipeline for semantic reading comprehension.

✨ **Key Features**

🧠 Dual-Engine Routing: Agentic Code Execution (Tabular Data): For CSVs, the LLM acts as a programmer, writing and executing deterministic pandas and matplotlib code to answer queries with 100% mathematical accuracy.

🧹 Vector RAG (Unstructured Data): For text documents, the system seamlessly switches to a Retrieval-Augmented Generation pipeline for semantic search and context-aware summarization.

🛡️ Hardened Execution Sandbox: Dynamically generated code is intercepted by an AST-level pre-flight scanner that blocks infinite loops, OS commands, and Python __dunder__ methods. Code runs in a restricted exec() environment with __builtins__ disabled to prevent sandbox escapes.

🧹 Hybrid Data Engineering: Raw datasets pass through a deterministic rule-based cleaning pipeline (normalizing headers, dates, currencies, and nulls) before falling back on an LLM agent to resolve ambiguous edge cases.

📈 Proactive Profiling (Tabular RAG): The system computes a structured statistical profile of the data (IQR outliers, Pearson correlations > 0.5, top categoricals) to feed the reasoning model, generating C-suite-level business insights without raw table dumps.

💬 Stateful Conversational Memory: A sliding-window memory buffer tracks previous queries and execution results, allowing for natural, context-aware follow-up questions.

🏗️ **System Architecture**

The application is heavily modularized to separate the UI, execution, and reasoning logic:

main.py: The core controller and Streamlit UI rendering engine

cleaning.py: The hybrid rule-based and LLM data standardization pipeline.

query_engine.py: The prompt orchestrator and code generation agent (Qwen 2.5 Coder).

sandbox.py: The AST security scanner and isolated Python execution environment.

profiler.py: The statistical engine that maps dataset schemas and calculates mathematical profiles.

rag_engine.py: The vector embedding and semantic search pipeline for unstructured document ingestion.

eval.py: The reasoning agent (Qwen 2.5 Base) for insight generation and strategic drill-down explanations.



🛠️ **Technology Stack**

Frontend & State Management: Streamlit

Local LLM Runtime: Ollama

AI Models: * qwen2.5-coder:3b (Code Generation & Math)

Data Visualization: Matplotlib (styled for dark-mode UI integration)



💡 **Usage Workflow**

Ingest: Upload a messy CSV or text document via the sidebar.

Auto-Clean: Click Initialize Auto-Clean Pipeline to standardize the data deterministically.

Query: Ask a natural language question (e.g., "What is the correlation between price and volume?").

Execute: The system will generate the code, scan it for security, run it, and output the mathematical result or auto-render a chart.

Analyze: Click Generate Insights to get a structured business breakdown of the data, and paste any bullet point into the Drill-Down engine for a deep-dive explanation.

qwen2.5:3b (Reasoning, Insights & RAG)

Data Engineering & Computing: Pandas, NumPy


💡 **Snapshots of the Project**

1) Main UI

<img width="1429" height="700" alt="image" src="https://github.com/user-attachments/assets/c2c582dc-ce95-4552-ae09-b19f1fc6982f" />


2) Data Cleaning
   
<img width="1472" height="697" alt="image" src="https://github.com/user-attachments/assets/0b8f3a1c-fca3-4c77-a762-ac02ef3ed13f" />


3) Auto Analyzing and Insight Generation
   
<img width="1750" height="701" alt="image" src="https://github.com/user-attachments/assets/bdc1d8e8-c131-48f3-9b21-f5abb9ab8bd1" />
<img width="1400" height="703" alt="image" src="https://github.com/user-attachments/assets/496f0197-977c-4d80-81ed-e3c24f6620ad" />
<img width="1379" height="703" alt="image" src="https://github.com/user-attachments/assets/d89b2176-1a81-4229-b69a-947067971544" />


4) Unstructured Data

<img width="1490" height="700" alt="image" src="https://github.com/user-attachments/assets/7bd593cb-8a43-4f26-b85c-ca3f0f959732" />
<img width="1909" height="553" alt="image" src="https://github.com/user-attachments/assets/80284c48-0374-4f04-9969-50304e91dec4" />


