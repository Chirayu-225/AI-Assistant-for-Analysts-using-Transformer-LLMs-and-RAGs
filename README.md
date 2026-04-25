# AI-Assistant-for-Analysts-using-Transformer-LLMs-and-RAGs

Analyst Assistant is a fully autonomous, air-gapped, multi-agent AI analytics platform. It empowers users to interrogate both structured datasets (CSVs) and unstructured documents using natural language, all while guaranteeing 100% data privacy and zero cloud dependency.

Unlike standard LLM chatbots that hallucinate math, this platform utilizes a Dual-Engine Architecture. It routes tabular queries to an Agentic Code Execution sandbox for deterministic accuracy, and text queries to a Vector RAG pipeline for semantic reading comprehension.

✨ Key Features

🧠 Dual-Engine Routing: Agentic Code Execution (Tabular Data): For CSVs, the LLM acts as a programmer, writing and executing deterministic pandas and matplotlib code to answer queries with 100% mathematical accuracy.

Vector RAG (Unstructured Data): For text documents, the system seamlessly switches to a Retrieval-Augmented Generation pipeline for semantic search and context-aware summarization.

🛡️ Hardened Execution Sandbox: Dynamically generated code is intercepted by an AST-level pre-flight scanner that blocks infinite loops, OS commands, and Python __dunder__ methods. Code runs in a restricted exec() environment with __builtins__ disabled to prevent sandbox escapes.

🧹 Hybrid Data Engineering: Raw datasets pass through a deterministic rule-based cleaning pipeline (normalizing headers, dates, currencies, and nulls) before falling back on an LLM agent to resolve ambiguous edge cases.

📈 Proactive Profiling (Tabular RAG): The system computes a structured statistical profile of the data (IQR outliers, Pearson correlations > 0.5, top categoricals) to feed the reasoning model, generating C-suite-level business insights without raw table dumps.

💬 Stateful Conversational Memory: A sliding-window memory buffer tracks previous queries and execution results, allowing for natural, context-aware follow-up questions.

🏗️ System Architecture
The application is heavily modularized to separate the UI, execution, and reasoning logic:
main.py: The core controller and Streamlit UI rendering engine
cleaning.py: The hybrid rule-based and LLM data standardization pipeline.
query_engine.py: The prompt orchestrator and code generation agent (Qwen 2.5 Coder).
sandbox.py: The AST security scanner and isolated Python execution environment.
profiler.py: The statistical engine that maps dataset schemas and calculates mathematical profiles.
rag_engine.py: The vector embedding and semantic search pipeline for unstructured document ingestion.
eval.py: The reasoning agent (Qwen 2.5 Base) for insight generation and strategic drill-down explanations.

🛠️ Technology Stack
Frontend & State Management: Streamlit
Local LLM Runtime: Ollama
AI Models: * qwen2.5-coder:3b (Code Generation & Math)
qwen2.5:3b (Reasoning, Insights & RAG)
Data Engineering & Computing: Pandas, NumPy
Data Visualization: Matplotlib (styled for dark-mode UI integration)
