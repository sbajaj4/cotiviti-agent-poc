# Managed Multi-Agent Payment Integrity System (PoC)

An enterprise-grade, low-latency automated clinical claim audit pipeline. This system leverages a hybrid approach combining deterministic quantitative rule engines with lightweight, pattern-optimized LLM workers to detect complex billing anomalies, upcoding, and documentation mismatches.

## Key Features
- **Hybrid Guardrails:** Utilizes local statistical calculations ($Z$-score analysis) to intercept massive cost outliers deterministically before hitting LLM rate-limits.
- **Pattern-Optimized Semantic Screening:** Employs `gemini-3.1-flash-lite` with specialized token-efficient system prompts to catch deep structural laterality mismatches (e.g., ICD-10 side designations vs. conflicting operational text).
- **Contract Enforcement:** Utilizes rigid Pydantic structures to output consistently formatted, validated JSON audit logs.

## Tech Stack & Dependencies
- **Runtime:** Python 3.12+
- **Package Manager:** [uv](https://github.com/astral-sh/uv)
- **Core LLM:** Google Gemini API (`gemini-3.1-flash-lite`)
- **Data Validation:** Pydantic

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/cotiviti-agent-poc.git](https://github.com/YOUR_USERNAME/cotiviti-agent-poc.git)
   cd cotiviti-agent-poc
   ```

2. **Configure Environment Variables:**

    Copy the example file and add your Google AI Studio API key (a free key should be sufficient):
    ```bash
   cp.env.example .env
   # Open .env and add your GEMINI_API_KEY
   ```

3. **Run the Pipeline Live:**

    This project uses `uv` for seamless, isolated script execution with zero configuration:
    ```bash
    uv run --env-file .env audit_pipeline.py
    ```

## Sample Output

Upon completion, the orchestrator generates a strictly structured `final_audit_results.json` artifact mapping metrics across 10 distinct scenario profiles, successfully identifying unit ceiling breaches, cost outliers, and semantic contradictions.