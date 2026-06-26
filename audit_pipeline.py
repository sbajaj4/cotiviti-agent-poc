# /// script
# dependencies = [
#   "google-genai>0.1.0",
#   "pydantic>=2.0.0"
# ]
# ///

import os
import json
import math
import time
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Initialize Google GenAI Client
client = genai.Client()

# =============================== #
# 1. Mock Dataset
# =============================== #
CLAIMS_DB = [
    {
        "claim_id": "CLM-001",
        "icd_10_code": "M17.11",  # Unilateral primary osteoarthritis, right knee
        "cpt_code": "27447",     # Total knee arthroplasty (knee replacement)
        "billing_units": 2,       # SCENARIO A: Unit Ceiling Breach (Matrix Max = 1). Triggers deep clinical loop.
        "historical_provider_costs": [1200, 1150, 1300, 1250, 1220], 
        "clinical_notes": "Patient underwent uncomplicated right total knee replacement. Left knee examined and normal."
    },
    {
        "claim_id": "CLM-002",
        "icd_10_code": "J45.909", # Unspecified asthma
        "cpt_code": "94640",     # Inhalation treatment
        "billing_units": 1,       
        "historical_provider_costs": [85, 90, 80, 95, 2400], # SCENARIO B: Severe Time-Series Outlier (Z > 2.5). Triggers guardrail override.
        "clinical_notes": "Acute asthma exacerbation treated with standard nebulizer in office. Symptoms resolved cleanly."
    },
    {
        "claim_id": "CLM-003",
        "icd_10_code": "I10",     # Essential hypertension
        "cpt_code": "99213",     # Outpatient doctor visit (15 min)
        "billing_units": 1,       # SCENARIO C: Complete Auto-Approval. Passes both Quant and Fast Semantic filters cleanly.
        "historical_provider_costs": [150, 160, 145, 155, 152],
        "clinical_notes": "Routine follow-up for blood pressure management. Patient compliant with daily Lisinopril medication."
    },
    {
        "claim_id": "CLM-004",
        "icd_10_code": "S62.610A", # Displaced fracture of index finger
        "cpt_code": "26735",     # Open treatment of fracture (Not in matrix)
        "billing_units": 1,       # SCENARIO D: Minor cost variation ignored by updated Z-score. Auto-approves cleanly.
        "historical_provider_costs": [850, 900, 875, 920, 895], 
        "clinical_notes": "Surgical fixation of right index finger fracture using micro-plates. Post-operative alignment looks excellent."
    },
    {
        "claim_id": "CLM-005",
        "icd_10_code": "E11.9",   # Type 2 diabetes mellitus
        "cpt_code": "99214",     # Outpatient doctor visit (extended) (Not in matrix)
        "billing_units": 1,       
        "historical_provider_costs": [210, 220, 205, 215, 600], # SCENARIO E: Unlisted code with severe cost anomaly (Z > 2.5). Flagged for review.
        "clinical_notes": "Quarterly diabetes management consultation. A1C stable at 6.8. Renewed Metformin prescription."
    },
    {
        "claim_id": "CLM-006",
        "icd_10_code": "M17.11",  # ICD-10 specifies RIGHT knee primary osteoarthritis
        "cpt_code": "27447",      # SCENARIO F: DUAL-PATH TRIGGER. Fast semantic screen catches that diagnosis code M17.11 (Right) flatly 
        "billing_units": 1,       # Math parameters look perfectly normal. Bypasses the initial quantitative rule checks.
        "historical_provider_costs": [1200, 1150, 1300, 1250, 1210], 
        "clinical_notes": "Successful left total knee replacement performed. Right knee previously replaced last calendar year.",
    },
    {
        "claim_id": "CLM-007",
        "icd_10_code": "G43.909", # Migraine, unspecified
        "cpt_code": "99213",     
        "billing_units": 1,       
        "historical_provider_costs": [150, 145, 160, 155, 380], # SCENARIO G: Evaluation code displaying a true Z-score breach (> 2.5).
        "clinical_notes": "Patient presents with acute refractory migraine. Discussion focused on adjusting preventive lifestyle measures."
    },
    {
        "claim_id": "CLM-008",
        "icd_10_code": "M79.606", # Pain in lower leg, unspecified
        "cpt_code": "73562",     # X-ray exam of knee, 3 views
        "billing_units": 5,       # SCENARIO H: Matrix Ceiling Breach (Matrix Max = 3). Flagged immediately on rules.
        "historical_provider_costs": [110, 115, 105, 120, 112],
        "clinical_notes": "Obtained standard radiological views of the left knee following minor trauma. No acute fracture observed."
    },
    {
        "claim_id": "CLM-009",
        "icd_10_code": "Z00.00",  # General adult examination
        "cpt_code": "99395",     
        "billing_units": 1,       # SCENARIO I: Clean operational baseline tracking. Auto-approved.
        "historical_provider_costs": [250, 240, 260, 255, 248],
        "clinical_notes": "Annual physical examination. All age-appropriate wellness screening metrics discussed and within standard limits."
    },
    {
        "claim_id": "CLM-010",
        "icd_10_code": "J01.90",  # Acute sinusitis
        "cpt_code": "99212",     
        "billing_units": 1,       # SCENARIO J: Clean operational baseline tracking. Auto-approved.
        "historical_provider_costs": [95, 100, 90, 105, 98],
        "clinical_notes": "Patient presenting with sinus congestion and pressure. Diagnosed with viral sinusitis and instructed on supportive care."
    }
]

# =============================== #
# 2. Pydantic Schema for Enforced Structured Outputs
# =============================== #

class AuditReportSchema(BaseModel):
    target_claim_id: str = Field(description="The unique identifier of the audited medical claim.")
    audit_status: str = Field(description="Must be exactly one of 'APPROVED', 'REJECTED', or 'FLAGGED_FOR_HUMAN_REVIEW'.")
    risk_classification: str = Field(description="Risk tier based on financial or clinical severity. must be: 'LOW', 'MEDIUM', or 'HIGH'.")
    quantitative_anomalies_found: List[str] = Field(description="List of raw data or statistical discrepancies caught.")
    clinical_justification: str = Field(description="The logical reasoning explaining the semantic contradiction or alignment between codes and notes.")
    confidence_score: float = Field(description="The agent's confidence in its verdict from 0.0 (completely unsure) to 1.0 (absolute certainty).")
    suggested_action: str = Field(description="Actionable corporate next step regarding payment or provider outreach.")

# =============================== #
# 3. Agent to Find Anomalies
# =============================== #

class PatternMatcherWorker:
    def __init__(self, name: str):
        self.name = name
        self.UNIT_CEILING_MATRIX = {
            "27447": 1,  # Total knee arthroplasty: max 1 unit per target anatomy side
            "73562": 3,  # Diagnostic knee X-ray (3 views): max 3 units per series
            "94640": 2,  # Inhalation treatments: max 2 units per calendar date
            "99213": 1,  # Outpatient evaluation visit: max 1 unit per encounter
        }
    
    def execute_task(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        alerts = []
        z_score = 0.0
        
        cpt = claim["cpt_code"]
        units = claim["billing_units"]
        
        if cpt in self.UNIT_CEILING_MATRIX:
            max_allowed = self.UNIT_CEILING_MATRIX[cpt]
            if units > max_allowed:
                alerts.append(
                    f"Unit Threshold Breach: CPT {cpt} billed for {units} units. Maximum allowed structural ceiling is {max_allowed} unit(s)."
                )
        
        costs = claim["historical_provider_costs"]
        current_cost = costs[-1]
        history = costs[:-1]
        n = len(history)
        
        if n > 1:
            mean = sum(history) / n
            variance = sum((x - mean) ** 2 for x in history) / n
            std_dev = math.sqrt(variance)
            
            if std_dev > 0:
                z_score = (current_cost - mean) / std_dev
                if z_score > 2.5:
                    alerts.append(f"Statistical Time-Series Anomaly: Cost of ${current_cost} yields a Z-Score of {z_score:.2f} (Extreme Outlier).")
        
        return {
            "status": "TRIGGERED" if alerts else "PASSED",
            "findings": alerts,
            "calculated_anomaly_score": round(min(1.0, max(0.0, z_score / 5.0)), 2)
        }

# =============================== #
# 4. Agent conducting evaluation
# =============================== #

class ClinicalReasonerWorker:
    def __init__(self, name: str):
        self.name = name
    
    def fast_semantic_screening(self, claim: Dict[str, Any]) -> bool:
        system_prompt = (
            "You are an expert clinical coding validation engine operating with absolute precision.\n\n"
            "CRITICAL LOGICAL RULE:\n"
            "Cross-reference the clinical definitions, anatomical parameters, and intent of the provided "
            "ICD-10 and CPT codes against the literal facts recorded in the Clinical Chart Notes.\n"
            "You must flag an absolute contradiction if there is an irreconcilable conflict between the "
            "administrative codes and the text notes, including but not limited to:\n"
            "- Mismatched Procedures: The CPT code describes an entirely different operation or service than what was performed.\n"
            "- Conflicting Diagnoses: The ICD-10 code indicates a condition that flatly contradicts the doctor's chart findings.\n"
            "- Anatomical/Laterality Errors: The codes specify a body part, organ system, or side (Left/Right) opposite to the text.\n\n"
            "EXAMPLES:\n"
            "- ICD-10: J01.90 (Sinusitis) | CPT: 99212 | Notes: 'Patient presents with a severe sprained ankle following a fall.' -> Output: CONFLICT\n"
            "- ICD-10: M75.111 (Rotator Cuff Tear, Right Shoulder) | CPT: 23412 | Notes: 'Patient presents for surgical repair of a massive left rotator cuff tear.' -> Output: CONFLICT\n"
            "- ICD-10: I10 (Hypertension) | CPT: 99213 | Notes: 'Routine follow-up for blood pressure management.' -> Output: CLEAR\n\n"
            "OUTPUT RULES:\n"
            "- Output exactly 'CONFLICT' if any structural, procedural, or factual contradiction is detected.\n"
            "- Output exactly 'CLEAR' if the codes, services, and chart notes align perfectly."
        )
        user_prompt = f"ICD-10: {claim['icd_10_code']}\nCPT: {claim['cpt_code']}\nNotes: {claim['clinical_notes']}"
        
        try:
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite',
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                )
            )
            return "CONFLICT" in response.text.upper()
        except Exception:
            return False
    
    def execute_task(self, claim: Dict[str, Any], pattern_findings: List[str]) -> str:
        system_prompt = (
            "You are a Senior Clinical Audit Specialist. Analyze the relationship between medical codes, "
            "clinical chart notes, and quantitative data exceptions. Use Chain-of-Thought reasoning to identify "
            "contradictions, fraudulent billing patterns, or medical necessity issues."
        )
        user_prompt = f"""
        Review the following data context:
        - Claim ID: {claim['claim_id']}
        - Diagnosis (ICD-10): {claim['icd_10_code']}
        - Procedure (CPT): {claim['cpt_code']}
        - Quantitative Alerts Raised: {json.dumps(pattern_findings)}
        - Clinical Chart Notes: "{claim['clinical_notes']}"
        
        Task: Provide a detailed, step-by-step clinical validation evaluation. Explain whether the text notes justify the bill or support the quantitative alerts.
        """
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.2)
        )
        return response.text

# =============================== #
# 3. Agent to Enforce Schema Validation
# =============================== #

class ReportStructurerWorker:
    def __init__(self, name: str):
        self.name = name
    
    def execute_task(self, claim_id: str, alerts: List[str], clinical_text: str, payload: Dict[str, Any]) -> AuditReportSchema:
        system_prompt = "You are a data harmonization pipeline agent. Package unstructured audit details into a clean JSON structure."
        
        user_prompt = f"""
        Construct the final compliance artifact for:
        Claim ID: {claim_id}
        Statistical Anomaly Score: {payload.get('calculated_anomaly_score', 0.0)}
        Alerts Raised: {json.dumps(alerts)}
        Clinical Text Analysis: {clinical_text}
        
        Task: Map these matrices into the requested JSON schema. Deduced confidence_score should reflect both text clarity and the statistical anomaly score weight.
        """
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=AuditReportSchema,
                temperature=0.1
            )
        )
        return AuditReportSchema.model_validate_json(response.text)

# =============================== #
# 3. Proactive orchestration agent
# =============================== #

class AuditOrchestrator:
    def __init__(self):
        self.quant_agent = PatternMatcherWorker("Quant-Data-Agent")
        self.clinical_agent = ClinicalReasonerWorker("Clinical-Gemini-Reasoner")
        self.struct_agent = ReportStructurerWorker("Structured-Schema-Agent")
        self.audit_log: List[Dict[str, Any]] = []
    
    def run_pipeline(self, dataset: List[Dict[str, Any]]):
        print("=============================================================")
        print("[Orchestrator]: Executing Managed Multi-Agent Payment Integrity System")
        print("=============================================================\n")
        
        total_claims = len(dataset)
        
        for idx, claim in enumerate(dataset, 1):
            progress = int((idx / total_claims) * 20)
            bar = "█" * progress + "░" * (20 - progress)
            print(f"Progress: |{bar}| {idx} / {total_claims} Claims Processed")
            print(f"--- COMMENCING AUDIT PROCESS FOR: {claim['claim_id']} ---")
            
            quant_state = self.quant_agent.execute_task(claim)
            
            if quant_state["status"] == "PASSED":
                print(f"[Quant Engine]: Cleared. Routing to high-speed semantic screening...")
                
                has_text_conflict = self.clinical_agent.fast_semantic_screening(claim)
                
                if not has_text_conflict:
                    print(f"[Orchestrator]: No semantic friction detected. Auto-Approved.\n")
                    self.audit_log.append({
                        "target_claim_id": claim["claim_id"],
                        "audit_status": "APPROVED",
                        "risk_classification": "LOW",
                        "quantitative_anomalies_found": [],
                        "clinical_justification": "Automated quantitative and fast-semantic checks passed standard baseline criteria.",
                        "confidence_score": 1.0,
                        "suggested_action": "Process payment immediately."
                    })
                    continue
                else:
                    print(f"[Orchestrator]: Hidden semantic mismatch caught! Escalating to deep multi-agent evaluation...")
                    quant_state["findings"] = ["Factual Laterality/Coding Contradiction discovered in chart notes."]
                    quant_state["calculated_anomaly_score"] = 0.20
            
            print(f"[Alerts Found]: {quant_state['findings']} (Score: {quant_state['calculated_anomaly_score']})")            
            print(f" -> Passing context to [{self.clinical_agent.name}]...")
            clinical_reasoning = self.clinical_agent.execute_task(claim, quant_state["findings"])
            
            time.sleep(5)
            
            print(f" -> Passing matrix to [{self.struct_agent.name}] for contract enforcement...")
            final_report = self.struct_agent.execute_task(
                claim["claim_id"], 
                quant_state["findings"], 
                clinical_reasoning,
                quant_state
            )
            
            if final_report.audit_status == "APPROVED" and quant_state["calculated_anomaly_score"] >= 0.50:
                print("[GUARDRAIL]: AI verdict contradicted severe data outliers. Overriding to review")
                final_report.audit_status = "FLAGGED_FOR_HUMAN_REVIEW"
                final_report.risk_classification = "HIGH"
                final_report.suggested_action = "Deterministic threshold breach. Route to manual clinical specialist loop."
            
            self.audit_log.append(final_report.model_dump())
            
            print("[Orchestrator]: Verified Structured Artifact Output:")
            print(json.dumps(final_report.model_dump(), indent=4))
            print("---------------------------------------------------------------\n")
        
        time.sleep(15)
                    
        approved_count = sum(1 for c in self.audit_log if c["audit_status"] == "APPROVED")
        rejected_count = sum(1 for c in self.audit_log if c["audit_status"] == "REJECTED")
        flagged_count = sum(1 for c in self.audit_log if c["audit_status"] == "FLAGGED_FOR_HUMAN_REVIEW")
        
        print("================================================================")
        print("SYSTEM METRICS SUMMARY REPORT")
        print("================================================================")
        print(f"AUTO-APPROVED CLAIMS          : {approved_count}")
        print(f"REJECTED CLAIMS               : {rejected_count}")
        print(f"FLAGGED FOR HUMAN AUDIT       : {flagged_count}")
        print(f"TOTAL OVERHEAD INGESTED       : {total_claims} Claims")
        print("================================================================")

        output_filename = "final_audit_results.json"
        with open(output_filename, "w") as f:
            json.dump(self.audit_log, f, indent=4)
        print(f"[Orchestrator]: Final audit run logged to static artifact: '{output_filename}'\n")

if __name__ == "__main__":
    orchestrator = AuditOrchestrator()
    orchestrator.run_pipeline(CLAIMS_DB)