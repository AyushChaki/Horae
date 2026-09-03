# 🛡️ HORAЕ
### AI-Powered Chargeback Intelligence & Merchant Risk Defense Platform

<p align="center">

  <strong>Turn transaction risk + policy intelligence + evidence into an actionable chargeback defense.</strong>

  <br/><br/>

  <a href="YOUR_STREAMLIT_URL">
    <img src="https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  </a>
  <a href="YOUR_GITHUB_URL">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white"/>
  </a>

</p>

<p align="center">

  <img src="https://img.shields.io/badge/Python-3.14+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-Risk%20Engine-EC4E20?style=flat-square"/>
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-0467DF?style=flat-square"/>
  <img src="https://img.shields.io/badge/Mistral-AI%20Defense-FF7000?style=flat-square"/>
  <img src="https://img.shields.io/badge/SentenceTransformers-RAG-00A67E?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Production%20Demo-success?style=flat-square"/>

</p>

---

## ⚡ What is Horae?

**Horae** is an AI-powered **chargeback intelligence and merchant risk defense platform** designed to help merchants identify risky transactions, quantify financial exposure, assess dispute evidence, retrieve applicable policies, and generate evidence-backed chargeback responses.

Instead of treating a chargeback as an isolated customer-support event, Horae connects the entire decision chain:

```text
TRANSACTION
     │
     ▼
┌─────────────────────────┐
│   RISK INTELLIGENCE     │
│ XGBoost + Behavioral    │
│ & Transaction Signals   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ COST-AWARE DECISIONING  │
│ Expected Value /        │
│ Financial Optimization  │
└────────────┬────────────┘
             │
             ▼
       CHARGEBACK
          EVENT
             │
             ▼
┌─────────────────────────┐
│       POLICY RAG        │
│ SentenceTransformers    │
│      + FAISS            │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   EVIDENCE ENGINE       │
│ Transaction + Dispute   │
│ Evidence Assessment     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    AI DEFENSE LAYER     │
│       Mistral AI        │
│ + Deterministic Fallback│
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ MERCHANT DECISION       │
│ DEFEND / REVIEW /       │
│ INSUFFICIENT EVIDENCE   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ AUDITABLE CASE OUTPUT   │
│ Evidence + Policy +     │
│ Recommendation + Draft  │
└─────────────────────────┘
