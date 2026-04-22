# Geospatial-Rent-Analytics-AI-Decision-Support-System
🚀 **Live App:** [Open Dashboard](https://geospatial-rent-analytics-ai-decision-support-system-ltzib4zib.streamlit.app)
- An end-to-end ML + AI-powered housing intelligence platform that analyzes HUD Fair Market Rent data across 3,200+ counties to forecast rent trends and generate personalized recommendations.
## ✨ Features
- 📊 Interactive dashboard with multi-tab analytics
- 📈 FY27 rent forecasting using Linear Regression + ARIMA ensemble
- 🤖 AI-powered county recommender (Llama 3 via Groq API)
- 🔍 Data explorer with filtering & sorting
- ☁️ Deployed on Streamlit Cloud
## 🛠 Tech Stack
- Python, Pandas, NumPy
- Machine Learning: Linear Regression, ARIMA
- Visualization: Streamlit
- LLM Integration: Groq API (Llama 3)
- Deployment: Streamlit Cloud
## 💻 Local Setup Instructions
```bash
git clone https://github.com/addagadanamratha/Geospatial-Rent-Analytics-AI-Decision-Support-System.git
cd Geospatial-Rent-Analytics-AI-Decision-Support-System
uv sync
# OR
pip install -r requirements.txt
streamlit run app.py
## 🧠 Architecture
Data → Feature Engineering → ML Models (LR + ARIMA) → Ensemble → Streamlit UI → LLM Recommendations
