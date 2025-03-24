# 🤖 Tech Support Chat Agent BR v4.0

A smart, memory-aware chatbot that uses GPT-3.5 and your own tech support documents to provide Tier-1 troubleshooting assistance — deployed with Streamlit.

---

## 🚀 Features

- 📁 Upload your own tech support Q&A CSV
- 🧠 RAG (Retrieval-Augmented Generation) with FAISS
- 💬 GPT-3.5-powered responses
- 📝 Multi-turn chat memory
- 🌐 Simple Streamlit interface
- 🔁 Reset and refresh chat anytime

---

## 📦 Files in This Repo

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit chatbot code |
| `requirements.txt` | Required Python libraries |
| `tech_support_sample_QA.csv` | (Optional) Example CSV to try locally |

---

## 📄 CSV Format
Upload a CSV file with two columns: `question`, `answer`

```csv
question,answer
My screen is black,Fully power off and reseat the RAM module
Laptop makes 3 beeps,Indicates memory issue. Try a different slot.
```

---

## 🔐 OpenAI API Key

You’ll need your own OpenAI API key to run the chatbot.
1. Get yours from https://platform.openai.com/account/api-keys
2. Paste it into the sidebar field in the app

---

## 🌍 How to Run (Locally)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Cloud

1. Fork or clone this repo to your GitHub
2. Visit https://streamlit.io/cloud
3. Connect your GitHub and deploy the repo

---

## 📬 Contact
Made by [Your Name]  
For feedback, email [your.email@example.com]

---

## 📜 License
MIT License. Feel free to use, modify, and build upon it!
