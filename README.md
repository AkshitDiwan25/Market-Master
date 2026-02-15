# Retail Intelligence AI Platform

AI-powered retail market intelligence platform that transforms structured sales data into executive-ready strategic insights.

This application enables users to:
- Upload or use a preloaded retail dataset
- Generate consulting-grade executive summaries
- Interact with an AI-powered retail strategy chatbot
- Generate AI-based market news relevant to the selected category

---

##  Problem Statement

Retail leaders often struggle to quickly extract decision-ready insights from raw sales datasets.  
Traditional BI tools provide charts, but not strategic recommendations.

This project solves that gap by:
- Converting structured retail sales data into consulting-grade insights
- Enabling conversational exploration of performance drivers
- Connecting dataset signals to broader market dynamics

---

##  Key Features

### 1. AI Executive Summary
Generates structured, boardroom-ready insights including:
- Executive Summary
- Geographic Split
- Competitive Landscape
- Strategic Recommendations (Quick Wins & Structural Moves)
- Macro Trend Linkage

### 2. AI Chatbot
Interactive retail strategy assistant that:
- Answers questions about category performance
- Responds based on dataset context
- Maintains conversational memory

### 3. AI Market News Module
Generates structured retail news summaries aligned with the selected category.

---

##  Tech Stack

- **Frontend/UI:** Gradio (responsive, theme-adaptive)
- **Backend Logic:** Python
- **LLM Provider:** Groq API
- **Model Used:** `openai/gpt-oss-20b`
- **Data Handling:** Pandas
- **Environment Management:** python-dotenv

---

##  Project Structure
├── app.py
├── market_master_sales.csv
├── .env
├── requirements.txt
└── README.md

##  How to Run Locally

### 1. Clone the Repository

git clone https://github.com/AkshitDiwan25/Market-Master.git
cd Market-Master

### 2. Create Virtual Environment (Recommended)

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

### 3. Install dependencies

pip install -r requirements.txt

### 4. Set up environment variables

Create a .env file in the root directory:

GROQ_API_KEY=your_actual_groq_api_key

### 5. Run the application
gradio app.py


### Dataset

The  preloaded Dataset is a sample file which doesn't contain real data; it contains masked brands with columns in data being:

Date, SKU, Category, Region, Channel, Units_Sold, Revenue, Price

Insights are generated from sample aggregated revenue and units.

News module generates AI news based on model knowledge.