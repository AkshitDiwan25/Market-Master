import gradio as gr
import pandas as pd
import os
from groq import Groq
from dotenv import load_dotenv

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DEFAULT_FILE = "market_master_sales.csv"


# -----------------------------
# DATA LOADING
# -----------------------------
def load_data(dataset_choice, uploaded_file):
    if dataset_choice == "Use Preloaded Dataset":
        if not os.path.exists(DEFAULT_FILE):
            return None, "Preloaded dataset not found."
        df = pd.read_csv(DEFAULT_FILE)
    else:
        if uploaded_file is None:
            return None, "Please upload a dataset."
        df = pd.read_csv(uploaded_file.name)

    return df, "Dataset loaded successfully."


# -----------------------------
# AI INSIGHTS
# -----------------------------
def generate_ai_insights(df, category):

    total_revenue = df["Revenue"].sum()
    total_units = df["Units_Sold"].sum()

    prompt = f"""
You are a senior retail strategy consultant.

Category: {category}
Total Revenue: {total_revenue}
Total Units Sold: {total_units}

Provide:
1. Executive summary (5 bullet points)
2. Strategic risks
3. Growth opportunities
4. Recommended actions

Keep it structured and professional.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a retail market intelligence expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content


# -----------------------------
# NEWS
# -----------------------------
def generate_news(category):
    prompt = f"""
Generate 10 concise retail market news updates for the category: {category}.
Each should include:
- A short headline
- 1 line explanation
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a retail market analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6
    )

    return response.choices[0].message.content


# -----------------------------
# CHATBOT
# -----------------------------
def chat_response(message, ui_history, llm_history, category):

    if ui_history is None:
        ui_history = []

    if llm_history is None:
        llm_history = []

    # Remove initial suggestion message
    if ui_history and "Try asking" in ui_history[0]["content"]:
        ui_history = []

    if not message.strip():
        return ui_history, llm_history, ""

    conversation = [
        {"role": "system", "content": f"You are a retail strategy assistant for {category}."}
    ] + llm_history + [
        {"role": "user", "content": message}
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=conversation,
        temperature=0.7
    )

    reply = response.choices[0].message.content

    llm_history = llm_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply}
    ]

    ui_history = ui_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply}
    ]

    return ui_history, llm_history, ""


# -----------------------------
# MASTER PIPELINE
# -----------------------------
def run_full_assessment(dataset_choice, uploaded_file, category):

    df, status_msg = load_data(dataset_choice, uploaded_file)

    if df is None:
        return None, status_msg, "", "", gr.update(interactive=False)

    status_msg = "Generating AI insights..."
    insights = generate_ai_insights(df, category)

    status_msg = "Fetching latest market news..."
    news = generate_news(category)

    status_msg = "Market assessment completed."

    return df, status_msg, insights, news, gr.update(interactive=True)


# -----------------------------
# UI
# -----------------------------
with gr.Blocks(
    theme=gr.themes.Base(),
    css="""
    body { background: #ffffff !important; }

    .gradio-container {
        width: 100% !important;
        padding-left: 60px;
        padding-right: 60px;
    }

    .card {
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        padding: 20px;
        background: #ffffff;
        height: 520px;
        display: flex;
        flex-direction: column;
    }

    .summary-content {
        overflow-y: auto;
        flex-grow: 1;
    }

    .chat-container {
        flex-grow: 1;
        overflow-y: auto;
    }

    .gr-chatbot {
        border: none !important;
        background: transparent !important;
    }

    .gr-chatbot .message.user {
        background: #f3f4f6 !important;
        border-radius: 12px !important;
    }

    .gr-chatbot .message.bot {
        background: #ffffff !important;
        border-radius: 12px !important;
    }
    """
) as demo:

    
    gr.Markdown("""
<div style="margin-bottom: 10px;">
  <h1 style="
    font-size: 44px;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 8px;
  ">
    Retail Intelligence
    <span style="
      background: linear-gradient(90deg, #2563eb, #9333ea);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    ">
      AI
    </span>
    Platform
  </h1>

  <div style="
    font-size: 17px;
    color: #6b7280;
    max-width: 720px;
  ">
    AI-powered diagnostics, executive insights & conversational analytics
  </div>
</div>
""")

    with gr.Row(equal_height=True):

        with gr.Column(scale=1):
            category = gr.Dropdown(
                ["Running Shoes", "Casual Shoes", "Sports Apparel",
                 "Backpacks", "Electronics Accessories"],
                value="Running Shoes",
                label="Retail Category"
            )

        with gr.Column(scale=1):

            dataset_choice = gr.Radio(
                ["Use Preloaded Dataset", "Upload Your Own Dataset"],
                value="Use Preloaded Dataset",
                label="Dataset Source"
            )

            download_btn = gr.DownloadButton(value=DEFAULT_FILE)
            file_upload = gr.File(label="Upload CSV", visible=False)

    def toggle_upload(choice):
        return (
            gr.update(visible=(choice == "Upload Your Own Dataset")),
            gr.update(visible=(choice == "Use Preloaded Dataset"))
        )

    dataset_choice.change(toggle_upload, dataset_choice, [file_upload, download_btn])

    submit_btn = gr.Button("Click to Generate AI Market Assessment and start the Chatbot", variant="primary")

    status = gr.Markdown()
    df_state = gr.State()

    with gr.Tabs():

        with gr.Tab("AI Insights & Chat"):

            with gr.Row():

                # LEFT — Executive Summary
                with gr.Column(scale=1):
                    with gr.Column(elem_classes="card"):
                        ai_summary = gr.Markdown(
                            "Click Generate to start analysis...",
                            elem_classes="summary-content"
                        )

                # RIGHT — Chat
                with gr.Column(scale=1):
                    with gr.Column(elem_classes="card"):

                        initial_suggestions = [{
                            "role": "assistant",
                            "content": """ **Try asking:**

• What are key growth drivers?  
• Are margins under pressure?  
• Which SKUs should we prioritize?  
• What risks should we monitor?  
"""
                        }]

                        chatbot = gr.Chatbot(
                            value=initial_suggestions,
                            height=360,
                            elem_classes="chat-container"
                        )

                        llm_history = gr.State([])

                        user_input = gr.Textbox(
                            placeholder="Ask about pricing, growth, SKU performance...",
                            show_label=False,
                            interactive=False
                        )

                        user_input.submit(
                            chat_response,
                            inputs=[user_input, chatbot, llm_history, category],
                            outputs=[chatbot, llm_history, user_input]
                        )

        with gr.Tab("Latest Market News (using AI)"):
            news_output = gr.Markdown(
                "Click Generate to fetch latest updates...",
                elem_classes="card"
            )

    submit_btn.click(
        run_full_assessment,
        inputs=[dataset_choice, file_upload, category],
        outputs=[df_state, status, ai_summary, news_output, user_input]
    )

demo.launch(share=True)