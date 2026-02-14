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
Generate 4 concise retail market news updates for the category: {category}.
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
def chat_response(message, history, category):

    if history is None:
        history = []

    # Clean history (remove metadata)
    clean_history = []
    for msg in history:
        clean_history.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    conversation = [
        {"role": "system", "content": f"You are a retail strategy assistant for {category}."}
    ] + clean_history + [
        {"role": "user", "content": message}
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=conversation,
        temperature=0.7
    )

    reply = response.choices[0].message.content

    # Return history in Gradio format
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})

    return history


# -----------------------------
# MASTER PIPELINE
# -----------------------------
def run_full_assessment(dataset_choice, uploaded_file, category):

    # Load data
    df, status_msg = load_data(dataset_choice, uploaded_file)
    if df is None:
        return None, status_msg, "", "", gr.update(interactive=False)

    # Generate insights
    insights = generate_ai_insights(df, category)

    # Generate news
    news = generate_news(category)

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

    h1 { margin-bottom: 8px; }

    .subtitle {
        color: #6b7280;
        margin-bottom: 30px;
        font-size: 16px;
    }

    .top-row { margin-bottom: 25px; }

    .inline-radio .wrap {
        display: flex !important;
        gap: 20px;
    }

    .download-inline button {
        margin-left: 8px;
        font-size: 14px;
        padding: 6px 10px;
        border-radius: 6px;
    }
    """
) as demo:

    gr.Markdown("""
# AI Retail Market Intelligence
<div class="subtitle">
AI-powered diagnostics, executive insights & conversational analytics
</div>
""")

    # -----------------------------
    # TOP CONTROLS
    # -----------------------------
    with gr.Row(elem_classes="top-row"):

        category = gr.Dropdown(
            ["Running Shoes", "Casual Shoes", "Sports Apparel",
             "Backpacks", "Electronics Accessories"],
            value="Running Shoes",
            label="Retail Category"
        )

        with gr.Column():

            dataset_choice = gr.Radio(
                ["Use Preloaded Dataset", "Upload Your Own Dataset"],
                value="Use Preloaded Dataset",
                label="Dataset Source",
                elem_classes="inline-radio"
            )

            download_btn = gr.DownloadButton(
                "⬇ Excel",
                value=DEFAULT_FILE,
                visible=True,
                elem_classes="download-inline"
            )

            file_upload = gr.File(
                label="Upload CSV",
                visible=False
            )

    # Toggle Upload
    def toggle_upload(choice):
        return (
            gr.update(visible=(choice == "Upload Your Own Dataset")),
            gr.update(visible=(choice == "Use Preloaded Dataset"))
        )

    dataset_choice.change(
        toggle_upload,
        dataset_choice,
        outputs=[file_upload, download_btn]
    )

    # -----------------------------
    # GENERATE
    # -----------------------------
    submit_btn = gr.Button(
        "Generate Market Assessment",
        variant="primary"
    )

    status = gr.Markdown()
    df_state = gr.State()

    # -----------------------------
    # TABS
    # -----------------------------
    with gr.Tabs():

        with gr.Tab("AI Insights & Chat"):

            ai_summary = gr.Markdown("Click Generate to start analysis...")

            chatbot = gr.Chatbot(height=300, type='messages')
            user_input = gr.Textbox(
                placeholder="Ask about pricing, growth, SKU performance...",
                show_label=False,
                interactive=False
            )

            user_input.submit(
                chat_response,
                inputs=[user_input, chatbot, category],
                outputs=chatbot,
                show_progress=True
            )

        with gr.Tab("Latest Market News"):

            news_output = gr.Markdown("Click Generate to fetch latest updates...")

    # SINGLE MASTER PIPELINE CALL
    submit_btn.click(
        run_full_assessment,
        inputs=[dataset_choice, file_upload, category],
        outputs=[df_state, status, ai_summary, news_output, user_input],
        show_progress=True
    )

demo.launch()