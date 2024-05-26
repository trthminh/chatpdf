import streamlit as st
from streamlit_chat import message
from ingestion import ingest_pdf
import tempfile

st.header("🤖 Minh's ChatPDF 🤖")
# Tạo sidebar ở phía trên trái
st.sidebar.title("🤖 Minh's ChatPDF 🤖")
st.sidebar.write("My chatbot, developed using GoogleGenerativeAI, Gemini, and Streamlit, features Streamlit-Chat, a straightforward component that offers a chat-app-like interface. This enhances the user experience, providing a visually appealing UI for the deployed chatbot on Streamlit.")

# Thêm thông tin của bạn vào sidebar
st.sidebar.title("My information")
st.sidebar.write("Github: https://github.com/trthminh")
st.sidebar.write("Linkedin: https://www.linkedin.com/in/trthminh/")


def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    return tmp_file_path

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_path = save_uploaded_file(uploaded_file)
    st.write(f"PDF file saved at: {pdf_path}")
    st.info("Loading pdf to vectorstore...")
    qa = ingest_pdf(pdf_path)
    st.success("Load pdf to vectorstore sucessfull")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here...")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if prompt:
    with st.spinner("Generating response..."):
        generated_response = qa.run(prompt)

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(generated_response)

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
        message(user_query, is_user=True)
        message(generated_response)
