import streamlit as st 
import backend as demo  # Replace with your backend filename

st.set_page_config(page_title="HR Q & A with RAG")  # Modify Heading

new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">HR Q & A with RAG 🎯</p>'
st.markdown(new_title, unsafe_allow_html=True)  # Modify Title

# Input for PDF URL
pdf_url = st.text_input("Enter PDF URL", "")
process_button = st.button("📥 Process PDF")

if process_button and pdf_url:
    with st.spinner("📀 Processing the document... Please wait!"):
        st.session_state.vector_index = demo.HRPolicyIndexer(pdf_url, "your_openai_api_key_here").create_index()
        st.success("✅ Document processed successfully! Now, ask your questions.")

# Search bar for questions
if 'vector_index' in st.session_state:
    input_text = st.text_input("Ask a question about the document")
    search_button = st.button("🔍 Search")
    
    if search_button and input_text:
        with st.spinner("🤖 Generating answer..."):
            hr_rag = demo.HRRAG(st.session_state.vector_index, "your_openai_api_key_here")
            response_content = hr_rag.get_response(input_text)
            st.write(response_content)