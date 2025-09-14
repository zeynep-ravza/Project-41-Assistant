import streamlit as st
from rag2 import generate_response  # senin RAG2 fonksiyonun

st.set_page_config(page_title="41 Asistan", page_icon="🤖")

# CSS
st.markdown("""
    <style>
    .title-text {
        color: #1E4A5F;
        font-size: 2.2em;
        font-weight: bold;
        margin: 0;
    }
    .subtitle-text {
        color: #1E4A5F;
        font-size: 1.1em;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Logo solda, başlık ve alt yazı sağda
col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo.png", width=80)
with col2:
    st.markdown("""
        <div>
            <h1 class="title-text">41 Asistan</h1>
            <div class="subtitle-text">Tüm sorularını yanıtlamak için buradayım!</div>
        </div>
    """, unsafe_allow_html=True)



# Sohbet geçmişini session_state ile sakla
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mesaj gönderme fonksiyonu
def send_message():
    user_input = st.session_state.user_input
    if user_input.lower() not in ["q", "çık", "exit", "quit"]:
        bot_response = generate_response(user_input)
        st.session_state.messages.append({"user": user_input, "bot": bot_response})
    st.session_state.user_input = ""

# Text input
st.text_input("Mesajınızı girin:", key="user_input", on_change=send_message)

# CSS ile chat balonları stili
st.markdown("""
<style>
.user-msg {
    background-color: #2C9FA3;
    color: white;
    padding: 10px;
    border-radius: 15px;
    margin: 5px 0;
    text-align: right;
    font-weight: 500;
    box-shadow: 1px 2px 5px rgba(0,0,0,0.2);
}
.bot-msg {
    background-color: #1E4A5F;
    color: white;
    padding: 10px;
    border-radius: 15px;
    margin: 5px 0;
    text-align: left;
    font-weight: 500;
    box-shadow: 1px 2px 5px rgba(0,0,0,0.2);
}
.msg-label {
    font-weight: bold;
    margin-bottom: 3px;
}
</style>
""", unsafe_allow_html=True)

# Mesajları ters sırayla göster
for msg in reversed(st.session_state.messages):
    st.markdown(f"<div class='msg-label'>👤 Kullanıcı:</div><div class='user-msg'>{msg['user']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='msg-label'>🤖 Bot:</div><div class='bot-msg'>{msg['bot']}</div>", unsafe_allow_html=True)
