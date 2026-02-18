import streamlit as st
from education_swarm import app, config

st.title("🎓 Multi-Agent Education System")

query = st.text_input("Enter a study topic")

if st.button("Generate Notes"):
    result = app.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config,
    )

    for m in result["messages"]:
        st.markdown(m.content)
