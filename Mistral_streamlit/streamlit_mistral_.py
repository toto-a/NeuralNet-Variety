import streamlit as st
import time
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from huggingface_hub import hf_hub_download

st.title("Mistral Bot")
(repo_id, model_file_name) = ("TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                                  "mistral-7b-instruct-v0.1.Q4_0.gguf")
model_path = hf_hub_download(repo_id=repo_id,
                             filename=model_file_name,
                            repo_type="model")




@st.cache_resource
def create_prompt_mistral(system_prompt):

    ##Set up the model
    llm = LlamaCpp(
            model_path=model_path,
            temperature=0,
            max_tokens=512,
            top_p=1,
            stop=["[INST]"],
            verbose=False,
            streaming=True,
    )
    
    template = """
    <s>[INST]{}[/INST]</s>
  
    [INST]{}[/INST]
    """.format(system_prompt, "{question}")

    # We create a prompt from the template so we can use it with Langchain
    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm_chain = prompt | llm  # LCEL

    return llm_chain


def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    system_prompt= st.text_area(
        label="System Prompt",
        value="You are a helpful AI assistant who answers questions in short sentences.",
        key="system_prompt"
    )



    
    if user_prompt := st.chat_input("Your message here", key="user_input"):
        st.session_state.messages.append({"role": "assistant", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Add user message to chat history
        llm_chain = create_prompt_mistral(system_prompt)
        full_response=llm_chain.invoke({"question": user_prompt})
        # Add assistant response to chat history
        with st.chat_message("assistant"):
            message_placeholder=st.markdown("▌")
            response_mistral=""

            for chunk in full_response.split() :
                response_mistral+=chunk + " "
                time.sleep(0.1)
                message_placeholder.markdown(response_mistral + "▌")
            message_placeholder.markdown(response_mistral)
    
        st.session_state.messages.append({"role": "assistant", "content": full_response})



if __name__=='__main__' :
    main()