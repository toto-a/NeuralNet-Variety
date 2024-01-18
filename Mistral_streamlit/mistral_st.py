import streamlit as st 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer




model_name = "mistralai/Mistral-7B-Instruct-v0.1"
model=AutoModelForCausalLM.from_pretrained(model_name)
tokenizer=AutoTokenizer.from_pretrained(model_name)

@torch.no_grad()
def generate_text(prompt):
    model.eval()
    input_ids=tokenizer.encode(prompt,return_tensors="pt").to("cuda")
    ouput=model.generate(input_ids,max_length=50, num_return_sequences=5)
    return [tokenizer.decode(ids,skip_special_tokens=True) for ids in ouput]




def main():
    st.set_page_config(
        page_title="Mistral response", page_icon=":bird:")

   

    st.header("Mistral Response")
    user_input=st.text_input("Enter a prompt :")

    if user_input:
        st.write("Generating best practice message...")

        result = generate_text(user_input)

        st.info(result)


if __name__=="__main__" :
    main()




