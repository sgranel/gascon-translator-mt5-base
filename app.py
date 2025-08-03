import gradio as gr
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

model_name = "Dvalinoc/gascon-translator-mt5-base"
model = MT5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = MT5Tokenizer.from_pretrained(model_name)

def translate(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

iface = gr.Interface(fn=translate, inputs="text", outputs="text", title="Gascon Translator")
iface.launch(server_name="0.0.0.0", server_port=7860)

