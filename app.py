import gradio as gr
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# 🔧 Chargement du modèle depuis Hugging Face
model_name = "Dvalinoc/gascon-translator-mt5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 📦 Détection de l'appareil
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 🧠 Fonction de traduction
def traduire_fr_gascon(texte):
    if not texte.strip():
        return "⚠️ Merci d’écrire quelque chose à traduire."
    prompt = f"traduction français vers gascon : {texte}"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,
        num_beams=4,
        early_stopping=True,
        decoder_start_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def traduire_gascon_fr(texte):
    if not texte.strip():
        return "⚠️ Mercés d’escriure quauquarren a revirar."
    prompt = f"traduction gascon vers français : {texte}"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,
        num_beams=4,
        early_stopping=True,
        decoder_start_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 🎨 Interface Gradio avec deux boutons
with gr.Blocks(title="Traducteur Gascon ↔ Français") as demo:
    gr.Markdown("## 🗣️ Traducteur Gascon ↔ Français")
    gr.Markdown("Modèle mT5 fine-tuné pour la traduction entre le français et le gascon.")
    
    with gr.Row():
        with gr.Column():
            texte_input = gr.Textbox(lines=3, label="Texte à traduire", placeholder="Ex : Je vais à la rivière / Que hè caud ?")
            bouton_fr_gascon = gr.Button("Français → Gascon")
            bouton_gascon_fr = gr.Button("Gascon → Français")
        with gr.Column():
            sortie = gr.Textbox(label="Traduction", show_copy_button=True)

    bouton_fr_gascon.click(fn=traduire_fr_gascon, inputs=texte_input, outputs=sortie)
    bouton_gascon_fr.click(fn=traduire_gascon_fr, inputs=texte_input, outputs=sortie)

demo.launch(server_name="0.0.0.0", server_port=7860)
