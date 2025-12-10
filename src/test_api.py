import os
import google.generativeai as genai

# 1) Assegura't que has fet abans a la terminal:
#    export GEMINI_API_KEY="LA_TEVA_CLAU"
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Falta la variable d'entorn GEMINI_API_KEY")

genai.configure(api_key=api_key)

# 2) TRIA EL MODEL NOU
model = genai.GenerativeModel("gemini-2.5-flash")
# Si vols provar el pro:
# model = genai.GenerativeModel("gemini-2.5-pro")

prompt = "Escriu una frase curta en català sobre un plat de cuina creativa."
resp = model.generate_content(prompt)

print("RESPOSTA:")
print(resp.text)


# mi clave: AIzaSyC4BYaVOuyotUP21golpE4xopl0Xk9OwT4 