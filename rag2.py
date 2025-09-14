import os
from RAG1 import retrieve
from predict_intent import predict_intent   # artık buradan alıyoruz
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_response(user_input: str):
    # 1. RAG1’den en yakın cevabı al
    soru, cevap, link, skor = retrieve(user_input)

    # 2. Intent tahmini yap
    intent = predict_intent(user_input)

    # 3. Gemini prompt
    system_prompt = f"""
    Sen bir belediye sohbet botusun.
    Kullanıcı sorusu: {soru}
    Bilgi tabanından en yakın cevap: {cevap}
    İlgili link: {link if link else "yok"}

    Intent: {intent}

    Kurallar:
    - intent 'acil_durum' ise sakinleştirici, güven verici bir dil kullan.
    - intent 'şikayet' ise "Mağduriyetiniz için özür dileriz..." benzeri empatik cevap ver.
    - intent 'teşekkür' ise kibar karşılık ver.
    - intent 'bilgi_sorma', 'oneri', 'randevu', 'islem_sorgu', 'destek' ise doğal, insansı cevap ver.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=system_prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )

    return response.text.strip()



