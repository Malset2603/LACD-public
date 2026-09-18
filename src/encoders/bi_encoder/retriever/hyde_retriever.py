import os
from openai import OpenAI
from src.utils.encoder.utils import BI_ENCODER_MAX_TOKEN_LENGTH
import torch
import numpy as np

hyde_prompt = lambda article: f"""
다음 법령과 충돌할 수 있는 법령을 찾아주세요.

법령:
{article}

충돌할 수 있는 법령:
"""


def hyde_retriever(model, laws_df, chroma_collection, article_to_check, top_k=500, batch_size = 8, tokenizer = None, llm_name = "gpt-4o-mini"):


    # OpenAI 모델 불러오기
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # llm 이 하나를 생성
    response = client.chat.completions.create(
        model=llm_name,
        messages=[
            {"role": "user", "content": hyde_prompt(article_to_check)}
        ],
        max_tokens=1000,
        temperature=0.0
    )

    generated_one = response.choices[0].message.content

    article_list = [article_to_check, generated_one]
    vector_list = []

    for a in article_list:
        inputs = tokenizer.encode_plus(
            a,
            add_special_tokens=True,
            max_length=BI_ENCODER_MAX_TOKEN_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(model.encoder.device)
        attention_mask = inputs["attention_mask"].to(model.encoder.device)

        with torch.no_grad():
            encoded_article = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = encoded_article.last_hidden_state[0, 0, :].cpu().numpy()

        vector_list.append(pooled_output)

    hyde_vector = np.mean(vector_list, axis=0)
    hyde_vector = hyde_vector.astype(np.float32)

    results = chroma_collection.query(
        query_embeddings=[hyde_vector.tolist()],
        n_results=top_k,
        include=["documents"]
    )
    hyde_top_k_articles = results["documents"][0]

    return hyde_vector, hyde_top_k_articles
    
