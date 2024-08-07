from fastapi import FastAPI, Request
from pydantic import BaseModel
import transformers
import torch
import os
import nest_asyncio
app = FastAPI()
nest_asyncio.apply()
# Hugging Face API 키 설정
os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'Your HUGGINGFACEHUB_API_TOKEN'

model_ids = {
    "kisti_koni": "../model/kisti_koni",
    "aya-23-8B-model": "../model/aya-23-8B-model",
    "ko-gemma-2-9b-it": "../model/ko-gemma-2-9b-it",
    "llama-3-bllossom": "../model/llama-3-bllossom",
}

pipelines = {
    model_key: transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    ) for model_key, model_id in model_ids.items()
}

for pipeline in pipelines.values():
    pipeline.model.eval()

class QueryRequest(BaseModel):
    model: str
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    inputData: str
    system: str

@app.post("/generate")
async def generate_text(request: QueryRequest):
    if request.model not in pipelines:
        raise HTTPException(status_code=404, detail="Model not found")

    pipeline = pipelines[request.model]

    messages = [
        {"role": "system", "content": request.system},
        {"role": "user", "content": request.inputData}
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=request.max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty
    )

    return {"generated_text": outputs[0]["generated_text"][len(prompt):]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
