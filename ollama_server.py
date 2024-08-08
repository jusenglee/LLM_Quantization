from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nest_asyncio
from langchain_community.chat_models import ChatOllama
import uvicorn
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_teddynote.messages import stream_response

app = FastAPI()
nest_asyncio.apply()

# Logging 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_ids = {
    "kisti_koni": ChatOllama(model="koni:latest" ),
    "aya-23-8B-model": ChatOllama(model="aya:8b" ),
    "ko-gemma-2-9b-it": ChatOllama(model="gemma:latest" ),
    "llama-3-bllossom": ChatOllama(model="llama-ko-bllossom-server:latest" ),
}

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
    logger.info(f"Received request: {request}")

    if request.model not in model_ids:
        logger.error(f"Model not found: {request.model}")
        raise HTTPException(status_code=404, detail="Model not found")

    model = model_ids[request.model]

    prompt = ChatPromptTemplate.from_template(request.system + "{topic}")

    try:
        chain = prompt | model
        
        response = chain.invoke({"topic": request.inputData})
        logger.info(f"response: {response.content}")

        return response.content
        
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail="Error generating text")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
