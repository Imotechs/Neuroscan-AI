import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com/v1/chat/completions"

SYSTEM_PROMPT = """You are a medical AI assistant specializing in brain MRI analysis. 
Given a classification result from a brain tumor detection model, provide a clear, 
clinically-relevant explanation. Include:
- What the diagnosis means 
- Which regions of the brain the model likely focused on (based on Grad-CAM)
- The confidence level and what it implies
- Suggested follow-up questions a radiologist might ask
Keep responses concise (2-3 paragraphs) and use plain language suitable for doctors."""


def build_context(prediction: str, confidence: float, all_confidence: dict) -> str:
    classes = sorted(all_confidence.items(), key=lambda x: -x[1])
    breakdown = "\n".join(f"  - {name}: {pct*100:.1f}%" for name, pct in classes)
    return (
        f"Classification result:\n"
        f"  Primary diagnosis: {prediction}\n"
        f"  Confidence: {confidence*100:.1f}%\n"
        f"  Full breakdown:\n{breakdown}"
    )


async def ask_deepseek(
    user_message: str,
    prediction: str,
    confidence: float,
    all_confidence: dict,
) -> str:
    if not API_KEY or API_KEY == "your-deepseek-api-key-here":
        return (
            "DeepSeek API key is not configured. "
            "Set DEEPSEEK_API_KEY in backend/.env to enable AI explanations."
        )

    context = build_context(prediction, confidence, all_confidence)

    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            BASE_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{context}\n\n{user_message}"},
                ],
                "max_tokens": 512,
                "temperature": 0.3,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
