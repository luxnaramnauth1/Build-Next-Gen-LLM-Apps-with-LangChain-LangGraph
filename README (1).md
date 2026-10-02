# llm_middleware_lab
    export OPENAI_API_KEY=sk-...        # omit to run in MOCK mode
    pip install -r requirements.txt
    uvicorn main:app --reload
    curl -X POST localhost:8000/api/chat -H "Content-Type: application/json" \
         -d '{"user_message":"Can I change my subscription plan next month?"}'
In MOCK mode add [sim-429], [sim-timeout], [sim-400] or [sim-down] to a message to simulate failures.
