# hitl_feedback_lab
Backend (FastAPI, mock model unless OPENAI_API_KEY is set):
    cd backend && pip install -r requirements.txt && uvicorn main:app --port 8000
Frontend (React + Vercel AI SDK v5 `useChat`):
    cd frontend && npm install
    npm run dev        # http://localhost:5173 (proxies /api to :8000)
    npm run build      # then http://localhost:8000 serves frontend/dist from FastAPI
Routes: POST /api/chat (streaming), POST /api/feedback, GET /api/feedback/stats
