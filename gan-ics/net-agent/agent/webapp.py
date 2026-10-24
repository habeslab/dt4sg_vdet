from __future__ import annotations
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from agent.dashboard_state import get_last, get_explanations

templates = Jinja2Templates(directory="templates")

app = FastAPI(title="net-agent IDS dashboard")

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/last")
def api_last():
    return JSONResponse(get_last())

@app.get("/api/explanations")
def api_explanations(limit: int = 20):
    return JSONResponse(get_explanations(limit))

@app.get("/health")
def health():
    return {"ok": True}
