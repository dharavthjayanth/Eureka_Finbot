"""
Financial Chatbot Backend — FastAPI + Pandas
Handles 1GB+ Excel/CSV datasets with AI-powered analysis
Powered by Claude API (Anthropic)
"""

import os
import json
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from chatbot import FinancialChatbot

app = FastAPI(title="FinBot — Financial Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

# Global state
state = {
    "df": None,
    "chatbot": None,
    "filename": None,
    "columns": [],
    "shape": None,
}


class ChatRequest(BaseModel):
    message: str

class FilterRequest(BaseModel):
    text: str

class ForecastRequest(BaseModel):
    periods: int = 6


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    index = frontend_path / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"status": "API running. No frontend found."}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and load an Excel or CSV file into memory."""
    try:
        filename = file.filename or "data"
        content = await file.read()

        if filename.endswith(".csv"):
            df = pd.read_csv(
                pd.io.common.BytesIO(content),
                low_memory=False,
                encoding="utf-8",
                encoding_errors="replace",
            )
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(pd.io.common.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Only .csv, .xlsx, .xls files supported.")

        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]

        # Auto-parse date columns
        for col in df.columns:
            if "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass

        # Auto-parse numeric columns
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    numeric = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
                    if numeric.notna().sum() > len(df) * 0.5:
                        df[col] = numeric
                except Exception:
                    pass

        state["df"] = df
        state["filename"] = filename
        state["columns"] = list(df.columns)
        state["shape"] = df.shape
        state["chatbot"] = FinancialChatbot(df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        return {
            "success": True,
            "filename": filename,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": list(df.columns),
            "numeric_columns": numeric_cols,
            "date_columns": date_cols,
            "categorical_columns": cat_cols,
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1_000_000, 2),
            "sample": df.head(3).fillna("").to_dict(orient="records"),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load file: {str(e)}")


@app.post("/chat")
async def chat(req: ChatRequest):
    if state["df"] is None or state["chatbot"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. Please upload a file first.")
    try:
        result = state["chatbot"].answer(req.message)
        return result
    except Exception as e:
        return {
            "answer": f"Sorry, I encountered an error: {str(e)}",
            "chart": None, "table": None,
            "error": traceback.format_exc(),
        }


@app.get("/info")
def dataset_info():
    if state["df"] is None:
        return {"loaded": False}
    df = state["df"]
    return {
        "loaded": True,
        "filename": state["filename"],
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1_000_000, 2),
    }


@app.post("/filter")
async def apply_filter(req: FilterRequest):
    if state["chatbot"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")
    result = state["chatbot"].apply_filter(req.text)
    state["df"] = state["chatbot"].df
    return result


@app.get("/kpis")
async def get_kpis():
    if state["chatbot"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")
    kpis = state["chatbot"].compute_kpis()
    return {"kpis": kpis, "filter_active": state["chatbot"].active_filter_desc}


@app.post("/forecast")
async def get_forecast(req: ForecastRequest):
    if state["chatbot"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")
    return state["chatbot"].get_forecast(periods=req.periods)


@app.get("/anomalies")
async def detect_anomalies():
    if state["chatbot"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")
    return state["chatbot"].detect_anomalies()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)