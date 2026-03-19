# 💹 FinBot — Financial Data Chatbot (Gemini Edition)

A Python backend chatbot that analyzes large Excel/CSV datasets (1GB+) using Google Gemini AI.

## Architecture

```
financial-chatbot/
├── app.py              # FastAPI server (REST API)
├── chatbot.py          # AI engine (Gemini + pandas code generation)
├── requirements.txt    # Python dependencies
└── frontend/
    └── index.html      # Chat UI (served by FastAPI)
```

## Setup

### 1. Install Dependencies

```bash
cd financial-chatbot
pip install -r requirements.txt
```

### 2. Set Your Gemini API Key

```bash
# On Linux/Mac:
export GEMINI_API_KEY=your_api_key_here

# On Windows:
set GEMINI_API_KEY=your_api_key_here
```

Get your free API key from: https://aistudio.google.com/app/apikey

### 3. Run the Server

```bash
python app.py
```

Open your browser at: **http://localhost:8000**

---

## Gemini Model Options

In `chatbot.py`, you can change the model:

| Model | Speed | Quality | Cost |
|---|---|---|---|
| `gemini-1.5-flash` | ⚡ Fast | Good | Free tier available |
| `gemini-1.5-pro` | Slower | Best | Paid |
| `gemini-2.0-flash` | ⚡ Fastest | Very Good | Free tier available |

To switch model, edit this line in `chatbot.py`:
```python
self.model = genai.GenerativeModel(model_name="gemini-1.5-flash", ...)
```

---

## Usage

1. **Upload your file** — drag & drop `.xlsx`, `.xls`, or `.csv`
2. **Ask questions** in plain English:
   - "Total PO amount by supplier"
   - "Monthly spend trend"
   - "Top 10 cost centers by spend"
   - "Show duplicate purchase orders"
   - "Outliers in PO net amount"
   - "Average order value by country"

---

## Handling 1GB+ Files

For very large CSVs, use chunked reading:

```python
chunks = pd.read_csv(filepath, chunksize=100_000)
df = pd.concat(chunks, ignore_index=True)
```

Or use **DuckDB** for SQL on large files without loading all into RAM:

```bash
pip install duckdb
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /upload | Upload dataset file |
| POST | /chat | Send a chat message |
| GET | /info | Get dataset info |
| GET | / | Serve frontend UI |

A Python backend chatbot that can analyze large Excel/CSV datasets (1GB+) using AI-powered natural language queries.

## Architecture

```
financial-chatbot/
├── app.py              # FastAPI server (REST API)
├── chatbot.py          # AI engine (Claude + pandas code generation)
├── requirements.txt    # Python dependencies
└── frontend/
    └── index.html      # Chat UI (served by FastAPI)
```

## Setup

### 1. Install Dependencies

```bash
cd financial-chatbot
pip install -r requirements.txt
```

### 2. Set Your Anthropic API Key

```bash
# On Linux/Mac:
export ANTHROPIC_API_KEY=your_api_key_here

# On Windows:
set ANTHROPIC_API_KEY=your_api_key_here
```

Get your API key from: https://console.anthropic.com/

### 3. Run the Server

```bash
python app.py
```

The server starts at: **http://localhost:8000**

Open your browser and go to **http://localhost:8000** to use the chatbot.

---

## Usage

1. **Upload your file** — drag & drop or click to upload `.xlsx`, `.xls`, or `.csv`
2. **Ask questions** — type natural language questions like:
   - "Total PO amount by supplier"
   - "Monthly spend trend"
   - "Top 10 cost centers by spend"
   - "Show duplicate purchase orders"
   - "Outliers in PO net amount"
   - "Average order value by country"
   - "Compare spend: this year vs last year"

---

## Handling 1GB+ Files

For large files, add chunked loading in `app.py`:

```python
# For CSV files > 500MB, use chunked reading
chunks = pd.read_csv(filepath, chunksize=100_000)
df = pd.concat(chunks, ignore_index=True)
```

Or use **DuckDB** for SQL-based querying on large files without loading all into RAM:

```bash
pip install duckdb
```

```python
import duckdb
conn = duckdb.connect()
conn.execute("CREATE TABLE data AS SELECT * FROM read_xlsx('your_file.xlsx')")
result = conn.execute("SELECT SUM(PO_Net_Amount) FROM data GROUP BY Supplier_description").df()
```

---

## API Endpoints

| Method | Endpoint   | Description               |
|--------|------------|---------------------------|
| POST   | /upload    | Upload dataset file       |
| POST   | /chat      | Send a chat message       |
| GET    | /info      | Get dataset info          |
| GET    | /          | Serve frontend UI         |

---

## Example Chat Request (API)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Total spend by supplier country"}'
```

Response:
```json
{
  "answer": "Here is the total spend by supplier country...",
  "chart": {
    "type": "bar",
    "title": "Total Spend by Supplier Country",
    "x": ["India", "Germany", "USA"],
    "y": [5000000, 3200000, 1800000]
  },
  "table": null,
  "code": "result = df.groupby('Supplier_Country description')['PO_Net_Amount'].sum().sort_values(ascending=False)"
}
```