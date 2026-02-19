# WageWatch India — Setup Guide

## Project Structure
```
wagegap_site/
├── backend/
│   ├── app.py              # Flask API
│   └── requirements.txt
└── frontend/
    └── index.html          # Single-file frontend (no build step)
```

## Backend Setup

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Run the API server
python app.py
# → Runs at http://localhost:5000
```

## Frontend Setup

No build step needed. Just open the HTML file:

```bash
# Option 1: Open directly in browser
open frontend/index.html

# Option 2: Serve with Python (avoids any CORS issues with local files)
cd frontend
python -m http.server 8080
# → Open http://localhost:8080
```

## API Endpoints

| Method | Endpoint       | Description                        |
|--------|---------------|------------------------------------|
| GET    | /api/meta     | Dropdown options (states, industries, education) |
| GET    | /api/stats    | Pre-computed dataset statistics    |
| GET    | /api/laws     | Labor law database                 |
| POST   | /api/analyze  | Analyze a single worker            |

### POST /api/analyze — Request Body
```json
{
  "gender":      "Female",
  "education":   "10th Pass",
  "experience":  8,
  "state":       "Uttar Pradesh",
  "industry":    "Textiles",
  "actual_wage": 12000
}
```

## Notes
- The frontend works fully offline (fallback logic built in) — no backend required to demo
- When the backend is running, it uses the full violation detection engine
- To integrate real ML model predictions, load the pickled models from `models/` in app.py
