# Resume Screening Classifier

Flask web app that classifies resumes into trained job categories using a TF-IDF vectorizer, a One-vs-Rest classifier, and a label encoder stored as pickle artifacts.

## What This Does

This project accepts pasted resume text or uploaded PDF, DOCX, and TXT files, extracts the resume content, cleans it with the same preprocessing style used during training, and predicts the most likely job category. The frontend also displays the top ranked labels from the classifier decision scores so the result is easier to inspect.

## Tech Stack

- Backend: Flask
- Machine learning: scikit-learn, TF-IDF, One-vs-Rest classifier
- File parsing: PyPDF2, python-docx
- Frontend: HTML, CSS, JavaScript

## Model Artifacts

| File | Purpose |
| --- | --- |
| `clf.pkl` | Trained resume category classifier |
| `tfidf.pkl` | Fitted text vectorizer |
| `encoder.pkl` | Label encoder for category names |
| `Resume Screening.csv` | Source dataset used in the notebook |

## Getting Started

```bash
cd "Resume Screening and  App"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`.

For Linux or macOS activation:

```bash
source .venv/bin/activate
```

## API

Health check:

```bash
GET /api/health
```

Predict from JSON text:

```bash
POST /api/predict
Content-Type: application/json

{
  "text": "Python machine learning pandas SQL dashboard..."
}
```

Predict from a file upload:

```bash
POST /api/predict
Form field: resume=<PDF, DOCX, or TXT file>
```

## How It Works

1. Text is extracted from the upload or read directly from the pasted input.
2. The resume text is normalized with URL, mention, punctuation, and non-ASCII cleanup.
3. The fitted TF-IDF vectorizer converts text into model features.
4. The classifier predicts the encoded label.
5. The label encoder converts the prediction into a readable category name.

## Deployment Notes

- Keep `clf.pkl`, `tfidf.pkl`, and `encoder.pkl` available in the project root.
- The included `requirements.txt` supports local Windows runs with Waitress and Linux deployments with Gunicorn.
- The root `.gitignore` intentionally does not ignore model artifacts. Use Git LFS or external storage if your hosting provider has file-size limits.

## Future Ideas

- Add confidence calibration using a classifier trained with probability estimates.
- Add batch resume screening with CSV export.
- Add a model card with training metrics from the notebook.
