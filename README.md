# AI Data Analyst

AI Data Analyst is a production-ready Streamlit analytics workspace that combines data profiling, interactive visualizations, natural-language analysis, anomaly detection, voice input/output, and AI-assisted SQL/Pandas code generation.

The app is designed as a demo-ready mini analytics platform: upload a CSV or Excel file, inspect the dataset, ask questions in natural language, generate charts, detect outliers, and produce executable analysis code.

## Highlights

- Upload CSV and Excel datasets.
- Use the bundled sample dataset for instant demos.
- Explore dataset shape, schema, data quality, missing values, summary statistics, correlations, and column profiles.
- Ask AI questions with chat history, suggested prompts, selectable OpenAI model, and configurable reasoning effort.
- Use voice input and AI voice output for analyst-style conversations.
- Generate Plotly charts with chart recommendations.
- Detect anomalies with IQR and Z-score methods.
- Generate SQL and Pandas code from natural-language requests.
- Use global filters and presentation mode for clean demos.
- Run with OpenAI SDK support for GPT-5 class models through the Responses API.

## Tech Stack

- Python 3.10+
- Streamlit
- pandas
- openpyxl
- Plotly
- matplotlib and seaborn
- OpenAI Python SDK
- python-dotenv

## Project Structure

```text
ai_data_analyst/
  app.py
  data_loader.py
  analyzer.py
  ai_engine.py
  visualization.py
  anomaly_detector.py
  code_generator.py
  utils.py
  requirements.txt
  sample_data.csv
  voice_recorder/
    index.html
```

## Quick Start

```powershell
cd ai_data_analyst
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Environment Variables

Create `ai_data_analyst/.env` from `ai_data_analyst/.env.example`:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_SSL=insecure
OPENAI_MODEL=gpt-5.2
```

`OPENAI_SSL=insecure` is intentionally supported for environments where corporate SSL inspection breaks OpenAI requests. This disables certificate verification for OpenAI API calls, so use it only in trusted local or corporate environments where this is required.

## Running The App

```powershell
cd ai_data_analyst
streamlit run app.py
```

The sidebar lets you upload data, load the sample dataset, view readiness checks, apply global filters, select the AI model, enable voice options, and navigate between analysis pages.

## Main Capabilities

### Overview

- Dataset preview with search and density controls.
- KPI cards for rows, columns, missing cells, numeric columns, date columns, and duplicates.
- Missing-value summary.
- Column type and profile tables.
- Correlation matrix for numeric fields.

### Ask AI

- Chat-style data questions.
- Suggested analyst prompts.
- OpenAI model selection, including GPT-5.2.
- Reasoning effort selector.
- Browser voice input through the microphone.
- AI voice output through OpenAI text-to-speech.

### Visualizations

- Histogram.
- Bar chart.
- Aggregated bar chart.
- Line chart.
- Scatter plot.
- Box plot.
- Correlation heatmap.
- Automatic chart recommendations based on selected columns.

### Insights And Anomalies

- Automated deterministic insights for trends, top categories, outliers, and distributions.
- IQR and Z-score anomaly detection.
- Highlighted anomalous rows.
- Plain-language glossary for statistical terms.

### Code Generator

- Natural-language request input.
- AI-generated SQL query.
- Pandas equivalent code.
- Short explanation and assumptions.

### Presentation Mode

- Clean executive-style dashboard view.
- KPIs, key insights, and high-impact charts only.
- Useful for demos and interview walkthroughs.

## Voice Features

Voice input uses the browser microphone through a lightweight Streamlit component in `voice_recorder/index.html`. The recorded audio is transcribed with the configured OpenAI transcription model. Voice output uses OpenAI text-to-speech and returns playable audio inside the UI.

Your browser may ask for microphone permission the first time you use voice input.

## Security Notes

- Do not commit `.env` or Streamlit secrets.
- API keys are loaded from environment variables or `.env`.
- Uploaded datasets are processed locally by Streamlit unless you send context to OpenAI through the AI features.
- The app only sends compact schema, sample rows, and summary context to OpenAI to reduce token usage.

## Troubleshooting

If OpenAI calls fail with SSL certificate errors, confirm:

```env
OPENAI_SSL=insecure
```

If voice input does not start, confirm the page is loaded from `localhost` and microphone permission is allowed in the browser.

If Streamlit reports duplicate chart IDs, restart the app and ensure you are running the latest code. All Plotly charts in the app use explicit unique keys.

## License

This project is intended as a portfolio, interview, and internal demo application. Add a license file before publishing for public reuse.
