# Always-On Memory Agent (Google ADK)

A lightweight, cost-effective background agent that continuously processes, consolidates, and serves memory. Built with **Google ADK** and **Gemini**, it runs 24/7 and supports multimodal ingestion (text, images, audio, video, PDFs).

## Features

- **Always-on background processing** — Watches an inbox directory and automatically ingests new files
- **Memory consolidation** — Periodically consolidates memories to reduce redundancy and surface connections
- **Multimodal ingestion** — Supports `.txt`, `.md`, `.json`, `.csv`, `.pdf`, images, audio, and video
- **REST API** — Query memories and ingest content via HTTP
- **Streamlit dashboard** — Visualize memories, run queries, and trigger operations
- **Adaptive agent** — Optional advanced agent with namespace scoping, episodic memory, procedural memory, and similarity retrieval

## Requirements

- Python 3.10+
- [Google AI API key](https://aistudio.google.com/apikey) (Gemini)

## Installation

```bash
git clone <repository-url>
cd always-on-memory-agent-google
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_api_key_here
# Optional:
# MODEL=gemini-3.1-flash-lite-preview
# MEMORY_DB=memory.db
```

## Quick Start

### Basic Agent

```bash
# Watch ./inbox, serve on :8888
python agent.py

# Custom watch directory and port
python agent.py --watch ./docs --port 9000

# Consolidate memories every 15 minutes
python agent.py --consolidate-every 15
```

### Adaptive Memory Agent (advanced)

```bash
# Watch ./inbox, serve on :8889
python adaptive_memory_agent.py

# Custom port
python adaptive_memory_agent.py --port 8890
```

### Dashboard

```bash
# Start the agent first, then:
streamlit run dashboard.py
```

## Usage

### Query memories

```bash
curl "http://localhost:8888/query?q=what+do+you+know"
```

### Ingest text

```bash
curl -X POST http://localhost:8888/ingest -d '{"text": "Some information to remember"}'
```

### Ingest via Adaptive Agent (with user context)

```bash
curl "http://localhost:8889/query?q=what+do+you+know&user_id=alice"
curl -X POST http://localhost:8889/ingest -d '{"text": "some info", "user_id": "alice"}'
```

### File ingestion

Drop files into the watched directory (default `./inbox`). Supported formats include text, markdown, JSON, CSV, PDF, images, and media files.

## Verify Gemini Connection

```bash
python test_gemini.py
```

## Project Structure

```
├── agent.py                 # Basic always-on memory agent
├── adaptive_memory_agent.py # Advanced agent (namespace, episodic, procedural memory)
├── dashboard.py             # Streamlit UI
├── test_gemini.py           # Gemini API connectivity test
├── requirements.txt
└── .env                     # API key (create locally, not committed)
```

## License

MIT License. See [LICENSE](LICENSE) for details.

Credits: Inspired by [always-on-memory-agent](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/gemini/agents/always-on-memory-agent)