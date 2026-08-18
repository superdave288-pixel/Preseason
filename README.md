# NFL All-in-One Prop App — Preseason Edition

Includes Anytime TD, rushing yards, receiving yards, and passing yards.

Data modes:
- Blended (recommended)
- Preseason only
- Regular season only

Blended mode defaults to a 25% preseason weight. Preseason player production is aggregated from nflverse play-by-play when available. If a player has no preseason data yet, the app falls back to regular-season history.

Run with:

```bash
pip install -r requirements.txt
streamlit run app.py
```
