"""
PolicyLens - National Environment Policy Analyser
Sri Lanka Ministry of Environment

Requirements:
    pip install flask PyPDF2 numpy scikit-learn

Run:
    python app.py
    Then open http://127.0.0.1:5000
"""

from flask import Flask, render_template_string, request, jsonify
import PyPDF2
import io
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# =============================================================
#  TEXT EXTRACTION
# =============================================================

def read_uploaded_bytes(file_bytes, filename):
    ext = filename.lower().rsplit('.', 1)[-1]
    if ext == "pdf":
        text = ""
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        except Exception as e:
            return ""
        return text
    else:
        return file_bytes.decode("utf-8", errors="ignore")


# =============================================================
#  TEXT CLEANING
# =============================================================

def clean_text(text):
    # Remove TOC lines: "Something .... 12"
    text = re.sub(r'[A-Za-z ,\-&/]+\.{2,}\s*\d+', '', text)
    # Remove standalone page numbers
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
    # Remove policy numbering like 4.1.1.1. at line starts
    text = re.sub(r'^\s*\d+(\.\d+){1,4}\.?\s*', '', text, flags=re.MULTILINE)
    # Remove ISBN / print lines
    text = re.sub(r'ISBN[^\n]+', '', text)
    text = re.sub(r'Printed by[^\n]+', '', text, flags=re.IGNORECASE)
    # Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# =============================================================
#  TEXTRANK SENTENCE SCORING
# =============================================================

def textrank_scores(sentences):
    if len(sentences) < 2:
        return np.ones(len(sentences))
    try:
        vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=8000)
        tfidf = vec.fit_transform(sentences)
    except Exception:
        return np.ones(len(sentences))

    sim = cosine_similarity(tfidf)
    np.fill_diagonal(sim, 0)
    row_sums = sim.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    sim = sim / row_sums

    damping = 0.85
    scores = np.ones(len(sentences)) / len(sentences)
    for _ in range(60):
        new_s = (1 - damping) / len(sentences) + damping * sim.T.dot(scores)
        if np.linalg.norm(new_s - scores) < 1e-6:
            break
        scores = new_s
    return scores


# =============================================================
#  CATEGORY KEYWORDS  (tuned for environment policy documents)
# =============================================================

GOAL_WORDS = [
    "goal", "objective", "aim", "vision", "mission",
    "will be in place", "will be raised", "will be developed",
    "will be mainstreamed", "will be fulfilled", "will be laid",
    "will be placed", "seeks to", "designed to", "committed to",
    "ensure", "promote", "strengthen", "improve", "foster",
    "establish", "increase", "reduce", "create", "inclusive",
    "sustainable", "aspire", "achieve", "protect", "conserve",
    "safeguard", "develop", "enhance", "positioned", "founded"
]

STRAT_WORDS = [
    "strateg", "action plan", "reform", "implement", "framework",
    "programme", "initiative", "approach", "invest", "fund",
    "partnership", "coordinat", "regulat", "monitor", "review",
    "assess", "training", "capacity", "digital", "technolog",
    "collaborat", "mechanism", "procedure", "allocat", "launch",
    "pilot", "introducing", "establishing", "developing",
    "measures will be", "will be introduced", "will be established",
    "will be implemented", "payment for ecosystem", "market-based",
    "early warning", "restoration fund", "management plan",
    "national spatial", "integrated system", "legal framework",
    "public-private", "awareness campaign", "mobile app"
]

DIR_WORDS = [
    "overall", "overarching", "long-term", "transform",
    "national policy", "national environment", "economic development",
    "sustainable development", "sustainability", "priority",
    "strategic", "commitment", "policy framework", "vision for",
    "future", "decade", "roadmap", "low-carbon", "climate-resilient",
    "green economy", "ecological", "biodiversity hotspot",
    "natural capital", "ecosystem services", "blue economy",
    "planetary boundaries", "multilateral", "SDG", "NEP",
    "environment policy", "key policy", "main purpose"
]


def cat_score(sentence, words):
    low = sentence.lower()
    return sum(1 for w in words if w in low)


# =============================================================
#  MAIN SUMMARISE FUNCTION
# =============================================================

def summarize(text):
    raw = re.sub(r'\n', ' ', text)
    sents = re.split(r'(?<=[.!?])\s+', raw)
    sents = [
        s.strip() for s in sents
        if 55 < len(s.strip()) < 380
        and not re.match(r'^\s*\d+[\.\)]\s', s.strip())
        and not re.match(r'^(Annex|Figure|Table|ISBN|Printed|Published|Ex:|Ministry of)', s.strip())
    ]

    if not sents:
        return [], [], []

    tr = textrank_scores(sents)

    scored = [
        {
            "s": s,
            "tr": tr[i],
            "g":  cat_score(s, GOAL_WORDS)  * tr[i],
            "st": cat_score(s, STRAT_WORDS) * tr[i],
            "d":  cat_score(s, DIR_WORDS)   * tr[i],
            "i":  i,
        }
        for i, s in enumerate(sents)
    ]

    used = set()

    def pick(key, n):
        ranked = sorted(scored, key=lambda x: x[key], reverse=True)
        out = []
        for item in ranked:
            if item["i"] not in used and item[key] > 0:
                out.append(item["s"])
                used.add(item["i"])
                if len(out) == n:
                    break
        return out

    goals  = pick("g",  5)
    strats = pick("st", 5)
    dirs   = pick("d",  3)

    fb = [x["s"] for x in sorted(scored, key=lambda x: x["tr"], reverse=True) if x["i"] not in used]
    if not goals:  goals  = fb[:4]
    if not strats: strats = fb[4:8]
    if not dirs:   dirs   = fb[8:11]

    def order(lst):
        return sorted(lst, key=lambda s: next((x["i"] for x in scored if x["s"] == s), 9999))

    return order(goals), order(strats), order(dirs)


# =============================================================
#  SCENARIOS  (Sri Lanka NEP-specific)
# =============================================================

SCENARIOS = {
    "Climate-Induced Disaster Recovery": {
        "icon": "🌊",
        "color": "#1565C0",
        "light": "#E3F2FD",
        "desc": (
            "Sri Lanka ranked 2nd in the Global Climate Risk Index (2018). "
            "This scenario adapts the NEP to prioritise rapid recovery and resilience "
            "after floods, landslides, and cyclones."
        ),
        "goals": [
            "Integrate climate risk management and disaster preparedness into all environmental planning at national and sub-national levels.",
            "Ensure rapid restoration of degraded ecosystems and critical infrastructure services after climate-induced disasters.",
            "Protect vulnerable coastal and inland communities through equitable resource allocation and social safety nets.",
        ],
        "strategies": [
            "Establish a national disaster-environment coordination unit under the Ministry of Environment with cross-ministry mandate.",
            "Implement ecosystem-based disaster risk reduction including mangrove and wetland restoration for coastal and flood protection.",
            "Introduce an emergency rehabilitation fund for rapid restoration of degraded land, water resources, and critical habitats.",
            "Scale up community-based early warning systems and climate literacy programmes across all nine provinces.",
            "Mainstream Climate Change Impact Assessments (CCIA) in EIA and SEA procedures for all new investment projects.",
        ],
        "direction": (
            "Policy shifts toward climate resilience and recovery. Environmental governance becomes the "
            "first line of defence — integrating disaster preparedness with long-term ecosystem restoration "
            "in alignment with Sri Lanka's NDC commitments under the UNFCCC and Paris Agreement."
        ),
    },
    "Youth & Digital Innovation": {
        "icon": "💻",
        "color": "#6A1B9A",
        "light": "#F3E5F5",
        "desc": (
            "Leverages Sri Lanka's young population and growing digital infrastructure to transform "
            "environmental governance through technology, citizen science, and youth-driven innovation."
        ),
        "goals": [
            "Position Sri Lanka as a regional leader in digital environmental monitoring and youth-driven conservation innovation.",
            "Empower youth and communities to actively participate in policy implementation through accessible digital platforms.",
            "Build a transparent, data-driven system for environmental reporting using real-time public participation.",
        ],
        "strategies": [
            "Launch a national mobile application for real-time citizen reporting of pollution, illegal deforestation, sand mining, and biodiversity threats.",
            "Establish green innovation hubs at universities and technology parks linking eco-startups with the Ministry of Environment.",
            "Integrate updated environmental content into national digital learning platforms for all school and university levels.",
            "Create a dedicated youth environment fund with annual grants and national recognition for youth-led conservation projects.",
            "Digitalise the MRV (Measurement, Reporting and Verification) system for GHG emissions with open public data access.",
        ],
        "direction": (
            "Policy pivots toward digital transformation and youth empowerment — making environmental "
            "governance participatory, transparent, and future-ready. Technology becomes the bridge "
            "between policy and grassroots action aligned with Sri Lanka's SDG commitments."
        ),
    },
    "Sustainability & Premium Green Economy": {
        "icon": "🌿",
        "color": "#2E7D32",
        "light": "#E8F5E9",
        "desc": (
            "Sri Lanka is one of 36 global biodiversity hotspots. This scenario positions the country "
            "as a premium green economy — maximising the value of natural capital while minimising "
            "ecological footprint."
        ),
        "goals": [
            "Position Sri Lanka as a premier global destination for sustainable eco-tourism, green agriculture, and nature-based services.",
            "Limit resource exploitation volume while maximising quality, sustainability, and ecosystem integrity across all sectors.",
            "Establish Sri Lanka's brand as a biodiversity hotspot with internationally recognised premium conservation standards.",
        ],
        "strategies": [
            "Introduce a national sustainability levy on extractive industries to fund conservation programmes and green innovation.",
            "Cap resource extraction in sensitive ecosystems through mandatory, independent environmental impact assessments.",
            "Partner with global eco-certification bodies and international donors to develop premium sustainable products and services.",
            "Mainstream economic valuation of ecosystem services in all national investment, budget, and resource allocation decisions.",
            "Develop payment for ecosystem services (PES) schemes scaled from successful pilots to all critical watersheds and marine zones.",
        ],
        "direction": (
            "Policy shifts decisively from volume-driven growth to high-value, low-impact sustainability. "
            "Sri Lanka's rich natural capital — from rainforests to coral reefs — becomes the engine of "
            "a premium green economy, setting a model for sustainable island development globally."
        ),
    },
}


# =============================================================
#  BLEND EXTRACTED POLICY WITH SCENARIO
# =============================================================

def blend(policy_goals, policy_strats, policy_dirs, scenario):
    def merge(policy_list, template_list, n=5):
        seen, merged = set(), []
        for item in policy_list[:3] + template_list:
            key = item.lower()[:80]
            if key not in seen:
                seen.add(key)
                merged.append(item)
            if len(merged) == n:
                break
        return merged

    m_goals  = merge(policy_goals,  scenario["goals"],      5)
    m_strats = merge(policy_strats, scenario["strategies"], 5)

    policy_dir_text = " ".join(policy_dirs[:2]).strip()
    if policy_dir_text:
        m_dir = (
            "ORIGINAL POLICY DIRECTION:\n" + policy_dir_text +
            "\n\nSCENARIO ADAPTATION:\n" + scenario["direction"]
        )
    else:
        m_dir = scenario["direction"]

    return m_goals, m_strats, m_dir


# =============================================================
#  ROUTES
# =============================================================

@app.route("/")
def index():
    return render_template_string(HTML, scenarios=SCENARIOS)


@app.route("/summarise", methods=["POST"])
def do_summarise():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded."}), 400
    raw  = read_uploaded_bytes(file.read(), file.filename)
    if not raw or len(raw.strip()) < 100:
        return jsonify({"error": "Could not extract text. Check the file format."}), 400
    text = clean_text(raw)
    goals, strats, dirs = summarize(text)
    return jsonify({"goals": goals, "strategies": strats, "direction": dirs})


@app.route("/generate", methods=["POST"])
def do_generate():
    data     = request.json or {}
    scenario = SCENARIOS.get(data.get("scenario"))
    if not scenario:
        return jsonify({"error": "Invalid scenario selected."}), 400
    m_goals, m_strats, m_dir = blend(
        data.get("goals", []),
        data.get("strategies", []),
        data.get("direction", []),
        scenario
    )
    draft  = "GOALS:\n" + "\n".join(f"  • {g}" for g in m_goals)
    draft += "\n\nSTRATEGIES:\n" + "\n".join(f"  • {s}" for s in m_strats)
    draft += f"\n\nDIRECTION:\n  {m_dir}"
    return jsonify({"draft": draft})


# =============================================================
#  HTML  (single-page app — all CSS and JS inline)
# =============================================================

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PolicyLens — Sri Lanka NEP Analyser</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+3:wght@300;400;600&display=swap" rel="stylesheet">
<style>
/* ── Reset & Base ─────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:       #F4F7F2;
  --surface:  #FFFFFF;
  --border:   #D6E4D2;
  --green-dk: #1B4332;
  --green-md: #2D6A4F;
  --green-lt: #52B788;
  --green-xs: #D8F3DC;
  --text:     #1C2B1E;
  --muted:    #607060;
  --danger:   #C62828;
  --radius:   12px;
  --shadow:   0 4px 20px rgba(27,67,50,.10);
}

html { scroll-behavior: smooth; }

body {
  font-family: 'Source Sans 3', sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  font-size: 15px;
  line-height: 1.6;
}

/* ── Header ─────────────────────────────────────────────── */
.header {
  background: linear-gradient(135deg, #0D2B1A 0%, #1B4332 45%, #2D6A4F 100%);
  color: #fff;
  padding: 0 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 72px;
  box-shadow: 0 2px 12px rgba(0,0,0,.30);
  position: sticky;
  top: 0;
  z-index: 100;
}

.header-brand {
  display: flex;
  align-items: center;
  gap: .8rem;
}

.header-logo {
  width: 42px;
  height: 42px;
  background: rgba(255,255,255,.15);
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.4rem;
}

.header-title {
  font-family: 'Playfair Display', serif;
  font-size: 1.35rem;
  letter-spacing: .3px;
}

.header-sub {
  font-size: .78rem;
  opacity: .72;
  margin-top: 1px;
  font-weight: 300;
}

.header-badge {
  background: rgba(255,255,255,.12);
  border: 1px solid rgba(255,255,255,.22);
  border-radius: 20px;
  padding: .35rem .9rem;
  font-size: .78rem;
  letter-spacing: .4px;
  text-transform: uppercase;
  font-weight: 600;
}

/* ── Workflow Steps Bar ──────────────────────────────────── */
.steps-bar {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: .75rem 2rem;
  display: flex;
  align-items: center;
  gap: 0;
  overflow-x: auto;
}

.step {
  display: flex;
  align-items: center;
  gap: .55rem;
  padding: .4rem .9rem;
  border-radius: 20px;
  font-size: .82rem;
  font-weight: 600;
  color: var(--muted);
  white-space: nowrap;
  transition: all .25s;
}
.step.active {
  background: var(--green-xs);
  color: var(--green-dk);
}
.step.done {
  color: var(--green-md);
}
.step-num {
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: var(--border);
  color: var(--muted);
  font-size: .75rem;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  transition: all .25s;
}
.step.active .step-num { background: var(--green-md); color: #fff; }
.step.done  .step-num  { background: var(--green-lt); color: #fff; }

.step-arrow {
  color: var(--border);
  font-size: 1rem;
  padding: 0 .25rem;
}

/* ── Layout ─────────────────────────────────────────────── */
.layout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.25rem;
  padding: 1.4rem 1.8rem;
  max-width: 1440px;
  margin: 0 auto;
}

@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; padding: 1rem; }
}

/* ── Panel ───────────────────────────────────────────────── */
.panel {
  background: var(--surface);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  overflow: hidden;
}

.panel-header {
  padding: 1rem 1.4rem .85rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: .6rem;
}

.panel-icon {
  width: 34px;
  height: 34px;
  border-radius: 8px;
  background: var(--green-xs);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.1rem;
  flex-shrink: 0;
}

.panel-title {
  font-family: 'Playfair Display', serif;
  font-size: 1.05rem;
  color: var(--green-dk);
  line-height: 1.2;
}

.panel-sub {
  font-size: .78rem;
  color: var(--muted);
  margin-top: 1px;
}

.panel-body { padding: 1.2rem 1.4rem; }

/* ── Upload Zone ─────────────────────────────────────────── */
.upload-zone {
  border: 2px dashed var(--green-lt);
  border-radius: 10px;
  padding: 1.6rem 1rem;
  text-align: center;
  background: #FAFDF9;
  cursor: pointer;
  transition: border-color .2s, background .2s, transform .15s;
  position: relative;
}
.upload-zone:hover {
  border-color: var(--green-md);
  background: var(--green-xs);
  transform: translateY(-1px);
}
.upload-zone input { display: none; }
.upload-zone .uz-icon { font-size: 2.2rem; line-height: 1; }
.upload-zone .uz-text {
  font-size: .85rem;
  color: var(--muted);
  margin-top: .5rem;
}
.upload-zone .uz-hint {
  font-size: .75rem;
  color: #AAB8AA;
  margin-top: .25rem;
}
.uz-filename {
  margin-top: .75rem;
  padding: .45rem .9rem;
  background: var(--green-xs);
  border-radius: 6px;
  font-size: .82rem;
  font-weight: 600;
  color: var(--green-dk);
  display: none;
  word-break: break-all;
}

/* ── Buttons ─────────────────────────────────────────────── */
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: .5rem;
  width: 100%;
  padding: .7rem 1.4rem;
  border: none;
  border-radius: 8px;
  font-family: 'Source Sans 3', sans-serif;
  font-size: .92rem;
  font-weight: 600;
  cursor: pointer;
  transition: all .2s;
  position: relative;
  overflow: hidden;
}

.btn-primary {
  background: linear-gradient(135deg, var(--green-md), var(--green-dk));
  color: #fff;
  box-shadow: 0 3px 12px rgba(27,67,50,.25);
}
.btn-primary:hover {
  background: linear-gradient(135deg, var(--green-dk), #0D2B1A);
  box-shadow: 0 5px 18px rgba(27,67,50,.35);
  transform: translateY(-1px);
}
.btn-primary:active { transform: translateY(0); }
.btn-primary:disabled { opacity: .55; cursor: not-allowed; transform: none; }

.btn-secondary {
  background: var(--green-xs);
  color: var(--green-dk);
  border: 1px solid var(--green-lt);
}
.btn-secondary:hover { background: #C7EACC; }

.btn .spin {
  display: inline-block;
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255,255,255,.35);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin .75s linear infinite;
  display: none;
}
.btn.loading .spin    { display: inline-block; }
.btn.loading .btn-lbl { opacity: .75; }
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Section Labels ──────────────────────────────────────── */
.sec-label {
  font-size: .75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .8px;
  color: var(--green-md);
  margin: 1.1rem 0 .4rem;
  display: flex;
  align-items: center;
  gap: .4rem;
}
.sec-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* ── Output Boxes ────────────────────────────────────────── */
.out-box {
  background: #FAFDF9;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: .9rem 1rem;
  min-height: 72px;
  font-size: .87rem;
  line-height: 1.75;
  white-space: pre-wrap;
  color: #334433;
  transition: border-color .2s;
}
.out-box:focus-within { border-color: var(--green-lt); }
.out-box.has-content { border-color: var(--green-lt); background: #F6FBF6; }

.out-placeholder {
  color: #AAB8AA;
  font-style: italic;
  font-size: .85rem;
}

/* ── Scenario Cards ──────────────────────────────────────── */
.scenario-grid {
  display: flex;
  flex-direction: column;
  gap: .7rem;
}

.sc-card {
  border: 2px solid var(--border);
  border-radius: 10px;
  padding: .95rem 1rem;
  cursor: pointer;
  transition: all .22s;
  position: relative;
  overflow: hidden;
}
.sc-card::before {
  content: '';
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 4px;
  background: transparent;
  transition: background .22s;
}
.sc-card:hover {
  border-color: var(--green-lt);
  box-shadow: 0 4px 14px rgba(82,183,136,.15);
  transform: translateX(2px);
}
.sc-card:hover::before   { background: var(--green-lt); }
.sc-card.selected         { border-color: var(--green-md); box-shadow: 0 4px 18px rgba(45,106,79,.18); }
.sc-card.selected::before { background: var(--green-md); }

.sc-card[data-id="Climate-Induced Disaster Recovery"].selected         { border-color:#1565C0; }
.sc-card[data-id="Climate-Induced Disaster Recovery"].selected::before { background:#1565C0; }
.sc-card[data-id="Youth & Digital Innovation"].selected                { border-color:#6A1B9A; }
.sc-card[data-id="Youth & Digital Innovation"].selected::before        { background:#6A1B9A; }

.sc-head {
  display: flex;
  align-items: center;
  gap: .55rem;
}
.sc-icon {
  font-size: 1.3rem;
  width: 36px;
  height: 36px;
  border-radius: 8px;
  background: var(--green-xs);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.sc-title {
  font-weight: 600;
  font-size: .95rem;
  color: var(--green-dk);
}
.sc-desc {
  font-size: .80rem;
  color: var(--muted);
  margin-top: .4rem;
  line-height: 1.55;
  padding-left: .2rem;
}
.sc-check {
  margin-left: auto;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  border: 2px solid var(--border);
  flex-shrink: 0;
  transition: all .2s;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: .7rem;
  color: transparent;
}
.sc-card.selected .sc-check {
  background: var(--green-md);
  border-color: var(--green-md);
  color: #fff;
}

/* ── Draft box ───────────────────────────────────────────── */
#draft-box {
  font-size: .86rem;
  line-height: 1.85;
  min-height: 160px;
}

/* ── Toast ───────────────────────────────────────────────── */
.toast {
  position: fixed;
  bottom: 1.5rem;
  right: 1.5rem;
  padding: .7rem 1.2rem;
  border-radius: 8px;
  font-size: .85rem;
  font-weight: 600;
  color: #fff;
  background: var(--green-dk);
  box-shadow: 0 4px 20px rgba(0,0,0,.25);
  transform: translateY(80px);
  opacity: 0;
  transition: all .3s cubic-bezier(.34,1.56,.64,1);
  pointer-events: none;
  z-index: 999;
  max-width: 320px;
}
.toast.show { transform: translateY(0); opacity: 1; }
.toast.error { background: var(--danger); }

/* ── Divider ─────────────────────────────────────────────── */
.divider {
  height: 1px;
  background: var(--border);
  margin: 1.1rem 0;
}

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
</style>
</head>
<body>

<!-- ═══════════════════ HEADER ═══════════════════ -->
<header class="header">
  <div class="header-brand">
    <div class="header-logo">🌿</div>
    <div>
      <div class="header-title">PolicyLens</div>
      <div class="header-sub">National Environment Policy Analyser · Sri Lanka</div>
    </div>
  </div>
  <div class="header-badge">Ministry of Environment · 2022</div>
</header>

<!-- ═══════════════════ WORKFLOW STEPS ═══════════════════ -->
<div class="steps-bar" id="steps-bar">
  <div class="step active" id="step1">
    <div class="step-num">1</div> Upload Policy
  </div>
  <div class="step-arrow">›</div>
  <div class="step" id="step2">
    <div class="step-num">2</div> Summarise
  </div>
  <div class="step-arrow">›</div>
  <div class="step" id="step3">
    <div class="step-num">3</div> Choose Scenario
  </div>
  <div class="step-arrow">›</div>
  <div class="step" id="step4">
    <div class="step-num">4</div> Generate Draft
  </div>
</div>

<!-- ═══════════════════ MAIN LAYOUT ═══════════════════ -->
<main class="layout">

  <!-- ───────────── LEFT PANEL: Upload + Summary ───────────── -->
  <div class="panel">
    <div class="panel-header">
      <div class="panel-icon">📄</div>
      <div>
        <div class="panel-title">Policy Document</div>
        <div class="panel-sub">Upload a PDF or TXT policy file to analyse</div>
      </div>
    </div>
    <div class="panel-body">

      <!-- Upload zone -->
      <div class="upload-zone" onclick="document.getElementById('file-input').click()" id="upload-zone">
        <input type="file" id="file-input" accept=".pdf,.txt" onchange="onFileSelect(this)">
        <div class="uz-icon">📁</div>
        <div class="uz-text">Click here to upload your policy document</div>
        <div class="uz-hint">Supported formats: PDF, TXT · Max recommended: 10 MB</div>
        <div class="uz-filename" id="uz-filename"></div>
      </div>

      <div style="margin-top:1rem;">
        <button class="btn btn-primary" id="btn-summarise" onclick="doSummarise()" disabled>
          <span class="spin"></span>
          <span class="btn-lbl">🔍 Summarise Policy Document</span>
        </button>
      </div>

      <div class="divider"></div>

      <!-- Goals -->
      <div class="sec-label">🎯 Policy Goals</div>
      <div class="out-box" id="box-goals">
        <span class="out-placeholder">Extracted policy goals will appear here after summarising.</span>
      </div>

      <!-- Strategies -->
      <div class="sec-label">⚙️ Key Strategies</div>
      <div class="out-box" id="box-strats">
        <span class="out-placeholder">Extracted strategies and action mechanisms will appear here.</span>
      </div>

      <!-- Direction -->
      <div class="sec-label">🧭 Overall Direction</div>
      <div class="out-box" id="box-dir">
        <span class="out-placeholder">The overarching policy direction will appear here.</span>
      </div>

    </div>
  </div>

  <!-- ───────────── RIGHT PANEL: Scenarios + Draft ───────────── -->
  <div class="panel">
    <div class="panel-header">
      <div class="panel-icon">🔄</div>
      <div>
        <div class="panel-title">Scenario Adaptation</div>
        <div class="panel-sub">Select a scenario to generate an adapted policy draft</div>
      </div>
    </div>
    <div class="panel-body">

      <!-- Scenario cards -->
      <div class="scenario-grid">
        {% for name, sc in scenarios.items() %}
        <div class="sc-card" data-id="{{ name }}" onclick="selectScenario('{{ name }}', this)">
          <div class="sc-head">
            <div class="sc-icon">{{ sc.icon }}</div>
            <div class="sc-title">{{ name }}</div>
            <div class="sc-check">✓</div>
          </div>
          <div class="sc-desc">{{ sc.desc }}</div>
        </div>
        {% endfor %}
      </div>

      <div style="margin-top:1rem;">
        <button class="btn btn-primary" id="btn-generate" onclick="doGenerate()" disabled>
          <span class="spin"></span>
          <span class="btn-lbl">✍️ Generate Adapted Policy Draft</span>
        </button>
      </div>

      <div class="divider"></div>

      <!-- Draft output -->
      <div class="sec-label">📝 Adapted Policy Draft</div>
      <div class="out-box" id="draft-box">
        <span class="out-placeholder">Your adapted policy draft will appear here. Summarise a document and select a scenario first for the best results.</span>
      </div>

      <!-- Copy button (hidden until draft is ready) -->
      <div id="copy-area" style="display:none; margin-top:.7rem;">
        <button class="btn btn-secondary" onclick="copyDraft()">
          📋 Copy Draft to Clipboard
        </button>
      </div>

    </div>
  </div>

</main>

<!-- Toast -->
<div class="toast" id="toast"></div>

<!-- ═══════════════════ JAVASCRIPT ═══════════════════ -->
<script>
/* ── State ─────────────────────────────────────────── */
let selectedScenario = null;
let summarised       = false;

/* ── Step tracker ───────────────────────────────────── */
function setStep(n) {
  for (let i = 1; i <= 4; i++) {
    const el = document.getElementById('step' + i);
    el.classList.remove('active', 'done');
    if (i < n)  el.classList.add('done');
    if (i === n) el.classList.add('active');
  }
}

/* ── File selection ─────────────────────────────────── */
function onFileSelect(input) {
  const f = input.files[0];
  if (!f) return;
  const nameEl = document.getElementById('uz-filename');
  nameEl.textContent = '✅  ' + f.name;
  nameEl.style.display = 'block';
  document.getElementById('btn-summarise').disabled = false;
  setStep(2);
}

/* ── Summarise ──────────────────────────────────────── */
function doSummarise() {
  const file = document.getElementById('file-input').files[0];
  if (!file) { showToast('Please select a file first.', true); return; }

  setBtnLoading('btn-summarise', true);
  clearBoxes();

  const fd = new FormData();
  fd.append('file', file);

  fetch('/summarise', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(data => {
      setBtnLoading('btn-summarise', false);
      if (data.error) { showToast(data.error, true); return; }

      fillBox('box-goals',  data.goals);
      fillBox('box-strats', data.strategies);
      fillBox('box-dir',    data.direction);

      summarised = true;
      updateGenerateBtn();
      setStep(3);
      showToast('✅  Policy summarised successfully!');
    })
    .catch(() => {
      setBtnLoading('btn-summarise', false);
      showToast('Failed to process the file. Please try again.', true);
    });
}

/* ── Scenario selection ─────────────────────────────── */
function selectScenario(name, el) {
  selectedScenario = name;
  document.querySelectorAll('.sc-card').forEach(c => c.classList.remove('selected'));
  el.classList.add('selected');
  updateGenerateBtn();
  if (summarised) setStep(4);
}

/* ── Generate draft ─────────────────────────────────── */
function doGenerate() {
  if (!selectedScenario) { showToast('Please select a scenario first.', true); return; }

  setBtnLoading('btn-generate', true);

  fetch('/generate', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      scenario:   selectedScenario,
      goals:      getBoxLines('box-goals'),
      strategies: getBoxLines('box-strats'),
      direction:  getBoxLines('box-dir'),
    })
  })
  .then(r => r.json())
  .then(data => {
    setBtnLoading('btn-generate', false);
    if (data.error) { showToast(data.error, true); return; }
    const box = document.getElementById('draft-box');
    box.textContent = data.draft;
    box.classList.add('has-content');
    document.getElementById('copy-area').style.display = 'block';
    setStep(4);
    showToast('✅  Draft generated successfully!');
  })
  .catch(() => {
    setBtnLoading('btn-generate', false);
    showToast('Failed to generate draft. Please try again.', true);
  });
}

/* ── Helpers ────────────────────────────────────────── */
function fillBox(id, arr) {
  const el = document.getElementById(id);
  if (!arr || !arr.length) {
    el.innerHTML = '<span class="out-placeholder">No content extracted for this section.</span>';
    return;
  }
  el.textContent = arr.map(s => '• ' + s).join('\n');
  el.classList.add('has-content');
}

function clearBoxes() {
  ['box-goals', 'box-strats', 'box-dir'].forEach(id => {
    const el = document.getElementById(id);
    el.classList.remove('has-content');
    el.innerHTML = '<span class="out-placeholder">Processing…</span>';
  });
}

function getBoxLines(id) {
  return document.getElementById(id).textContent
    .split('\n')
    .map(s => s.replace(/^[•\-]\s*/, '').trim())
    .filter(s => s.length > 10 && !s.startsWith('Extracted') && !s.startsWith('Processing'));
}

function updateGenerateBtn() {
  document.getElementById('btn-generate').disabled = !selectedScenario;
}

function setBtnLoading(id, on) {
  document.getElementById(id).classList.toggle('loading', on);
  document.getElementById(id).disabled = on;
}

function copyDraft() {
  const text = document.getElementById('draft-box').textContent;
  navigator.clipboard.writeText(text)
    .then(() => showToast('📋  Draft copied to clipboard!'))
    .catch(() => showToast('Could not copy. Please select and copy manually.', true));
}

/* ── Toast ──────────────────────────────────────────── */
let toastTimer;
function showToast(msg, isError = false) {
  clearTimeout(toastTimer);
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast' + (isError ? ' error' : '');
  requestAnimationFrame(() => {
    requestAnimationFrame(() => t.classList.add('show'));
  });
  toastTimer = setTimeout(() => t.classList.remove('show'), 3500);
}
</script>
</body>
</html>"""


# =============================================================
#  ENTRY POINT
# =============================================================

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  PolicyLens — Sri Lanka NEP Analyser")
    print("  http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True)