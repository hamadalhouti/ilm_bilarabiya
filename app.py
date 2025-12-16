import json
import html
import re
import streamlit as st
from openai import OpenAI
import requests
from io import BytesIO
import os

# -----------------------------
# Optional deps (URL / PDF input)
# -----------------------------
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except Exception:
    BeautifulSoup = None
    HAS_BS4 = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except Exception:
    PdfReader = None
    HAS_PYPDF = False

# -----------------------------
# Optional deps (PDF export)
# -----------------------------
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAS_REPORTLAB = True
except Exception:
    canvas = None
    A4 = None
    pdfmetrics = None
    TTFont = None
    HAS_REPORTLAB = False

# Arabic shaping (recommended for correct Arabic in PDF)
try:
    import arabic_reshaper
    HAS_RESHAPER = True
except Exception:
    arabic_reshaper = None
    HAS_RESHAPER = False

try:
    from bidi.algorithm import get_display
    HAS_BIDI = True
except Exception:
    get_display = None
    HAS_BIDI = False


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="العِلم بالعربية", page_icon="🧠", layout="wide")

# -----------------------------
# Styling (RTL)
# -----------------------------
st.markdown(
    """
    <style>
      html, body, [class*="stApp"] { direction: rtl; }
      .rtl { direction: rtl; text-align: right; unicode-bidi: plaintext; }
      .center { text-align: center; }
      .muted { opacity: 0.82; font-size: 0.95rem; }
      textarea, input { direction: rtl !important; text-align: right !important; }

      .card {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 14px 16px;
        margin: 10px 0;
        background: rgba(255,255,255,0.03);
      }
      mark { padding: 0.08em 0.22em; border-radius: 0.35em; }
      .tag {
        display:inline-block; padding:2px 8px; border-radius:999px;
        border: 1px solid rgba(255,255,255,0.18); opacity:0.9; font-size:0.85rem;
        margin-inline-start: 8px;
      }
      .warnbox {
        border: 1px solid rgba(255,180,0,0.35);
        background: rgba(255,180,0,0.08);
        border-radius: 12px;
        padding: 10px 12px;
        margin: 8px 0 14px 0;
      }
      code { direction:ltr; text-align:left; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='center'>🧠 العِلم بالعربية</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='center muted'>مصطلحات علمية حسب التخصص + ترجمة ورقة بحثية كاملة (نص/رابط/PDF) + تحميل PDF</p>",
    unsafe_allow_html=True
)
st.divider()

# -----------------------------
# API key
# -----------------------------
if "OPENAI_API_KEY" not in st.secrets:
    st.error("ما لقيت OPENAI_API_KEY داخل .streamlit/secrets.toml")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------
# Modes (default: شامل)
# -----------------------------
DOMAIN_PRESETS = {
    "شامل": {
        "label": "General Science (Auto-detect domain from text)",
        "keep_terms_hint": "Keep scientific/technical terms across environmental, marine, medical, chemistry, and physics.",
        "bonus_regex": r"\b(eutrophication|bioaccumulation|biomagnification|acidification|nitrification|denitrification|salinity|thermocline|upwelling|zooplankton|phytoplankton|pathogenesis|cytokines|immunotherapy|stoichiometry|catalysis|redox|spectroscopy|quantum|relativity|diffraction|entropy|semiconductor)\b",
    },
    "بيئي": {
        "label": "Environmental Science",
        "keep_terms_hint": "Focus on ecology, pollution, nutrients, cycles, toxicology, climate, soil/water/air processes.",
        "bonus_regex": r"\b(eutrophication|bioaccumulation|biomagnification|acidification|nitrification|denitrification|sedimentation|microplastics|remediation|biogeochemical|ecotoxicology|runoff|watershed|nutrient\s+loading)\b",
    },
    "بحري": {
        "label": "Marine Science",
        "keep_terms_hint": "Focus on oceanography, marine ecology, reefs, plankton, salinity, currents, hypoxia.",
        "bonus_regex": r"\b(salinity|thermocline|halocline|upwelling|zooplankton|phytoplankton|benthic|pelagic|anoxia|coral\s+bleaching|ocean\s+acidification|trophic)\b",
    },
    "طبي": {
        "label": "Medical/Biomedical",
        "keep_terms_hint": "Focus on anatomy, physiology, pathology, immunology, pharmacology, diagnostics, diseases.",
        "bonus_regex": r"\b(pathogenesis|inflammation|cytokines|immunotherapy|biomarker|metastasis|homeostasis|pathophysiology|pharmacokinetics|antibiotic\s+resistance)\b",
    },
    "كيميائي": {
        "label": "Chemistry",
        "keep_terms_hint": "Focus on reactions, equilibria, thermodynamics, kinetics, spectroscopy, analytical methods, materials.",
        "bonus_regex": r"\b(stoichiometry|catalysis|electronegativity|redox|oxidation|reduction|equilibrium|enthalpy|kinetics|spectroscopy|chromatography|polymerization)\b",
    },
    "فيزيائي": {
        "label": "Physics",
        "keep_terms_hint": "Focus on mechanics, waves, electricity/magnetism, quantum, relativity, materials/semiconductors.",
        "bonus_regex": r"\b(quantum|relativity|wavelength|frequency|diffraction|interference|entropy|momentum|acceleration|electromagnetic|conductivity|semiconductor)\b",
    },
}

# -----------------------------
# Helpers
# -----------------------------
def safe_json_loads(s: str):
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def call_ai_text(system_prompt: str, user_prompt: str, model: str = "gpt-4.1-mini", max_output_tokens: int = 2000) -> str:
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        temperature=0.2,
        max_output_tokens=max_output_tokens,
    )
    return (getattr(resp, "output_text", "") or "").strip()

def call_ai_json(system_prompt: str, user_prompt: str, model: str = "gpt-4.1-mini", max_output_tokens: int = 2800) -> str:
    return call_ai_text(system_prompt, user_prompt, model=model, max_output_tokens=max_output_tokens)

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

# -----------------------------
# PDF export helpers (ReportLab) + Arabic shaping
# -----------------------------
def _shape_arabic_for_pdf(s: str) -> str:
    # For correct Arabic: reshape + bidi display
    if HAS_RESHAPER and HAS_BIDI:
        try:
            return get_display(arabic_reshaper.reshape(s))
        except Exception:
            return s
    return s

def _register_arabic_font():
    """
    حاول تسجيل خط عربي:
    - الأفضل: fonts/Amiri-Regular.ttf
    - ثم محاولات شائعة في الأنظمة
    """
    if not HAS_REPORTLAB:
        return None

    candidates = [
        os.path.join("fonts", "Amiri-Regular.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("AR_FONT", path))
                return "AR_FONT"
            except Exception:
                pass
    return None

def _wrap_lines(text: str, font_name: str, font_size: int, max_width: float) -> list[str]:
    """
    لفّ النص إلى أسطر بحيث لا يتجاوز العرض.
    """
    lines_out = []
    for para in (text or "").splitlines():
        if not para.strip():
            lines_out.append("")
            continue

        words = para.split(" ")
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            try:
                width = pdfmetrics.stringWidth(test, font_name, font_size)
            except Exception:
                width = len(test) * (font_size * 0.45)

            if width <= max_width:
                cur = test
            else:
                if cur:
                    lines_out.append(cur)
                cur = w
        if cur:
            lines_out.append(cur)
    return lines_out

def build_pdf_bytes(title: str, text: str) -> bytes:
    """
    يبني PDF:
    - محاذاة يمين
    - تشكيل عربي إذا توفر arabic_reshaper + python-bidi
    """
    if not HAS_REPORTLAB:
        raise RuntimeError("reportlab غير مثبتة. ثبّتها عبر: python3 -m pip install reportlab")

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    margin = 50
    x_right = width - margin
    y = height - margin

    font_name = _register_arabic_font() or "Helvetica"
    font_size = 12
    title_size = 14
    line_gap = 18

    # Title
    c.setFont(font_name, title_size)
    shaped_title = _shape_arabic_for_pdf(title) if title else ""
    c.drawRightString(x_right, y, shaped_title)
    y -= (line_gap + 6)

    # Body
    c.setFont(font_name, font_size)
    max_width = width - 2 * margin

    lines = _wrap_lines(text, font_name, font_size, max_width)

    for line in lines:
        if y < margin + 20:
            c.showPage()
            y = height - margin
            c.setFont(font_name, font_size)

        if not line.strip():
            y -= line_gap
            continue

        shaped = _shape_arabic_for_pdf(line)
        c.drawRightString(x_right, y, shaped)
        y -= line_gap

    c.save()
    return buffer.getvalue()

# -----------------------------
# Block publishing/journal terms only
# -----------------------------
PUBLISHING_BLOCKLIST = {
    "open access", "peer reviewed", "peer-reviewed", "double-blind", "single-blind",
    "refereed", "refereed journal", "impact factor", "scopus", "web of science",
    "doi", "publisher", "copyright", "creative commons", "cc-by",
    "preprint", "editorial", "submission", "acceptance rate", "author guidelines"
}

def is_publishing_term(term: str) -> bool:
    t = normalize_spaces(term).lower()
    if t in PUBLISHING_BLOCKLIST:
        return True
    pub_patterns = [
        r"\bopen\s*access\b",
        r"\bpeer[-\s]*review(ed)?\b",
        r"\bdouble[-\s]*blind\b",
        r"\bimpact\s*factor\b",
        r"\bcreative\s*commons\b",
        r"\bweb\s*of\s*science\b",
        r"\bscopus\b",
        r"\bdoi\b",
    ]
    return any(re.search(p, t) for p in pub_patterns)

# -----------------------------
# Occurrence validation + fallback locating
# -----------------------------
def validate_occurrences(text: str, term: str, occs: list) -> list:
    n = len(text)
    term_norm = normalize_spaces(term).lower()
    term_loose = re.sub(r"[^\w\s\-]", "", term_norm)

    good = []
    for oc in occs or []:
        try:
            s = int(oc.get("start"))
            e = int(oc.get("end"))
        except Exception:
            continue
        s = max(0, min(n, s))
        e = max(0, min(n, e))
        if e <= s:
            continue

        snippet = text[s:e]
        sn_norm = normalize_spaces(snippet).lower()
        sn_loose = re.sub(r"[^\w\s\-]", "", sn_norm)

        if sn_norm == term_norm or sn_loose == term_loose:
            good.append((s, e))

    good.sort(key=lambda x: (x[0], x[1]))
    merged = []
    for s, e in good:
        if not merged:
            merged.append([s, e])
        else:
            ls, le = merged[-1]
            if s <= le:
                merged[-1][1] = max(le, e)
            else:
                merged.append([s, e])
    return [{"start": s, "end": e} for s, e in merged]

def find_occurrences_fallback(text: str, term: str, max_hits: int = 6) -> list:
    hits = []
    if not term.strip():
        return hits

    if re.search(r"[A-Za-z]", term):
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        for m in pattern.finditer(text):
            hits.append({"start": m.start(), "end": m.end()})
            if len(hits) >= max_hits:
                break
        return hits

    idx = 0
    while idx < len(text) and len(hits) < max_hits:
        pos = text.find(term, idx)
        if pos == -1:
            break
        hits.append({"start": pos, "end": pos + len(term)})
        idx = pos + len(term)
    return hits

# -----------------------------
# Inclusive scoring (single-word friendly) + domain bonus
# -----------------------------
def scientific_score(term: str, bonus_regex: str) -> int:
    t = term.strip()
    lo = t.lower()

    if is_publishing_term(t):
        return -999

    score = 0
    if t.isupper() and 2 <= len(t) <= 12:
        score += 4
    if re.search(r"[\dα-ωΑ-Ω\-\(\)/]", t):
        score += 3
    if len(t.split()) >= 2:
        score += 2
    if len(lo) >= 9:
        score += 2

    if re.search(
        r"\b(ase|itis|osis|cation|lysis|genic|phage|phyte|cyte|enzyme|protein|genome|polymer|spectro|chromato|thermo|kinetic|quantum|electro|magnet|neuro|immuno|patho|toxic|ecolog|ocean|marine)\b",
        lo
    ):
        score += 2

    if bonus_regex and re.search(bonus_regex, lo):
        score += 2

    if len(lo) <= 3 and not t.isupper():
        score -= 2

    return score

def normalize_terms(raw_terms: list, text: str, bonus_regex: str) -> list:
    buckets = {}
    for t in raw_terms or []:
        term = (t.get("term") or "").strip()
        if not term:
            continue
        if is_publishing_term(term):
            continue
        if scientific_score(term, bonus_regex) < 0:
            continue

        key = normalize_spaces(term).lower()
        item = {
            "term": term,
            "arabic": (t.get("arabic") or "").strip(),
            "category": (t.get("category") or "").strip(),
            "definition": (t.get("definition") or "").strip(),
            "example": (t.get("example") or "").strip(),
            "occurrences": [],
        }

        occs = validate_occurrences(text, term, t.get("occurrences") or [])
        if not occs:
            occs = find_occurrences_fallback(text, term, max_hits=6)

        if not occs:
            continue
        item["occurrences"] = occs

        if key not in buckets:
            buckets[key] = item
        else:
            if len(term) > len(buckets[key]["term"]):
                buckets[key]["term"] = term

            buckets[key]["occurrences"].extend(occs)
            buckets[key]["occurrences"] = validate_occurrences(text, buckets[key]["term"], buckets[key]["occurrences"])
            if not buckets[key]["occurrences"]:
                buckets[key]["occurrences"] = find_occurrences_fallback(text, buckets[key]["term"], max_hits=6)

            if item["arabic"] and not buckets[key]["arabic"]:
                buckets[key]["arabic"] = item["arabic"]
            if len(item["definition"]) > len(buckets[key]["definition"]):
                buckets[key]["definition"] = item["definition"]
            if len(item["example"]) > len(buckets[key]["example"]):
                buckets[key]["example"] = item["example"]
            if item["category"] and not buckets[key]["category"]:
                buckets[key]["category"] = item["category"]

    out = []
    for v in buckets.values():
        first = v["occurrences"][0]["start"] if v["occurrences"] else 10**18
        v["_first"] = first
        out.append(v)
    out.sort(key=lambda x: x["_first"])
    for v in out:
        v.pop("_first", None)
    return out

def highlight_terms(text: str, terms: list) -> str:
    spans = []
    n = len(text)
    for t in terms:
        for oc in (t.get("occurrences") or []):
            try:
                s = int(oc["start"]); e = int(oc["end"])
            except Exception:
                continue
            if 0 <= s < e <= n:
                spans.append((s, e))

    spans.sort(key=lambda x: (x[0], -(x[1]-x[0])))
    cleaned = []
    last_end = -1
    for s, e in spans:
        if s >= last_end:
            cleaned.append((s, e))
            last_end = e

    out = []
    idx = 0
    for s, e in cleaned:
        out.append(html.escape(text[idx:s]).replace("\n", "<br>"))
        out.append(f"<mark>{html.escape(text[s:e])}</mark>")
        idx = e
    out.append(html.escape(text[idx:]).replace("\n", "<br>"))
    return f"<div class='rtl'>{''.join(out)}</div>"

def render_term_card(item: dict):
    term = (item.get("term") or "").strip()
    ar = (item.get("arabic") or "").strip()
    definition = (item.get("definition") or "").strip()
    example = (item.get("example") or "").strip()
    category = (item.get("category") or "").strip()

    st.markdown(
        f"""
        <div class="card rtl">
          <h3>📌 {html.escape(term)} {f"<span class='tag'>{html.escape(category)}</span>" if category else ""}</h3>
          <p><b>الترجمة العربية:</b> {html.escape(ar) if ar else "—"}</p>
          <p><b>التعريف:</b> {html.escape(definition) if definition else "—"}</p>
          {f"<p><b>مثال في نفس السياق:</b> {html.escape(example)}</p>" if example else ""}
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Paper translation utilities
# -----------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    if not HAS_PYPDF:
        raise RuntimeError("مكتبة pypdf غير مثبتة. ثبّتها عبر: python3 -m pip install pypdf")
    reader = PdfReader(uploaded_file)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(parts).strip()

def extract_text_from_url(url: str) -> str:
    if not HAS_BS4:
        raise RuntimeError("مكتبة beautifulsoup4 غير مثبتة. ثبّتها عبر: python3 -m pip install beautifulsoup4")
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def chunk_text(text: str, max_chars: int = 6000) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    paras = re.split(r"\n\s*\n", text)
    chunks = []
    cur = ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            if len(p) <= max_chars:
                cur = p
            else:
                sentences = re.split(r"(?<=[\.\!\?])\s+", p)
                cur2 = ""
                for s in sentences:
                    s = s.strip()
                    if not s:
                        continue
                    if len(cur2) + len(s) + 1 <= max_chars:
                        cur2 = (cur2 + " " + s).strip()
                    else:
                        if cur2:
                            chunks.append(cur2)
                        cur2 = s
                cur = cur2
    if cur:
        chunks.append(cur)
    return chunks

def translate_full_text_to_ar(text: str) -> str:
    system = (
        "أنت مترجم علمي عربي محترف.\n"
        "ترجم النص إلى العربية الفصحى بأسلوب أكاديمي واضح.\n"
        "- حافظ على المصطلحات العلمية بدقة.\n"
        "- لا تضف شرحًا خارج الترجمة.\n"
        "- حافظ على العناوين والترقيم والنقاط.\n"
    )
    chunks = chunk_text(text, max_chars=6000)
    outputs = []
    for i, ch in enumerate(chunks, start=1):
        user = f"الجزء {i}/{len(chunks)}:\n\n{ch}"
        out = call_ai_text(system, user, max_output_tokens=2000)
        outputs.append(out)
    return "\n\n".join(outputs).strip()


# -----------------------------
# UI: Tabs
# -----------------------------
tab1, tab2 = st.tabs(["🔎 استخراج المصطلحات", "🌍 ترجمة ورقة بحثية كاملة"])

# -----------------------------
# Tab 1: Terms extraction
# -----------------------------
with tab1:
    preset = st.selectbox("اختر التخصص:", list(DOMAIN_PRESETS.keys()), index=0)
    cfg = DOMAIN_PRESETS[preset]

    article = st.text_area("الصق النص هنا:", height=280, placeholder="Paste here...")

    c1, c2 = st.columns([1, 1])
    with c1:
        run_terms = st.button("🧠 استخراج المصطلحات وتحديدها", key="run_terms")
    with c2:
        clear_terms = st.button("🧹 مسح", key="clear_terms")

    if clear_terms:
        for k in ["terms", "highlighted"]:
            st.session_state.pop(k, None)
        st.rerun()

    if run_terms and article.strip():
        with st.spinner("جاري التحليل..."):
            system_prompt = (
                f"أنت مساعد عربي يحدد المصطلحات العلمية/التقنية فقط.\n"
                f"السياق/التخصص المفضل: {cfg['label']}.\n"
                f"توجيه: {cfg['keep_terms_hint']}\n\n"
                "ممنوع استخراج مصطلحات النشر الأكاديمي أو الإدارة مثل: open access, peer reviewed, double-blind, refereed, impact factor, DOI, Scopus.\n\n"
                "المطلوب:\n"
                "- استخرج المصطلحات العلمية/التقنية المهمة فقط (تشمل الكلمات الواحدة مثل eutrophication).\n"
                "- أعطِ ترجمة عربية دقيقة قدر الإمكان.\n"
                "- التعريف: 3 إلى 6 جمل قصيرة واضحة (بدون كتابة: 'المصطلح X هو').\n"
                "- مثال قصير مرتبط بسياق النص.\n"
                "- مهم جداً: أعِد مواضع المصطلح داخل النص بدقة عبر start/end (0-based) بحيث text[start:end] يطابق ظهور المصطلح.\n"
                "- أعطِ حتى 6 occurrences.\n\n"
                "شكل الإخراج: JSON فقط بالشكل:\n"
                "{"
                "\"terms\":["
                "{"
                "\"term\":\"\","
                "\"arabic\":\"\","
                "\"category\":\"\","
                "\"definition\":\"\","
                "\"example\":\"\","
                "\"occurrences\":[{\"start\":0,\"end\":0}]"
                "}"
                "]"
                "}"
            )

            out = call_ai_json(system_prompt, f"النص:\n{article}", max_output_tokens=2800)
            data = safe_json_loads(out)

            if not data or "terms" not in data:
                st.error("الناتج غير مفهوم. جرّب نص أقصر أو أوضح.")
                st.code(out)
                st.stop()

            raw_terms = data.get("terms", [])
            if not raw_terms:
                st.warning("ما لقيت مصطلحات واضحة. جرّب فقرة علمية أكثر.")
                st.stop()

            terms = normalize_terms(raw_terms, article, cfg["bonus_regex"])

            if not terms:
                st.warning("لم أستطع استخراج مصطلحات علمية واضحة من النص الحالي. جرّب فقرة أكثر تخصصاً.")
                st.stop()

            st.session_state["terms"] = terms
            st.session_state["highlighted"] = highlight_terms(article, terms)

    terms = st.session_state.get("terms")
    highlighted = st.session_state.get("highlighted")
    if terms and highlighted:
        st.markdown("<div class='rtl'><h2>🖍️ النص مع تحديد المصطلحات</h2></div>", unsafe_allow_html=True)
        st.markdown(highlighted, unsafe_allow_html=True)

        st.markdown("<div class='rtl'><h2>✅ شرح المصطلحات</h2></div>", unsafe_allow_html=True)
        for t in terms:
            render_term_card(t)

# -----------------------------
# Tab 2: Full paper translation + PDF download only
# -----------------------------
with tab2:
    st.markdown("<div class='rtl muted'>اختر طريقة الإدخال ثم اضغط ترجمة. بعدها يمكنك تحميل PDF.</div>", unsafe_allow_html=True)

    missing_msgs = []
    if not HAS_BS4:
        missing_msgs.append("ميزة **رابط URL** تحتاج: `beautifulsoup4`")
    if not HAS_PYPDF:
        missing_msgs.append("ميزة **PDF (استخراج نص)** تحتاج: `pypdf`")
    if not HAS_REPORTLAB:
        missing_msgs.append("ميزة **تحميل PDF** تحتاج: `reportlab`")
    if HAS_REPORTLAB and (not HAS_RESHAPER or not HAS_BIDI):
        missing_msgs.append("لأفضل PDF عربي: ثبّت `arabic-reshaper` و `python-bidi` (موصى به)")

    if missing_msgs:
        st.markdown(
            "<div class='warnbox rtl'>"
            "<b>ملاحظة:</b> بعض الميزات قد تكون غير مفعّلة لأن مكتبات اختيارية غير مثبتة.<br>"
            + "<br>".join([f"• {m}" for m in missing_msgs])
            + "<br><br><b>أوامر التثبيت:</b><br>"
            + ("<code>python3 -m pip install beautifulsoup4</code><br>" if not HAS_BS4 else "")
            + ("<code>python3 -m pip install pypdf</code><br>" if not HAS_PYPDF else "")
            + ("<code>python3 -m pip install reportlab</code><br>" if not HAS_REPORTLAB else "")
            + ("<code>python3 -m pip install arabic-reshaper python-bidi</code><br>" if HAS_REPORTLAB and (not HAS_RESHAPER or not HAS_BIDI) else "")
            + "</div>",
            unsafe_allow_html=True
        )

    input_method = st.radio("طريقة إدخال الورقة:", ["نسخ/لصق", "رابط URL", "PDF"], horizontal=True)

    source_text = ""
    if input_method == "نسخ/لصق":
        source_text = st.text_area("الصق نص الورقة هنا:", height=280)

    elif input_method == "رابط URL":
        url = st.text_input("ضع رابط الورقة/المقال:")
        fetch_disabled = not HAS_BS4
        if st.button("📥 جلب النص من الرابط", disabled=fetch_disabled):
            if not url.strip():
                st.warning("اكتب الرابط أولاً.")
            else:
                with st.spinner("جاري جلب النص..."):
                    try:
                        source_text = extract_text_from_url(url.strip())
                        st.session_state["paper_text"] = source_text
                    except Exception as e:
                        st.error(f"فشل جلب النص: {e}")

        source_text = st.session_state.get("paper_text", "")
        if source_text:
            st.text_area("النص المستخرج (يمكنك تعديله قبل الترجمة):", value=source_text, height=220)

    else:  # PDF input
        pdf = st.file_uploader("ارفع ملف PDF:", type=["pdf"])
        extract_disabled = not HAS_PYPDF
        if pdf is not None and st.button("📄 استخراج النص من PDF", disabled=extract_disabled):
            with st.spinner("جاري استخراج النص..."):
                try:
                    source_text = extract_text_from_pdf(pdf)
                    st.session_state["paper_text"] = source_text
                except Exception as e:
                    st.error(f"فشل استخراج النص: {e}")

        source_text = st.session_state.get("paper_text", "")
        if source_text:
            st.text_area("النص المستخرج (يمكنك تعديله قبل الترجمة):", value=source_text, height=220)

    colA, colB = st.columns([1, 1])
    with colA:
        do_translate = st.button("🌍 ترجمة كاملة إلى العربية")
    with colB:
        clear_paper = st.button("🧹 مسح الورقة")

    if clear_paper:
        st.session_state.pop("paper_text", None)
        st.session_state.pop("paper_translation_raw", None)
        st.session_state.pop("paper_pdf_bytes", None)
        st.rerun()

    if do_translate:
        text_to_translate = (source_text or "").strip()
        if not text_to_translate:
            st.warning("ما عندي نص أترجمه. اختر طريقة إدخال وأدخل النص/استخرجه أولاً.")
        else:
            with st.spinner("جاري الترجمة..."):
                try:
                    ar_raw = translate_full_text_to_ar(text_to_translate)
                    st.session_state["paper_translation_raw"] = ar_raw

                    # Pre-build PDF bytes if possible
                    if HAS_REPORTLAB:
                        pdf_bytes = build_pdf_bytes("الترجمة العربية", ar_raw)
                        st.session_state["paper_pdf_bytes"] = pdf_bytes
                    else:
                        st.session_state["paper_pdf_bytes"] = None

                except Exception as e:
                    st.error(f"خطأ أثناء الترجمة: {e}")

    ar_raw = st.session_state.get("paper_translation_raw")
    pdf_bytes = st.session_state.get("paper_pdf_bytes")

    if ar_raw:
        st.markdown("<div class='rtl'><h2>✅ الترجمة العربية</h2></div>", unsafe_allow_html=True)
        st.text_area("الترجمة:", value=ar_raw, height=360)

        if HAS_REPORTLAB and pdf_bytes:
            st.download_button(
                "⬇️ تحميل PDF",
                data=pdf_bytes,
                file_name="translation_ar.pdf",
                mime="application/pdf"
            )
            if not (HAS_RESHAPER and HAS_BIDI):
                st.info("ملاحظة: قد لا تكون الحروف العربية متصلة تمامًا في PDF إلا بعد تثبيت arabic-reshaper و python-bidi ووضع خط عربي (مثل Amiri).")
        else:
            st.warning("ميزة تحميل PDF غير متاحة حالياً. ثبّت reportlab.")
