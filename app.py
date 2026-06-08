# =============================================================================
# APLIKASI DEMO WSD RULE‑BASED + POS TAGGING
# =============================================================================
import streamlit as st
import pandas as pd
import stanza

# -------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Demo WSD | Rule‑Based + POS Tagging",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }
    .sense-id {
        font-size: 2rem;
        font-weight: 800;
        color: #1f77b4;
    }
    .sense-desc {
        font-size: 1.2rem;
        font-weight: 600;
        color: #333;
        margin-top: 0.3rem;
    }
    .sense-info {
        font-size: 0.95rem;
        color: #666;
        margin-top: 0.8rem;
        font-style: italic;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        padding: 0.6rem 0.5rem;
        border-radius: 10px;
        border: none;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #145a8b;
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# LOAD MODEL STANZA & DATASET
# -------------------------------------------------------------------------
@st.cache_resource
def load_nlp():
    try:
        nlp = stanza.Pipeline('id', processors='tokenize,pos', use_gpu=False)
        return nlp
    except Exception as e:
        st.error(f"Gagal memuat model Stanza: {e}")
        st.info("Jalankan `stanza.download('id')` terlebih dahulu di terminal.")
        st.stop()

nlp = load_nlp()

@st.cache_data
def load_dataset():
    df = pd.read_excel("dataset.xlsx", engine="openpyxl")
    return df

df = load_dataset()

# -------------------------------------------------------------------------
# ATURAN DENGAN POS UNTUK SEMUA
# -------------------------------------------------------------------------
RULES = {
    "alam": [
        {
            "keywords": [
                "semesta", "galaksi", "planet", "bintang", "kosmos",
                "jagat raya", "angkasa", "tata surya", "alam semesta",
                "luar angkasa", "bima sakti", "matahari", "bulan"
            ],
            "pos_target": "NOUN",
            "synset_id": 9466280.0,
            "deskripsi": "dunia; alam semesta"
        },
        {
            "keywords": [
                "hutan", "liar", "cagar", "lingkungan", "fenomena",
                "bebas", "ibu alam", "hukum alam", "ekosistem",
                "kehidupan", "alam sekitar", "keseimbangan alam",
                "di alam", "oleh alam", "dengan alam", "alam liar",
                "alam bebas", "flora", "fauna", "habitat",
                "konservasi", "pencemaran", "lingkungan hidup"
            ],
            "pos_target": "NOUN",
            "synset_id": 9366762.0,
            "deskripsi": "lingkungan kehidupan / alam sekitar"
        }
    ],
    "anggur": [
        {
            "keywords": [
                "minum", "botol", "gelas", "fermentasi", "mabuk",
                "anggur merah", "anggur putih", "arak", "minuman keras",
                "pesta", "khamr", "alkohol", "mabuk anggur",
                "minuman", "tuak", "sampanye", "bar"
            ],
            "pos_target": "NOUN",
            "synset_id": 7758680.0,
            "deskripsi": "minuman fermentasi (anggur)"
        },
        {
            "keywords": [
                "buah", "kebun", "pohon", "manis", "kebun anggur",
                "petik anggur", "anggur hijau", "anggur hitam",
                "vitikultur", "anggur segar", "jus anggur",
                "selai anggur", "anggur organik", "tanaman anggur"
            ],
            "pos_target": "NOUN",
            "synset_id": 1190840.0,
            "deskripsi": "buah anggur"
        }
    ],
    "atas": [
        {
            "keywords": [
                "bagian atas", "sisi atas", "pihak atas", "orang atas",
                "lapisan atas", "atasan", "permukaan atas",
                "tingkat atas", "ujung atas", "batas atas",
                "sebelah atas", "ruang atas", "lantai atas"
            ],
            "pos_target": "NOUN",
            "synset_id": 5037617.0,
            "deskripsi": "bagian atas (nomina)"
        },
        {
            "keywords": [
                "di atas", "ke atas", "dari atas",
                "menuju atas", "sampai atas", "melayang di atas",
                "terbang di atas", "berada di atas", "ke atas sana"
            ],
            "pos_target": "ADP",
            "synset_id": 125993.0,
            "deskripsi": "preposisi lokatif"
        },
        {
            "keywords": [
                "atas nama", "berdasarkan atas", "atas dasar",
                "kemenangan atas", "diskusi atas", "kontrol atas",
                "kuasa atas", "hak atas", "menang atas"
            ],
            "pos_target": "ADP",
            "synset_id": 125993.0,
            "deskripsi": "preposisi metaforis/abstrak"
        },
        {
            "keywords": [
                "atas", "puncak", "teratas", "atasnya",
                "paling atas", "sebelah atas"
            ],
            "pos_target": "NOUN",
            "synset_id": 5037617.0,
            "deskripsi": "nomina umum (fallback)"
        }
    ],
    "dasar": [
        {
            "keywords": [
                "sekolah", "pendidikan", "guru", "kelas", "siswa",
                "belajar", "pelajaran", "dasar-dasar", "kurikulum",
                "sekolah dasar", "tingkat dasar", "pengetahuan dasar"
            ],
            "pos_target": "NOUN",
            "synset_id": 901060.0,
            "deskripsi": "pokok/asas (pendidikan dasar)"
        },
        {
            "keywords": [
                "pemikiran", "argumen", "landasan", "prinsip",
                "filosofi", "teori", "dasar pemikiran", "fondasi",
                "asas", "pijakan", "fundamental", "landasan teori",
                "kerangka berpikir", "dasar hukum"
            ],
            "pos_target": "NOUN",
            "synset_id": 1856419.0,
            "deskripsi": "landasan pemikiran"
        }
    ],
    "kayu": [
        {
            "keywords": [
                "pohon", "hutan", "batang", "tebang", "illegal",
                "tanaman", "kayu jati", "kayu mahoni", "pohon jati",
                "penebangan", "hutan lindung", "kayu ulin",
                "kayu meranti", "kayu pinus", "kayu hutan"
            ],
            "pos_target": "NOUN",
            "synset_id": 14943580.0,
            "deskripsi": "pohon penghasil kayu"
        },
        {
            "keywords": [
                "meja", "kursi", "papan", "tongkat", "peti",
                "kotak", "pensil", "lemari kayu", "perabot",
                "mebel", "patung kayu", "bahan kayu",
                "kerajinan kayu", "pintu kayu", "jendela kayu",
                "bingkai kayu", "mainan kayu", "alat kayu"
            ],
            "pos_target": "NOUN",
            "synset_id": 2576489.0,
            "deskripsi": "kayu sebagai bahan"
        }
    ],
    "perdana": [
        {
            "keywords": [
                "menteri", "kabinet", "jabatan", "memimpin",
                "perdana menteri", "ketua menteri", "pm",
                "pemerintahan", "dilantik", "parlemen"
            ],
            "pos_target": "NOUN",
            "synset_id": 1012855.0,
            "deskripsi": "perdana menteri (ketua menteri)"
        },
        {
            "keywords": [
                "kali", "pertama", "untuk pertama", "pertama kali",
                "perdana kalinya", "debut", "premier", "mulai"
            ],
            "pos_target": "ADJ",
            "synset_id": 3012209.0,
            "deskripsi": "pertama (adjektiva)"
        }
    ]
}

# Default sense berdasarkan frekuensi di data latih
default_synset = {}
for lemma in RULES.keys():
    lemma_train = df[df["lemma"] == lemma]
    if not lemma_train.empty:
        most_common = lemma_train["synset_id"].value_counts().idxmax()
        default_synset[lemma] = most_common
    else:
        default_synset[lemma] = None

# Deskripsi untuk setiap synset
deskripsi_synset = {}
for lemma, rules in RULES.items():
    for rule in rules:
        deskripsi_synset[rule["synset_id"]] = rule["deskripsi"]
for syn in default_synset.values():
    if syn not in deskripsi_synset:
        deskripsi_synset[syn] = "Makna lain (default)"

# -------------------------------------------------------------------------
# FUNGSI POS TAGGING & PREDIKSI
# -------------------------------------------------------------------------
def get_pos_tags(kalimat):
    doc = nlp(kalimat)
    tags = []
    for sent in doc.sentences:
        for word in sent.words:
            tags.append((word.text, word.upos))
    return tags

def predict_synset(sentence, lemma):
    tagged = get_pos_tags(sentence)

    # Cari posisi lemma pertama
    lemma_pos = None
    for i, (word, tag) in enumerate(tagged):
        if word.lower() == lemma.lower():
            lemma_pos = i
            break

    if lemma_pos is None:
        return default_synset.get(lemma), "default (lemma tidak ditemukan)"

    pos_target = tagged[lemma_pos][1]
    lower_sent = sentence.lower()
    rules = RULES.get(lemma, [])

    for rule in rules:
        keyword_match = any(kw in lower_sent for kw in rule["keywords"])
        pos_ok = True
        if rule.get("pos_target") is not None:
            pos_ok = (pos_target == rule["pos_target"])

        if keyword_match and pos_ok:
            return rule["synset_id"], f"Aturan: kata kunci + POS '{pos_target}'"

    # Fallback ke default
    return default_synset.get(lemma), f"Default (tidak ada aturan cocok, POS lemma = '{pos_target}')"

# -------------------------------------------------------------------------
# UI UTAMA
# -------------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="main-title">🧠 Word Sense Disambiguation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Metode <b>Rule‑based</b> + POS Tagging</div>', unsafe_allow_html=True)

col_input1, col_input2 = st.columns([3, 1])
with col_input1:
    kalimat = st.text_input(
        "📝 Masukkan kalimat:",
        placeholder="Contoh: Bagian atas kue pengantin itu sangat indah.",
        key="input_kalimat"
    )
with col_input2:
    lemma = st.selectbox("🎯 Pilih lemma:", list(RULES.keys()))

col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
with col_btn2:
    prediksi_clicked = st.button("✨ Prediksi Makna")

if prediksi_clicked:
    if not kalimat.strip():
        st.warning("⚠️ Silakan masukkan kalimat terlebih dahulu.")
    else:
        synset_id, keterangan = predict_synset(kalimat.strip(), lemma)
        if synset_id is None:
            st.error("Makna tidak dapat ditentukan (lemma tidak ada dalam data latih).")
        else:
            deskripsi = deskripsi_synset.get(synset_id, "Tidak ada deskripsi")

            st.markdown(f"""
            <div class="result-card">
                <div style="font-size: 1rem; color: #888;">📌 Makna Terprediksi</div>
                <div class="sense-id">{synset_id}</div>
                <div class="sense-desc">📘 {deskripsi}</div>
                <div class="sense-info">ℹ️ {keterangan}</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🔎 Lihat POS Tagging Kalimat"):
                tagged = get_pos_tags(kalimat)
                df_tags = pd.DataFrame(tagged, columns=["Token", "POS Tag"])
                st.dataframe(df_tags, use_container_width=True, hide_index=True)

# -------------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("📚 Daftar Makna & Aturan (POS Penuh)")
    for lem, rules in RULES.items():
        with st.expander(f"**{lem}** ({len(rules)} aturan)"):
            for r in rules:
                pos_str = f"`{r['pos_target']}`" if r['pos_target'] else "—"
                keywords_preview = ', '.join(r['keywords'][:10])
                if len(r['keywords']) > 10:
                    keywords_preview += '...'
                st.markdown(f"""
                <div style="margin-bottom:10px; padding:10px; background:#1f77b4; border-radius:8px;">
                    <b>Synset ID:</b> <code>{r['synset_id']}</code><br>
                    <b>Deskripsi:</b> {r['deskripsi']}<br>
                    <b>Kata Kunci:</b> <small>{keywords_preview}</small><br>
                    <b>POS Target:</b> {pos_str}
                </div>
                """, unsafe_allow_html=True)
    st.caption("Default sense diambil dari frekuensi tertinggi di data latih.")