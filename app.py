# =============================================================================
# APLIKASI DEMO WSD RULE‑BASED + POS TAGGING (Streamlit + Stanza)
# DENGAN PERBAIKAN ATURAN UNTUK "ATAS" & "ALAM"
# =============================================================================
import streamlit as st
import pandas as pd
import stanza

# -------------------------------------------------------------------------
# 0. KONFIGURASI HALAMAN
# -------------------------------------------------------------------------
st.set_page_config(page_title="Demo WSD", layout="centered")
st.title("🔍 Demo Word Sense Disambiguation (WSD)")
st.markdown("**Metode:** Rule‑based + POS Tagging (Bahasa Indonesia) **dengan perbaikan aturan**")

# -------------------------------------------------------------------------
# 1. LOAD MODEL STANZA & DATASET
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
# 2. DEFINISI ATURAN YANG SUDAH DISEMPURNAKAN
# -------------------------------------------------------------------------
RULES = {
    "alam": [
        {
            "keywords": ["semesta", "galaksi", "planet", "bintang", "kosmos", "jagat raya"],
            "pos_target": None,
            "synset_id": 9466280.0,
            "deskripsi": "dunia; alam semesta"
        },
        {
            "keywords": [
                "hutan", "liar", "cagar", "lingkungan", "fenomena", "bebas",
                "ibu alam", "hukum alam", "ekosistem", "kehidupan", "alam sekitar",
                "keseimbangan alam", "di alam", "oleh alam", "dengan alam"
            ],
            "pos_target": None,
            "synset_id": 9366762.0,
            "deskripsi": "lingkungan kehidupan / alam sekitar"
        }
    ],
    "anggur": [
        {
            "keywords": ["minum", "botol", "gelas", "fermentasi", "mabuk", "merah", "putih"],
            "pos_target": None,
            "synset_id": 7758680.0,
            "deskripsi": "minuman fermentasi (anggur)"
        },
        {
            "keywords": ["buah", "kebun", "pohon", "manis"],
            "pos_target": None,
            "synset_id": 1190840.0,
            "deskripsi": "buah anggur"
        }
    ],
    "atas": [
        {
            # Aturan untuk "bagian atas" (nomina) – didahulukan dan tidak pakai POS
            "keywords": ["bagian atas", "sisi atas", "pihak atas", "orang atas", "lapisan atas"],
            "pos_target": None,          # tidak peduli POS
            "synset_id": 5037617.0,
            "deskripsi": "bagian atas (nomina)"
        },
        {
            # Aturan untuk preposisi "di atas", "ke atas", "dari atas"
            "keywords": ["di atas", "ke atas", "dari atas"],
            "pos_target": None,          # langsung kunci frasa, bukan POS
            "synset_id": 125993.0,
            "deskripsi": "bagian yang lebih tinggi (preposisi)"
        }
    ],
    "dasar": [
        {
            "keywords": ["sekolah", "pendidikan", "guru", "kelas", "siswa", "belajar"],
            "pos_target": None,
            "synset_id": 901060.0,
            "deskripsi": "pokok/asas (pendidikan dasar)"
        },
        {
            "keywords": ["pemikiran", "argumen", "landasan", "prinsip", "filosofi", "teori"],
            "pos_target": None,
            "synset_id": 1856419.0,
            "deskripsi": "landasan pemikiran"
        }
    ],
    "kayu": [
        {
            "keywords": ["pohon", "hutan", "batang", "tebang", "illegal", "tanaman"],
            "pos_target": None,
            "synset_id": 14943580.0,
            "deskripsi": "pohon penghasil kayu"
        },
        {
            "keywords": ["meja", "kursi", "papan", "tongkat", "peti", "kotak", "pensil"],
            "pos_target": None,
            "synset_id": 2576489.0,
            "deskripsi": "kayu sebagai bahan"
        }
    ],
    "perdana": [
        {
            "keywords": ["menteri", "kabinet", "jabatan", "memimpin"],
            "pos_target": "NOUN",
            "synset_id": 1012855.0,
            "deskripsi": "perdana menteri (ketua menteri)"
        },
        {
            "keywords": ["kali", "pertama"],
            "pos_target": "ADJ",
            "synset_id": 3012209.0,
            "deskripsi": "pertama (adjektiva)"
        }
    ]
}

# Default sense berdasarkan data latih
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
# 3. FUNGSI POS TAGGING (STANZA)
# -------------------------------------------------------------------------
def get_pos_tags(kalimat):
    """Kembalikan list (token_text, upos_tag) dari kalimat."""
    doc = nlp(kalimat)
    tags = []
    for sent in doc.sentences:
        for word in sent.words:
            tags.append((word.text, word.upos))
    return tags

# -------------------------------------------------------------------------
# 4. FUNGSI PREDIKSI MAKNA (DISEMPURNAKAN)
# -------------------------------------------------------------------------
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

    pos_target = tagged[lemma_pos][1]  # tetap diambil untuk lemma lain
    lower_sent = sentence.lower()
    rules = RULES.get(lemma, [])

    for rule in rules:
        # Cek kata kunci (cocok jika salah satu keyword muncul sebagai substring)
        keyword_match = any(kw in lower_sent for kw in rule["keywords"])

        # Cek POS target hanya jika disyaratkan
        pos_ok = True
        if rule.get("pos_target") is not None:
            pos_ok = (pos_target == rule["pos_target"])

        if keyword_match and pos_ok:
            return rule["synset_id"], f"aturan: keyword + POS '{pos_target}'"

    # Fallback ke default
    return default_synset.get(lemma), f"default (POS lemma: {pos_target})"

# -------------------------------------------------------------------------
# 5. UI STREAMLIT
# -------------------------------------------------------------------------
st.subheader("Masukkan Kalimat & Pilih Lemma Target")

kalimat = st.text_input("Kalimat:", placeholder="Contoh: Bagian atas kue pengantin itu indah sekali.")
lemma = st.selectbox("Lemma target:", list(RULES.keys()))

if st.button("Prediksi Makna"):
    if not kalimat.strip():
        st.warning("Silakan masukkan kalimat terlebih dahulu.")
    else:
        synset_id, keterangan = predict_synset(kalimat.strip(), lemma)
        if synset_id is None:
            st.error("Makna tidak dapat ditentukan (lemma tidak ada dalam data latih).")
        else:
            deskripsi = deskripsi_synset.get(synset_id, "Tidak ada deskripsi")
            st.success(f"**Makna Terprediksi:** `{synset_id}`")
            st.info(f"📘 **Deskripsi:** {deskripsi}")
            st.caption(f"ℹ️ **Keterangan:** {keterangan}")

            with st.expander("Lihat POS Tagging Kalimat"):
                tagged = get_pos_tags(kalimat)
                st.write(tagged)

# -------------------------------------------------------------------------
# 6. SIDEBAR
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("📚 Daftar Makna & Aturan")
    for lem, rules in RULES.items():
        st.markdown(f"**{lem}**")
        for r in rules:
            st.markdown(f"- `{r['synset_id']}` → {r['deskripsi']}")
    st.divider()
    st.markdown("""
    **Perbaikan Aturan Utama:**
    - **atas** : menggunakan frasa lengkap (`bagian atas`, `di atas`) tanpa bergantung POS.
    - **alam** : menambah kata kunci `ibu alam`, `di alam`, `ekosistem`, dll.
    """)
    st.caption("Default sense diambil dari frekuensi tertinggi di data latih.")
