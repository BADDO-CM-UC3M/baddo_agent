import os
import re
import json
from typing import TypedDict, Annotated, Sequence, List, Dict, Tuple
from contextlib import contextmanager
import operator

import spacy
from langdetect import detect as langdetect_detect
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage
import psycopg2
from openai import OpenAI
from ollama import Client as OllamaClient

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found")

EMBEDDING_MODEL = "text-embedding-3-large"
GENERATION_MODEL = "gpt-5-mini"
CLASSIFICATION_MODEL = "gpt-5-nano"
OLLAMA_MODEL = "qwen3:8b"

OLLAMA_HOST = os.getenv("OLLAMA_HOST")

PG_DBNAME = os.getenv("PG_DBNAME")
PG_USER = os.getenv("PG_USER")
PG_PASS = os.getenv("PG_PASS")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")

TOP_K = 10
TOP_K_RERANK = 10
MIN_LENGTH = 200
MAX_ITERATIONS = 2

FRAGMENTS_PER_GAP = 3

SEMANTIC_WEIGHT = 0.7
KEYWORDS_WEIGHT = 0.3
BOOST_PER_LATIN_TERM = 0.25
BOOST_PER_PERSON = 0.3
BOOST_PER_PLACE = 0.1
MAX_NER_BOOST = 0.50

BIOGRAPHIES_PATH = "biografias.json"
GLOSSARY_PATH = "glosario_traducido.json"
CURRENTS_PATH = "corrientes.json"

try:
    with open("latin_words.txt", encoding="utf-8") as f:
        LATIN_WORDS = {line.strip().lower() for line in f if line.strip()}
except FileNotFoundError:
    LATIN_WORDS = set()
    print("latin_words.txt not found, will not be applied")

try:
    with open("lemario.txt", encoding="utf-8") as f:
        SPANISH_WORDS = {line.strip().lower() for line in f if line.strip()}
except FileNotFoundError:
    SPANISH_WORDS = set()

_LANGUAGE_MODELS = {
    "es": "es_core_news_sm",
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "it": "it_core_news_sm",
    "fr": "fr_core_news_sm",
}

SPACY_MODELS: Dict[str, spacy.language.Language] = {}

for _lang, _model in _LANGUAGE_MODELS.items():
    try:
        SPACY_MODELS[_lang] = spacy.load(_model)
        print(f"spaCy model loaded: {_model}")
    except OSError:
        print(f"WARNING: Model '{_model}' not found. "
              f"Install it with: python -m spacy download {_model}")

TRANSLATED_GLOSSARY: Dict = {}
BIOGRAPHIES: Dict = {}
CURRENTS: Dict = {}

def load_resources():
    global TRANSLATED_GLOSSARY, BIOGRAPHIES, CURRENTS

    try:
        with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
            content = re.sub(r"[\t\r\x00-\x08\x0b-\x0c\x0e-\x1f]", " ", f.read())
            TRANSLATED_GLOSSARY = json.loads(content)
        print(f"Translated glossary: {len(TRANSLATED_GLOSSARY)} terms")
    except Exception as e:
        print(f"Translated glossary error: {e}")

    try:
        with open(BIOGRAPHIES_PATH, "r", encoding="utf-8") as f:
            content = re.sub(r"[\t\r\x00-\x08\x0b-\x0c\x0e-\x1f]", " ", f.read())
            BIOGRAPHIES = json.loads(content)
        print(f"Biographies: {len(BIOGRAPHIES)} entries")
    except Exception as e:
        print(f"Biographies error: {e}")

    try:
        with open(CURRENTS_PATH, "r", encoding="utf-8") as f:
            content = re.sub(r"[\t\r\x00-\x08\x0b-\x0c\x0e-\x1f]", " ", f.read())
            CURRENTS = json.loads(content)
        print(f"Currents: {len(CURRENTS)} entries")
    except Exception as e:
        print(f"Currents error: {e}")


load_resources()
client = OpenAI(api_key=OPENAI_API_KEY)
ollama_client = OllamaClient(host=OLLAMA_HOST)

_DB_AUTHORS: List[str] = []


def load_db_authors():
    global _DB_AUTHORS
    if _DB_AUTHORS:
        return
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT autor FROM chunks_w_context WHERE autor IS NOT NULL;")
            _DB_AUTHORS = [row[0] for row in cur.fetchall() if row[0]]
            print(f"Authors in DB: {len(_DB_AUTHORS)}")
    except Exception as e:
        print(f"Error loading DB authors: {e}")
        _DB_AUTHORS = []


def detect_db_authors(text: str) -> List[str]:

    load_db_authors()
    text_lower = text.lower()
    matches = []

    for author in _DB_AUTHORS:
        author_lower = author.lower()

        if author_lower in text_lower:
            matches.append(author)
            continue

        parts = author_lower.split()

        matched = False
        if len(parts) >= 2:
            for length in range(len(parts), 1, -1):
                for start in range(len(parts) - length + 1):
                    fragment = " ".join(parts[start:start + length])
                    if len(fragment) > 4 and fragment in text_lower:
                        matches.append(author)
                        matched = True
                        break
                if matched:
                    break

        if matched:
            continue

        for part in parts:
            part_norm = re.sub(r"[''`’‘´]", "", part)
            if len(part_norm) > 5 and part_norm in text_lower:
                matches.append(author)
                break
            if len(part) > 5 and part in text_lower:
                matches.append(author)
                break

    result = list(dict.fromkeys(matches))
    if result:
        print(f"DB authors detected: {result}")
    else:
        print(f"DB authors: none detected")
    return result


def detect_biography_authors(text: str) -> List[str]:

    text_lower = text.lower()
    matches = []

    for author in BIOGRAPHIES.keys():
        author_lower = author.lower()

        if author_lower in text_lower:
            matches.append(author)
            continue

        parts = author_lower.split()

        matched = False
        if len(parts) >= 2:
            for length in range(len(parts), 1, -1):
                for start in range(len(parts) - length + 1):
                    fragment = " ".join(parts[start:start + length])
                    if len(fragment) > 4 and fragment in text_lower:
                        matches.append(author)
                        matched = True
                        break
                if matched:
                    break

        if matched:
            continue

        for part in parts:
            part_norm = re.sub(r"[''`’‘´]", "", part)
            if len(part_norm) > 5 and part_norm in text_lower:
                matches.append(author)
                break
            if len(part) > 5 and part in text_lower:
                matches.append(author)
                break

    result = list(dict.fromkeys(matches))
    if result:
        print(f"Bibliography authors detected: {result}")
    else:
        print(f"Bibliography authors: none detected")
    return result

def extract_year_ranges(text: str) -> List[Dict]:

    prompt = f"""Analiza este texto y extrae SOLO referencias temporales que indican CUÁNDO FUERON ESCRITAS O PUBLICADAS las fuentes académicas, no fechas históricas de eventos.

TEXTO: {text}

REGLAS:
- INCLUIR: referencias a cuándo publicaron los autores
  Ejemplos: "período franquista (1939-1975)", "textos del año 1936", "publicaciones de 1950",
  "autores del siglo XX", "época democrática (1975-actualidad)", "del siglo veinte"
- EXCLUIR: fechas de eventos históricos
  Ejemplos: "año 585 d.C.", "III Concilio de Toledo del año 589", "reinado de Leovigildo", "siglo VI"
- "actualidad", "hoy", "presente" = 2025
- Solo años entre 1900 y 2099
- Si es un año exacto ("textos de 1936", "publicaciones de 1950"), año_min y año_max deben ser IGUALES y exacto = true
- Si es un rango ("1939-1975", "siglo XX"), exacto = false

Ejemplos de respuesta:
- "textos del año 1936" →
  {{"rangos": [{{"año_min": 1936, "año_max": 1936, "etiqueta": "1936", "exacto": true}}]}}
- "franquismo (1939-1975) y democracia (1975-actualidad)" →
  {{"rangos": [{{"año_min": 1939, "año_max": 1975, "etiqueta": "franquismo", "exacto": false}}, {{"año_min": 1975, "año_max": 2025, "etiqueta": "democracia", "exacto": false}}]}}
- "publicaciones entre 1985 y 2010" →
  {{"rangos": [{{"año_min": 1985, "año_max": 2010, "etiqueta": "1985-2010", "exacto": false}}]}}
- "autores del siglo XX" →
  {{"rangos": [{{"año_min": 1900, "año_max": 1999, "etiqueta": "siglo XX", "exacto": false}}]}}
- "solo hay fechas históricas como 585 d.C." →
  {{"rangos": []}}

Responde ÚNICAMENTE con JSON:"""

    try:
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json"
        )
        data = json.loads(response["message"]["content"])
        raw_ranges = data.get("rangos", [])

        valid_ranges = []
        for r in raw_ranges:
            year_min = r.get("año_min")
            year_max = r.get("año_max")
            if (year_min and year_max and
                    1900 <= year_min <= 2099 and
                    1900 <= year_max <= 2099):
                exact = r.get("exacto", year_min == year_max)
                valid_ranges.append({
                    "year_min": year_min,
                    "year_max": year_max,
                    "label": r.get("etiqueta", f"{year_min}-{year_max}"),
                    "exact": exact,
                })

        if valid_ranges:
            summary = [
                (r["label"], r["year_min"], "=" if r["exact"] else f"–{r['year_max']}")
                for r in valid_ranges
            ]
            print(f"Academic ranges detected: {summary}")

        return valid_ranges

    except Exception as e:
        print(f"Error extracting ranges with Ollama: {e}")
        return []


def build_year_condition(ranges: List[Dict]) -> tuple:

    if not ranges:
        return "", []

    parts = []
    params = []
    for r in ranges:
        if r.get("exact", False):
            parts.append("(año ~ '^\\d+$' AND año::integer = %s)")
            params.append(r["year_min"])
        else:
            parts.append("(año ~ '^\\d+$' AND año::integer >= %s AND año::integer <= %s)")
            params += [r["year_min"], r["year_max"]]

    cond = " AND (" + " OR ".join(parts) + ")"
    return cond, params


def build_db_authors_condition(authors: List[str]) -> tuple:

    if not authors:
        return "", []

    placeholders = ", ".join(["%s"] * len(authors))
    cond = f" AND autor IN ({placeholders})"
    return cond, authors

def detect_language(text: str) -> str:
    try:
        lang = langdetect_detect(text)
        return lang if lang in _LANGUAGE_MODELS else "es"
    except Exception:
        return "es"


def lemmatize_text(text: str, lang: str = "es") -> set:
    nlp = SPACY_MODELS.get(lang)
    if nlp is None:
        return set(re.findall(r"\b\w+\b", text.lower()))

    doc = nlp(text.lower())
    return {
        token.lemma_
        for token in doc
        if not token.is_punct and not token.is_space and len(token.text) > 2
    }


def extract_words(text: str) -> List[str]:
    return re.findall(r"\b[a-záéíóúñ]+\b", text.lower())


def latin_words_in_text(text: str) -> List[str]:
    if not LATIN_WORDS or not SPANISH_WORDS:
        return []
    return [w for w in extract_words(text) if w in LATIN_WORDS and w not in SPANISH_WORDS]


def normalize_for_filter(text: str) -> List[str]:
    import unicodedata
    def remove_accents(s: str) -> str:
        return "".join(
            c for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )
    original = text.lower()
    without_accents = remove_accents(original)
    if without_accents != original:
        return [original, without_accents]
    return [original]


def extract_ner_entities(text: str) -> Dict[str, List[str]]:
    LEADING_PUNCTUATION = "¿?¡!.,;:()"
    FUNCTION_WORDS = {
        "según", "segun", "para", "sobre", "desde", "hasta", "entre",
        "ante", "bajo", "con", "contra", "de", "del", "desde", "en",
        "hacia", "mediante", "por", "sin", "sobre", "tras", "el", "la",
        "los", "las", "un", "una", "unos", "unas", "al", "que", "y",
        "o", "e", "ni", "pero", "sino", "aunque", "como", "cuando",
        "where", "the", "a", "an", "of", "in", "on", "at", "to", "for",
        "qué", "que", "quién", "quien", "quiénes", "quienes",
        "cómo", "como", "cuándo", "cuando", "dónde", "donde",
        "cuál", "cual", "cuáles", "cuales", "cuánto", "cuanto",
    }
    SPAN_CUT_CHARS = set("¿?¡!")

    def clean_span(name: str) -> str:
        for i, c in enumerate(name):
            if c in SPAN_CUT_CHARS:
                name = name[:i]
                break
        tokens = name.strip().split()
        while tokens and tokens[0].lower().strip(LEADING_PUNCTUATION) in FUNCTION_WORDS:
            tokens = tokens[1:]
        while tokens and tokens[-1].lower().strip(LEADING_PUNCTUATION) in FUNCTION_WORDS:
            tokens = tokens[:-1]
        return " ".join(tokens)

    lang = detect_language(text)
    nlp = SPACY_MODELS.get(lang)

    persons: List[str] = []
    places: List[str] = []

    if nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            raw_name = ent.text.strip()

            first_real_token = None
            for _t in raw_name.split():
                _t_clean = _t.lstrip(LEADING_PUNCTUATION)
                if _t_clean and _t_clean.lower() not in FUNCTION_WORDS:
                    first_real_token = _t_clean
                    break

            if first_real_token is None or not first_real_token[0].isupper():
                continue

            name = clean_span(raw_name.lower())
            if len(name) <= 2 or name.strip(LEADING_PUNCTUATION) in FUNCTION_WORDS:
                continue
            if any(c in name for c in SPAN_CUT_CHARS):
                continue
            authors_lower = {a.lower() for a in BIOGRAPHIES.keys()}
            is_author = (
                name in authors_lower or
                (len(name) > 5 and any(name in a for a in authors_lower))
            )
            if is_author:
                continue

            if ent.label_ in ("PER", "PERSON"):
                persons.append(name)
            elif ent.label_ in ("LOC", "GPE", "FAC"):
                places.append(name)
    else:
        candidates = re.findall(r"(?<=[.!?]\s|;|\n)[A-ZÁÉÍÓÚ][a-záéíóú]+|(?<=\s)[A-ZÁÉÍÓÚ][a-záéíóú]+", text)
        persons = list({c.lower() for c in candidates if len(c) > 2})

    persons = list(dict.fromkeys(p for p in persons if p))
    places = list(dict.fromkeys(l for l in places if l))

    return {"persons": persons, "places": places}

def find_available_terms(text: str) -> Dict[str, str]:
    """Searches for glossary terms"""
    terms: Dict[str, str] = {}
    text_lower = text.lower()
    text_words = set(re.findall(r"\b\w+\b", text_lower))

    lang = detect_language(text)
    text_lemmas = lemmatize_text(text, lang=lang)

    for term_es, translations in TRANSLATED_GLOSSARY.items():
        definition_es = translations["es"]["definicion"]
        term_es_lower = term_es.lower()
        search_term = translations.get(lang, translations["es"])["termino"]

        if term_es_lower in text_lower:
            terms[term_es] = definition_es
            continue

        if search_term in text_lower:
            terms[term_es] = definition_es
            continue

        found = False
        for variant in {term_es_lower, search_term}:
            variant_words = variant.split()
            if len(variant_words) > 1:
                surname = variant_words[-1]
                if len(surname) > 3 and surname in text_words:
                    terms[term_es] = definition_es
                    found = True
                    break
        if found:
            continue

        term_words_set = set(re.findall(r"\b\w+\b", search_term))
        if len(term_words_set) > 1 and term_words_set.issubset(text_words):
            terms[term_es] = definition_es
            continue

        if SPACY_MODELS.get(lang) is not None:
            term_lemmas = lemmatize_text(search_term, lang=lang)

            if term_lemmas and term_lemmas.issubset(text_lemmas):
                terms[term_es] = definition_es
                continue

            if len(term_lemmas) == 1:
                if next(iter(term_lemmas)) in text_lemmas:
                    terms[term_es] = definition_es

    return terms

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    original_question: str
    sub_questions: List[str]
    main_keywords: List[str]
    rag_context: str
    rag_documents: List[Dict]
    reranked_documents: List[Dict]
    seen_texts: List[str]
    latin_terms: List[str]
    technical_terms: List[str]
    proper_names: List[str]
    question_entities: Dict[str, List[str]]
    filter_terms: List[str]
    mentioned_authors: List[str]
    year_ranges: List[Dict]
    db_authors_filter: List[str]
    needs_glossary: bool
    glossary_definitions: Dict[str, str]
    needs_bibliography: bool
    bibliography_authors: Dict[str, str]
    needs_currents: bool
    currents_definitions: Dict[str, str]

    # PHASE 1: INITIAL ANSWER (RAG with the question only + top-K)
    initial_phase_paragraphs: List[Dict]
    initial_answer: str

    # PHASE 2: CONTEXTUALIZATION (sub-questions + extra paragraphs that are ONLY
    # used if they provide necessary context without drifting from the question)
    contextualization_phase_paragraphs: List[Dict]
    contextualized_answer: str

    # PHASE 3: REFINEMENT (iterative gaps)
    refinements: List[Dict]      # one per cycle: answer, gaps, fragments
    current_answer: str
    is_complete: bool
    identified_gaps: List[str]
    current_cycle: int
    final_answer: str

@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=PG_DBNAME, user=PG_USER, password=PG_PASS,
            host=PG_HOST, port=PG_PORT
        )
        yield conn
    except Exception as e:
        print(f"DB error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def prepare_question(state: AgentState) -> AgentState:

    print("\n" + "="*80)
    print("NODE 1: PREPARATION (keywords, NER, latin, years, DB authors)")
    print("="*80)

    question = state["original_question"]
    print(f"Question: {question}")

    keywords = expand_query_with_llm(question)
    print(f"\nExtracted keywords: {keywords}")
    state["main_keywords"] = keywords

    latin_terms = latin_words_in_text(question)
    state["latin_terms"] = latin_terms
    if latin_terms:
        print(f"Latin terms: {latin_terms}")

    entities = extract_ner_entities(question)
    state["question_entities"] = entities
    if entities["persons"]:
        print(f"Persons (NER): {entities['persons']}")
    if entities["places"]:
        print(f"Places (NER): {entities['places']}")

    mentioned_authors = detect_biography_authors(question)
    state["mentioned_authors"] = mentioned_authors

    all_entities = entities.get("persons", []) + entities.get("places", [])
    filter_terms = list(dict.fromkeys(
        latin_terms +
        [a.lower() for a in mentioned_authors] +
        all_entities
    ))
    state["filter_terms"] = filter_terms
    print(f"🔎 DB filter (text): {filter_terms}")

    year_ranges = extract_year_ranges(question)
    state["year_ranges"] = year_ranges

    db_authors = detect_db_authors(question)
    state["db_authors_filter"] = db_authors

    state["seen_texts"] = []

    return state


def rag_main_question(state: AgentState) -> AgentState:

    print("\n" + "="*80)
    print("NODE 2: MAIN QUESTION RAG")
    print("="*80)

    question     = state["original_question"]
    keywords     = state.get("main_keywords", [])
    filter_terms = state.get("filter_terms", [])
    year_ranges  = state.get("year_ranges", [])
    db_authors   = state.get("db_authors_filter", [])

    print(f"\nSearching: {question}...")
    entities          = state.get("question_entities", {"persons": [], "places": []})
    mentioned_authors = state.get("mentioned_authors", [])
    entities_with_authors = {**entities, "authors": [a.lower() for a in mentioned_authors]}

    docs = semantic_similarity_search(
        query=question,
        top_k=TOP_K,
        filter_terms=filter_terms,
        entities=entities_with_authors,
        keywords=keywords,
        year_ranges=year_ranges,
        db_authors=db_authors,
    )

    state["rag_documents"] = docs
    print(f"\nTotal main question docs: {len(docs)}")
    return state


def generate_sub_questions(state: AgentState) -> AgentState:

    print("\n" + "="*80)
    print("NODE 3: SUB-QUESTION GENERATION (about the initial answer)")
    print("="*80)

    question       = state["original_question"]
    initial_answer = state.get("initial_answer", "")

    prompt = f"""Eres un especialista en historiografía de la Hispania visigoda.

## TAREA

Te paso la pregunta original y una RESPUESTA INICIAL ya generada con los párrafos más directamente relevantes. Tu única tarea es proponer sub-preguntas que sirvan para localizar el CONTEXTO (marco) que esa respuesta necesita para quedar bien encuadrada, NUNCA para añadir datos concretos como fechas, nombres, cifras o eventos puntuales (eso es trabajo de la siguiente fase de refinamiento).

## QUÉ SE ENTIENDE POR CONTEXTO AQUÍ

- Marco histórico-político general en el que ocurre lo que la pregunta plantea (situación de la monarquía, estructuras de poder, relaciones entre poderes).
- Marco religioso, eclesiástico o doctrinal de la época si la pregunta lo toca.
- Encuadre geográfico (provincia, sede, frontera) sólo a nivel descriptivo, no para listar nombres exactos.
- Encuadre social o económico si es necesario para entender la pregunta.
- Antecedentes o consecuencias inmediatas que ayuden a entender el porqué de lo preguntado.

## QUÉ NO ES CONTEXTO (y por tanto NO debes preguntarlo)

- Fechas exactas, años de reinado, datas de concilios.
- Nombres concretos de obispos, reyes, autores, ciudades.
- Cifras, cantidades, números de cánones.
- Cualquier dato puntual y verificable: eso lo busca el refinamiento, no esta fase.

## REGLAS

- Parte siempre de la RESPUESTA INICIAL: identifica qué necesita encuadrarse para que el lector la entienda mejor sin desviarse del tema.
- No repitas información que la respuesta inicial ya cubre.
- No introduzcas temas ajenos a la pregunta original.
- Si la respuesta inicial ya está bien encuadrada, devuelve pocas sub-preguntas (incluso 0).
- Las sub-preguntas se usarán en un RAG semántico, así que formula de forma que existan párrafos en libros que las respondan.

**Pregunta original:** {question}

**Respuesta inicial:**
{initial_answer}

### Estrategia de división:
- Si la pregunta solicita un **dato puntual** (una fecha, un nombre, un hecho específico): genera **máximo 3 sub-preguntas directas**
- Si requiere **análisis amplio** (contexto, evolución, comparación, interpretación): genera **tantas sub-preguntas como sean necesarias** para cubrir todos los aspectos relevantes

## FORMATO DE SALIDA

Devuelve únicamente las sub-preguntas, una por línea, sin numeración, viñetas ni comentarios adicionales.
"""

    try:
        response = client.chat.completions.create(
            model=CLASSIFICATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=1
        )

        text = response.choices[0].message.content.strip()
        sub_questions = []

        for line in text.split("\n"):
            line = line.strip()
            line = re.sub(r"^\d+[.)\]\s]*", "", line)
            line = re.sub(r"^[-•]\s*", "", line)
            if line and "?" in line:
                sub_questions.append(line)

        if not sub_questions:
            sub_questions = [question]

        print(f"\nGenerated {len(sub_questions)} sub-questions:")
        for i, sp in enumerate(sub_questions, 1):
            print(f"   {i}. {sp}")

        state["sub_questions"] = sub_questions

    except Exception as e:
        print(f"Error: {e}")
        state["sub_questions"] = [question]

    return state


def expand_query_with_llm(question: str) -> List[str]:
    """Extracts keywords that appear literally in the question"""

    prompt = f"""De las siguientes palabras que aparecen en la pregunta, selecciona las MAS IMPORTANTES para buscar información relevante.

PREGUNTA: {question}

INSTRUCCIONES:
1. Selecciona SOLO palabras que aparecen LITERALMENTE en la pregunta
2. NO inventes palabras ni sinónimos
3. Prioriza:
   - Nombres propios (personas, lugares)
   - Términos técnicos o históricos específicos
   - Conceptos clave
4. EXCLUYE palabras genéricas (que, como, cuál, efecto, etc.)
5. Máximo 6 palabras
5. Las palabras deben ser REALMENTE importantes

FORMATO: palabras separadas por comas (usa EXACTAMENTE como aparecen en la pregunta, respeta mayúsculas/minúsculas).

Ejemplos:
 - Pregunta: ¿Qué efecto tuvo en la provincia visigoda de la Gallaecia la conversión al catolicismo de Recaredo?
    Keywords importantes: Gallaecia, conversión, catolicismo, Recaredo
 - Pregunta: ¿A qué se refiere José Orlandis Rovira cuando habla de la "pre-conversión"?
    Keywords importantes: pre-conversión, José Orlandis Rovira
 - Pregunta: Según Isidro Gomá, ¿qué momentos del catolicismo propiciaron el crecimiento y la expansión de la historia política de España?
    Keywords importantes: Isidro Gomá, catolicismo, crecimiento, expansión, historia política de España

Keywords importantes:"""

    try:
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response["message"]["content"].strip()
        candidate_keywords = [k.strip() for k in raw.split(",") if k.strip()]

        question_lower = question.lower()
        valid_keywords = []

        for kw in candidate_keywords:
            kw_clean = re.sub(r"[.,;:!?]+$", "", kw)
            if kw_clean.lower() in question_lower and len(kw_clean) > 2:
                valid_keywords.append(kw_clean.lower())

        return valid_keywords

    except Exception as e:
        print(f"Error extracting keywords with Ollama: {e}")
        words = re.findall(r"\b[A-ZÁÉÍÓÚa-záéíóúñ]{4,}\b", question)
        stopwords = {"que", "como", "cual", "donde", "cuando", "quien", "para",
                     "sobre", "entre", "desde", "hasta", "tuvo", "fue", "esta", "ese", "esa"}
        return [p.lower() for p in words if p.lower() not in stopwords]


def build_filter_terms_for(
    text: str,
    base_terms: List[str],
    base_ranges: List[Dict] = None,
    base_db_authors: List[str] = None,
) -> Tuple[List[str], List[Dict], List[str]]:

    base_ranges     = base_ranges or []
    base_db_authors = base_db_authors or []

    own_latin    = latin_words_in_text(text)
    own_authors  = detect_biography_authors(text)
    own_entities = extract_ner_entities(text)
    entity_list  = (
        own_entities.get("persons", []) +
        own_entities.get("places", [])
    )
    combined_terms = list(dict.fromkeys(
        base_terms + own_latin +
        [a.lower() for a in own_authors] +
        entity_list
    ))
    new_terms = [t for t in combined_terms if t not in base_terms]
    if new_terms:
        print(f"   New own filters: {new_terms}")

    db_authors = detect_db_authors(text)

    ranges = extract_year_ranges(text)

    return combined_terms, ranges, db_authors


def rag_sub_questions(state: AgentState) -> AgentState:

    print("\n" + "="*80)
    print("NODE 4: SUB-QUESTION RAG")
    print("="*80)

    sub_questions     = state.get("sub_questions", [state["original_question"]])
    keywords          = state.get("main_keywords", [])
    base_filter_terms = state.get("filter_terms", [])
    base_year_ranges  = state.get("year_ranges", [])
    base_db_authors   = state.get("db_authors_filter", [])
    existing_docs     = state.get("rag_documents", [])
    existing_texts    = {d["text"] for d in existing_docs}

    new_docs = []

    for i, sp in enumerate(sub_questions, 1):
        print(f"\nSub-question {i}/{len(sub_questions)}: {sp}...")

        sp_filter_terms, sp_ranges, sp_db_authors = build_filter_terms_for(
            sp,
            base_filter_terms,
            base_ranges=base_year_ranges,
            base_db_authors=base_db_authors,
        )

        sp_entities = extract_ner_entities(sp)
        sp_authors  = detect_biography_authors(sp)
        sp_entities_with_authors = {
            **sp_entities,
            "authors": [a.lower() for a in sp_authors]
        }

        sp_lower    = sp.lower()
        sp_keywords = [kw for kw in keywords if kw in sp_lower]

        try:
            docs = semantic_similarity_search(
                query=sp,
                top_k=5,
                filter_terms=sp_filter_terms,
                entities=sp_entities_with_authors,
                keywords=sp_keywords,
                year_ranges=sp_ranges,
                db_authors=sp_db_authors,
            )
            for doc in docs:
                doc["sub_question_origin"] = i
                doc["sub_question_text"]   = sp
            new_docs.extend(docs)
            print(f"   {len(docs)} docs | filter: {sp_filter_terms}")
        except Exception as e:
            print(f"   Error: {e}")

    extra_docs  = [d for d in new_docs if d["text"] not in existing_texts]
    all_docs    = existing_docs + extra_docs
    unique_docs = deduplicate_documents(all_docs)

    print(f"\nCombined total: {len(unique_docs)} unique ({len(extra_docs)} new from sub-questions)")
    unique_docs.sort(key=lambda x: x["total_score"], reverse=True)

    state["rag_documents"] = unique_docs
    state["current_cycle"] = state.get("current_cycle", 0)

    return state


def expand_variants(terms: List[str]) -> List[str]:
    variants = []
    for t in terms:
        variants.extend(normalize_for_filter(t))
    return list(dict.fromkeys(variants))


def and_condition(variants: List[str]) -> tuple:
    cond = " AND ".join(["LOWER(texto_completo) LIKE %s" for _ in variants])
    params = [f"%{v}%" for v in variants]
    return cond, params


def or_condition(variants: List[str]) -> tuple:
    cond = " OR ".join(["LOWER(texto_completo) LIKE %s" for _ in variants])
    params = [f"%{v}%" for v in variants]
    return cond, params


def run_filtered_query(
    cur, query_embedding, cond, params, limit, label,
    extra_cond: str = "", extra_params: list = None
) -> list:
    extra_params = extra_params or []
    sql = f"""
        SELECT texto_completo, contexto, titulo_obra, autor, año, pagina, 
               (1 - (embedding <=> %s::vector)) AS semantic_score
        FROM chunks_w_context
        WHERE LENGTH(texto_completo) >= %s AND ({cond}){extra_cond}
        ORDER BY semantic_score DESC
        LIMIT %s;
    """
    cur.execute(sql, [query_embedding, MIN_LENGTH] + params + extra_params + [limit])
    rows = cur.fetchall()
    if rows:
        print(f"   [RAG] {label}: {len(rows)} docs ✓")
    else:
        print(f"   [RAG] {label}: 0 docs")
    return rows


def semantic_similarity_search(
    query: str,
    top_k: int = 10,
    filter_terms: List[str] = None,
    entities: Dict[str, List[str]] = None,
    exclude_texts: List[str] = None,
    keywords: List[str] = None,
    year_ranges: List[Dict] = None,
    db_authors: List[str] = None,
) -> List[Dict]:

    try:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
        query_embedding = response.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return []

    filter_terms = filter_terms or []
    entities     = entities or {"persons": [], "places": []}
    exclude_set  = set(exclude_texts or [])
    keywords     = keywords or []
    year_ranges  = year_ranges or []
    db_authors   = db_authors or []

    persons             = entities.get("persons", [])
    places              = entities.get("places", [])
    entity_authors      = entities.get("authors", [])
    persons_and_authors = list(dict.fromkeys(entity_authors + persons))
    latin_terms         = [t for t in filter_terms
                           if t not in persons_and_authors and t not in places]

    v_persons = expand_variants(persons_and_authors)
    v_latin   = expand_variants(latin_terms)
    v_places  = expand_variants(places)

    print(f"   [RAG] T1-persons: {persons_and_authors} | T2-latin: {latin_terms} | T3-places: {places}")

    year_cond, year_params = build_year_condition(year_ranges)

    db_authors_cond, db_authors_params = build_db_authors_condition(db_authors)

    filtered_extra_cond   = year_cond + db_authors_cond
    filtered_extra_params = year_params + db_authors_params

    t4_extra_cond   = year_cond
    t4_extra_params = year_params

    if filtered_extra_cond:
        active_filters = []
        if year_ranges:
            active_filters.append(f"years: {[r['label'] for r in year_ranges]}")
        if db_authors:
            active_filters.append(f"DB authors (tiers 1-3): {db_authors}")
        print(f"   [RAG] Structured filters: {', '.join(active_filters)}")

    t1_rows: list = []
    t2_rows: list = []
    t3_rows: list = []
    t4_rows: list = []

    with get_db_connection() as conn:
        cur = conn.cursor()

        if v_persons:
            cond, params = or_condition(v_persons)
            t1_rows = run_filtered_query(
                cur, query_embedding, cond, params, top_k,
                "Tier 1 (persons/authors)",
                filtered_extra_cond, filtered_extra_params
            )

        if v_latin:
            cond, params = or_condition(v_latin)
            t2_rows = run_filtered_query(
                cur, query_embedding, cond, params, top_k,
                "Tier 2 (latin)",
                filtered_extra_cond, filtered_extra_params
            )

        if v_places:
            cond, params = or_condition(v_places)
            t3_rows = run_filtered_query(
                cur, query_embedding, cond, params, top_k,
                "Tier 3 (places)",
                filtered_extra_cond, filtered_extra_params
            )

        if t4_extra_cond:
            general_sql = f"""
                SELECT texto_completo, contexto, titulo_obra, autor, año, pagina, 
                       (1 - (embedding <=> %s::vector)) AS semantic_score
                FROM chunks_w_context
                WHERE LENGTH(texto_completo) >= %s{t4_extra_cond}
                ORDER BY semantic_score DESC
                LIMIT %s;
            """
            cur.execute(general_sql,
                        [query_embedding, MIN_LENGTH] + t4_extra_params + [top_k])
            t4_label = "Tier 4 (semantic + years)"
        else:
            general_sql = """
                SELECT texto_completo, contexto, titulo_obra, autor, año, pagina, 
                       (1 - (embedding <=> %s::vector)) AS semantic_score
                FROM chunks_w_context
                WHERE LENGTH(texto_completo) >= %s
                ORDER BY semantic_score DESC
                LIMIT %s;
            """
            cur.execute(general_sql, [query_embedding, MIN_LENGTH, top_k])
            t4_label = "Tier 4 (pure semantic)"

        t4_rows = cur.fetchall()
        print(f"   [RAG] {t4_label}: {len(t4_rows)} docs" + (" ✓" if t4_rows else ""))

    def row_to_dict(row, phase: str) -> Dict:
        full_text, context, title, author, year, page, sem_score = row
        raw_sem_score = float(sem_score)
        return {
            "text":                    full_text,
            "context":                 context or "",
            "title":                   title or "Desconocido",
            "author":                  author or "Desconocido",
            "year":                    year or "Desconocido",
            "page":                    page or "Desconocida",
            "semantic_score":          raw_sem_score,
            "original_semantic_score": raw_sem_score,
            "latin_matches":           0,
            "latin_boost":             0.0,
            "ner_matches":             0,
            "ner_boost":               0.0,
            "person_hits":             [],
            "place_hits":              [],
            "person_boost":            0.0,
            "place_boost":             0.0,
            "retrieval_phase":         phase,
            "keyword_hits":            0,
            "total_score":             raw_sem_score,
        }

    seen:       set        = set()
    candidates: List[Dict] = []

    for rows, phase in [
        (t1_rows, "tier1_persons"),
        (t2_rows, "tier2_latin"),
        (t3_rows, "tier3_places"),
        (t4_rows, "tier4_semantic"),
    ]:
        for row in rows:
            normalized_text = re.sub(r"\s+", " ", row[0].lower().strip())
            if normalized_text not in seen and row[0] not in exclude_set:
                seen.add(normalized_text)
                candidates.append(row_to_dict(row, phase))

    candidates = _apply_boosts(
        documents=candidates,
        keywords=keywords,
        persons=persons,
        places=places,
        latin_terms=latin_terms,
    )

    candidates.sort(key=lambda x: -x["total_score"])

    used_tiers = {}
    for doc in candidates[:top_k]:
        t = doc["retrieval_phase"]
        used_tiers[t] = used_tiers.get(t, 0) + 1
    print(f"   [RAG] Pool: {len(candidates)} | Top {top_k} after boost: {used_tiers}")

    return candidates[:top_k]


def _apply_boosts(
    documents: List[Dict],
    keywords: List[str],
    persons: List[str],
    places: List[str],
    latin_terms: List[str],
) -> List[Dict]:

    for doc in documents:
        text_lower = doc["text"].lower()

        kw_hits = sum(1 for kw in keywords if kw in text_lower) if keywords else 0
        kw_score = (kw_hits / len(keywords)) if keywords else 0.0
        kw_boost = 0.0
        if kw_hits >= 6: kw_boost = 0.30
        elif kw_hits >= 5: kw_boost = 0.25
        elif kw_hits >= 4: kw_boost = 0.20
        elif kw_hits >= 3: kw_boost = 0.15
        elif kw_hits >= 2: kw_boost = 0.10
        elif kw_hits >= 1: kw_boost = 0.05

        person_hits  = [p for p in persons if p and p in text_lower]
        person_boost = len(person_hits) * BOOST_PER_PERSON

        place_hits      = [l for l in places if l and l in text_lower]
        place_boost_val = len(place_hits) * BOOST_PER_PLACE
        total_ner_boost = min(person_boost + place_boost_val, MAX_NER_BOOST)

        latin_hits  = [w for w in latin_terms if w and w in text_lower]
        latin_boost = len(latin_hits) * BOOST_PER_LATIN_TERM

        doc["keyword_hits"]  = kw_hits
        doc["person_hits"]   = person_hits
        doc["person_boost"]  = person_boost
        doc["place_hits"]    = place_hits
        doc["place_boost"]   = place_boost_val
        doc["ner_matches"]   = len(person_hits) + len(place_hits)
        doc["ner_boost"]     = total_ner_boost
        doc["latin_matches"] = len(latin_hits)
        doc["latin_boost"]   = latin_boost
        doc["total_score"]   = (
            doc["semantic_score"] * SEMANTIC_WEIGHT
            + kw_score * KEYWORDS_WEIGHT
            + kw_boost
            + total_ner_boost
            + latin_boost
        )

    return documents


def deduplicate_documents(docs: List[Dict], threshold: float = 0.9) -> List[Dict]:
    if not docs:
        return []

    unique_docs = []
    seen_texts  = set()

    for doc in sorted(docs, key=lambda x: x["total_score"], reverse=True):
        normalized_text = re.sub(r"\s+", " ", doc["text"].lower().strip())

        if normalized_text in seen_texts:
            continue

        is_duplicate = any(
        simple_similarity(normalized_text, st) > threshold
        and doc.get("author") == unique_docs[i].get("author")  # same author
        for i, st in enumerate(seen_texts))


        if not is_duplicate:
            unique_docs.append(doc)
            seen_texts.add(normalized_text)

    return unique_docs


def simple_similarity(t1: str, t2: str) -> float:
    w1, w2 = set(t1.split()), set(t2.split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def rerank_documents(state: AgentState) -> AgentState:
    print("\n" + "="*80)
    print("NODE 5: RERANK")
    print("="*80)

    docs = state.get("rag_documents", [])

    if not docs:
        state["reranked_documents"] = []
        return state

    final_docs = docs[:TOP_K_RERANK]

    print(f"Top {len(final_docs)} selected")
    print("\nTop 5 documents:")
    for i, doc in enumerate(final_docs[:5], 1):
        latin_info = ""
        if doc.get("latin_matches", 0) > 0:
            latin_info = f" | {doc['latin_matches']} lat (+{doc.get('latin_boost', 0):.3f})"

        ner_info = ""
        if doc.get("ner_matches", 0) > 0:
            persons = doc.get("person_hits", [])
            places  = doc.get("place_hits", [])
            parts   = []
            if persons:
                parts.append(f"{len(persons)}per (+{doc.get('person_boost', 0):.3f})")
            if places:
                parts.append(f"{len(places)}loc (+{doc.get('place_boost', 0):.3f})")
            ner_info = f" | NER: {', '.join(parts)}"

        print(f"   {i}. {doc['author']} - {doc['title']} - {doc['year']}")
        print(f"      KW:{doc['keyword_hits']} | Sem:{doc.get('original_semantic_score', doc['semantic_score']):.3f} | "
              f"Total:{doc['total_score']:.3f}{latin_info}{ner_info}")

    state["seen_texts"] = [d["text"] for d in final_docs]

    state["reranked_documents"] = final_docs
    return state


def process_glossary_bibliography_currents(state: AgentState) -> AgentState:
    print("\n" + "="*80)
    print("NODE 6: GLOSSARY/BIBLIOGRAPHY/CURRENTS")
    print("="*80)

    question = state["original_question"]
    docs     = state.get("reranked_documents", [])

    if not docs:
        state.update({
            "needs_glossary":       False,
            "needs_bibliography":   False,
            "needs_currents":       False,
            "technical_terms":      [],
            "proper_names":         [],
            "glossary_definitions": {},
            "bibliography_authors": {},
            "mentioned_authors":    [],
            "currents_definitions": {},
        })
        return state

    context   = "\n".join([doc["text"] for doc in docs])
    full_text = f"{question} {context}"

    available_terms = find_available_terms(full_text)
    print(f"Glossary terms found: {len(available_terms)}")

    useful_terms = (
        select_useful_terms(question, context, available_terms)
        if available_terms else []
    )

    state["technical_terms"]      = useful_terms
    state["needs_glossary"]       = len(useful_terms) > 0
    state["glossary_definitions"] = (
        {t: available_terms[t] for t in useful_terms}
        if useful_terms else {}
    )

    if useful_terms:
        print(f"Useful glossary: {useful_terms}")

    available_authors = find_available_authors(question, context, docs)
    print(f"Bibliography authors: {len(available_authors)}")

    useful_authors = (
        select_useful_authors(question, context, available_authors)
        if available_authors else []
    )

    state["proper_names"]         = useful_authors
    state["needs_bibliography"]   = len(useful_authors) > 0
    state["bibliography_authors"] = (
        {a: available_authors[a] for a in useful_authors}
        if useful_authors else {}
    )

    if useful_authors:
        print(f"Useful bibliography: {useful_authors}")

    useful_currents = select_useful_currents(question, context)

    state["needs_currents"]       = len(useful_currents) > 0
    state["currents_definitions"] = (
        {c: CURRENTS[c] for c in useful_currents if c in CURRENTS}
        if useful_currents else {}
    )

    if useful_currents:
        print(f"Useful currents: {useful_currents}")

    return state


def select_useful_terms(question: str, context: str, available_terms: Dict) -> List[str]:
    term_list = "\n".join([f"- {t}" for t in available_terms.keys()])

    prompt = f"""Analiza esta pregunta y contexto. De los siguientes términos del glosario que aparecen en el texto, ¿cuáles serían REALMENTE útiles para responder la pregunta?

Pregunta: {question}

Contexto (primeros 800 caracteres):
{context[:800]}

Términos disponibles en el glosario:
{term_list}

Selecciona SOLO los términos que ayudarían a responder mejor la pregunta.
Si ninguno es necesario, responde "NINGUNO".
Responde solo con los nombres de los términos, uno por línea.

Términos útiles:"""

    try:
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response["message"]["content"].strip()

        if "NINGUNO" in answer.upper():
            return []

        terms = []
        for line in answer.split("\n"):
            term = re.sub(r"^[\d\.\-\*\)\]\s]+", "", line).strip().replace('"', "").replace("'", "")
            if term and term in available_terms:
                terms.append(term)

        return terms
    except Exception as e:
        print(f"Error selecting useful terms with Ollama: {e}")
        return []


def find_available_authors(question: str, context: str, docs: List[Dict]) -> Dict[str, str]:
    authors = {}

    for doc in docs:
        author = doc.get("author")
        if author and author != "Desconocido" and author in BIOGRAPHIES:
            authors[author] = BIOGRAPHIES[author]

    text = f"{question} {context}".lower()
    for key, value in BIOGRAPHIES.items():
        if key.lower() in text and key not in authors:
            authors[key] = value

    return authors


def select_useful_authors(question: str, context: str, available_authors: Dict) -> List[str]:
    author_list = "\n".join([f"- {a}" for a in available_authors.keys()])

    prompt = f"""Analiza esta pregunta y contexto. De los siguientes autores de la bibliografía que aparecen en el texto, ¿cuáles serían REALMENTE útiles para contextualizar mejor la respuesta?

Pregunta: {question}

Contexto (primeros 800 caracteres):
{context[:800]}

Autores disponibles en la bibliografía:
{author_list}

Selecciona SOLO los autores cuya biografía ayudaría a entender mejor el contexto.
Si ninguno es necesario, responde "NINGUNO".
Responde solo con los nombres completos, uno por línea.

Autores útiles:"""

    try:
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response["message"]["content"].strip()

        if "NINGUNO" in answer.upper():
            return []

        authors = []
        for line in answer.split("\n"):
            name = re.sub(r"^[\d\.\-\*\)\]\s]+", "", line).strip().replace('"', "").replace("'", "")
            if name and name in available_authors:
                authors.append(name)

        return authors
    except Exception as e:
        print(f"Error selecting useful authors with Ollama: {e}")
        return []

def select_useful_currents(question: str, context: str) -> List[str]:

    prompt = f"""Analiza esta pregunta y contexto-

Pregunta: {question}

Contexto (primeros 800 caracteres):
{context[:800]}

Decide si es relevante para responder la pregunta tener contexto sobre alguna de las siguientes corrientes historiográficas:

- Historicismo: Surgido en Alemania, promovió el método histórico crítico basado en la investigación documental exhaustiva en archivos para mostrar "lo que realmente sucedió", destacando figuras como Niebuhr y Ranke.
- Positivismo: Buscaba establecer leyes generales para predecir el futuro de la sociedad, centrándose en el estudio inductivo de los "hechos" a través de los documentos y aspirando a una neutralidad científica.
- Historiografías nacionales: Influenciadas por el nacionalismo romántico del siglo XIX, se enfocaron en construir una identidad y patriotismo a través de la historia de naciones y pueblos en lugar de gobernantes.
- Materialismo histórico: Basado en las ideas de Karl Marx, considera que el motor de la historia es la lucha de clases y las relaciones de producción, dividiendo el pasado en modos de producción (esclavista, feudal, capitalista).
- Historia económica: Influenciada por el marxismo, se especializa en el estudio de magnitudes cuantitativas (precios, producción, consumo) a lo largo de periodos de larga duración.
- Historia social: Estudia las sociedades de masas utilizando herramientas de la sociología y la estadística, alejándose de las técnicas puramente narrativas del positivismo.
- Annales: Fundada en 1929, rechazó el "hecho" aislado en favor de una historia analítica socioeconómica e interdisciplinar, promoviendo conceptos como la "larga duración" y la "historia total".
- Historia serial: Derivada de Annales, utiliza la estadística de manera sistemática para estudiar procesos de media y larga duración.
- Historia de las mentalidades: Se enfoca en las creencias y opiniones compartidas por grupos sociales que operan de forma inconsciente en la conducta cotidiana.
- Historiografía marxista de posguerra: Evolución del materialismo histórico que se alejó de la ortodoxia soviética para centrarse en una historia más cultural e intelectual, destacando autores como Thompson y Hobsbawm.
- Cliometría norteamericana: Corriente norteamericana que aplica modelos matemáticos y econométricos al análisis de grandes cantidades de datos estadísticos asistidos por la informática.
- Microhistoria: Surgida en los años setenta, reduce la escala de análisis a la observación "microscópica" de hechos singulares para arrojar luz sobre procesos sociales más amplios.
- Historia de género: Busca incluir a la mujer como sujeto histórico universal, criticando la "historia en masculino" y expandiéndose posteriormente al estudio de identidades sexuales (Gay studies).
- Historia cultural: Se consolidó en los ochenta, centrando su estudio en los símbolos, significados y representaciones de las sociedades en contextos específicos.
- Historia desde abajo: Explora los acontecimientos históricos desde la perspectiva de las clases populares y grupos tradicionalmente ignorados por la academia.
- Historia de la vida cotidiana: Da prioridad a las experiencias ordinarias, la cultura material (comida, vestido) y las emociones, utilizando frecuentemente fuentes orales.
- Historia de ultramar: Evolucionó de la historia colonial hacia el estudio autónomo de culturas no europeas y la expansión europea desde perspectivas antropológicas.
- Historia oral: Utiliza el recuerdo personal y la tradición transmitida de generación en generación como fuentes primarias de investigación.
- Historia política y diplomática: Renovó sus objetos de estudio abandonando el enfoque militarista y cortesano para incorporar perspectivas socioculturales e internacionales, reinterpretando conceptos como poder, violencia, estado o símbolos políticos.
- Historia de la cultura popular: Estudia las manifestaciones culturales de las masas, aunque enfrenta dificultades metodológicas al intentar describir sucesos ordinarios a través de fuentes que generalmente registran sucesos extraordinarios.
- Historia de la lectura: Analiza el quién, qué, dónde y cuándo de la lectura desde perspectivas macro y microanalíticas, utilizando modelos estadísticos y fuentes particulares, aunque todavía sin lograr comprender los procesos internos de los lectores.
- Historia social del lenguaje: Sirve de nexo entre la historia tradicional y el giro lingüístico, utilizando los códigos y la evolución del lenguaje como síntomas para analizar y definir culturas.
- Historia de las imágenes: Va más allá de la historia del arte al incluir materiales visuales no artísticos (artefactos, coleccionables), investigando autoría, canonicidad e interpretación de las fuentes visuales.
- Historia de la religiosidad: Corriente de gran vitalidad desde los setenta que amplía el estudio de la religión hacia creencias populares, piedad, espiritualidad y representaciones de la muerte.
- Historia del medio ambiente: Surgida de la preocupación por la destrucción humana del entorno natural, exige un enfoque necesariamente interdisciplinar para reconstruir la relación histórica entre el ser humano y su medio.
- Historia mundial e historia global: Evolucionó desde el estudio de imperios hacia lo económico y sociocultural, dividiéndose entre quienes estudian las interacciones entre pueblos y culturas y quienes se centran en comparaciones y contrastes.
- Historia comparada: Analiza paralelismos y diferencias entre realidades históricas próximas desde una perspectiva global, consolidándose como método desde los años cincuenta.

Devuelve cada corriente RELEVANTE UNA por línea, sin numerar ni usar símbolos, solo el nombre de la corriente exactamente como aparece en la lista.
Si ninguna corriente es relevante, responde "NINGUNA".
"""

    try:
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response["message"]["content"].strip()

        if "NINGUNA" in answer.upper():
            return []

        currents = []
        for line in answer.split("\n"):
            current = re.sub(r"^[\d\.\-\*\)\]\s]+", "", line).strip().replace('"', "").replace("'", "")
            if current and current in CURRENTS:
                currents.append(current)

        return currents
    except Exception as e:
        print(f"Error selecting useful currents with Ollama: {e}")
        return []


# PHASE 1: INITIAL ANSWER (question only + top-K paragraphs)

def generate_initial_answer(state: AgentState) -> AgentState:
    """
    Generates initial answer using ONLY the paragraphs retrieved with the main question 
    """
    print("\n" + "="*80)
    print("PHASE 1: INITIAL ANSWER (direct top-K paragraphs)")
    print("="*80)

    question = state["original_question"]
    docs = state.get("rag_documents", [])[:TOP_K]

    state["initial_phase_paragraphs"] = [
        {
            "author": d.get("author"),
            "title":  d.get("title"),
            "year":   d.get("year"),
            "page":   d.get("page"),
            "text":   d.get("text"),
            "score":  d.get("total_score", d.get("similarity", 0)),
        }
        for d in docs
    ]

    if not docs:
        state["initial_answer"] = "No se encontró información."
        return state

    ctx_docs = "\n\n".join([
        f"[{d.get('author','?')} - {d.get('title','?')} - {d.get('year','?')} - p.{d.get('page','?')}]\n{d.get('text','')}"
        for d in docs
    ])

    prompt = f"""Eres un experto historiador especializado en la Hispania visigoda.

# PREGUNTA
{question}

# FUENTES HISTORICAS
{ctx_docs}

# INSTRUCCIONES CRITICAS
1. Usa SOLO informacion ESPECIFICA de las fuentes proporcionadas
2. Incluye nombres propios completos, fechas exactas, lugares concretos
3. Cita textualmente cuando sea relevante usando formato APA7: Se incluyen siempre de manera general el primer apellido del autor, el segundo apellido del autor (si lo hay, ya que en ocasiones solo tienen un apellido), la fecha de publicación y la página o páginas usadas con una “p.” si es solo una página o con “pp.” si son varias. Todo esto entre paréntesis y con un punto al final de la oración. Si citas varias páginas seguidas puedes unirlas con un guión (pp. 5-8). Si son varias páginas las usadas, pero son independientes entre sí, sepáralas con una coma “,” y la última de esas páginas con una “y” (pp. 5, 7-9 y 21). En la fecha de publicación es posible que te aparezcan a veces letras “a”, “b”, “c”… Si esto ocurre debes incluirlas en la cita justo después de la fecha (2021a), (1995b), (2024a) … EJEMPLOS: (Fernández Cadenas, 2021, p. 25) / (Menéndez-Reigada, 1939, pp. 6-7, 9-10 y 31) / (Godoy Fernández y Vilella Masana, 1991, pp. 121 y 124-125) / Contres autores o más: (Torres López et al., 1963, p. 230) / Citar una cita: Se inicia con el mismo formato estándar. Después se incluye la frase “como se citó en” y se cita la fuente donde aparece la cita. Ejemplo: (Nuñez Seixas, 2015, pp. 25-26, como se citó en García Sanjuan, 2023, p. 34) / Citar varias fuentes ejemplo: (Mazorriaga, 1946, p. 25; Orlandis Rovira, 1956, pp. 159 y 163)
4. NO uses conocimiento externo. NUNCA menciones ni cites obras, autores o datos que no aparezcan en los fragmentos proporcionados arriba. Si una fuente no está en el contexto dado, no existe para ti y no puedes citarla.
5. ATRIBUCIÓN ESTRICTA: Antes de atribuir una idea, argumento o distinción a un autor concreto, verifica que esa afirmación aparece EXPLÍCITAMENTE en el fragmento de ese autor. Es preferible omitir una atribución que equivocarla. Si dos fragmentos de autores distintos tratan el mismo tema, no mezcles sus ideas.
6. Si usas comillas para reproducir texto de una fuente, ese texto debe ser EXACTAMENTE literal, carácter por carácter, tal como aparece en el fragmento. Si no puedes garantizar la literalidad exacta, NO uses comillas y parafrasea en su lugar indicando que es una paráfrasis.
7. FOCO ABSOLUTO EN LA PREGUNTA: Responde EXCLUSIVAMENTE lo que se pregunta. Omite cualquier información que no sea directamente necesaria para responder, aunque esté en las fuentes y sea interesante. Contexto histórico general, conversiones de otros personajes, decretos conciliares no preguntados, cronologías de eventos no solicitados: todo eso debe omitirse si la pregunta no lo pide explícitamente.
8. Si mencionas una cita en latín u otro idioma, añade la traducción al español entre paréntesis SIEMPRE
9. Nunca propongas siguientes pasos, tan sólo contesta a la pregunta
10. Si el contexto no sirve para contestar, di claramente "No se encontró información relevante en las fuentes proporcionadas para responder a esta pregunta."
11. Deber tener cuidado al mencionar qué es fuente primaria y qué interpretación moderna. Si no lo tienes claro, menciónalo.

# IMPORTANTE
- NO ES NECESARIO USAR TODOS LOS FRAGMENTOS. Añade sólo lo que realmente responda a la pregunta. Si un fragmento no aporta nada directamente relevante, ignóralo completamente.
- Da los datos concretos que la pregunta pide: nombres, fechas, lugares, documentos, cifras
- La respuesta no debe desviarse hacia temas adyacentes aunque aparezcan en las fuentes
- Explica por qué llegas a cada conclusión basándote en las fuentes, sé específico en el razonamiento
- Si hay contradicciones entre fuentes, mencionalo y cita las fuentes contradictorias
- NUNCA repitas el mismo caso o ejemplo más de una vez en la misma respuesta
- Al citar de forma explícita, revisa que usas la misma ortografía y exactamente las mismas palabras que en la fuente
- Si la fuente usa grafía medieval o latina (uos, Spiritu, fidelis en lugar de vos, Spíritu, fiel), cópiala exactamente. No modernices la ortografía bajo ningún concepto.

# FORMATO
Párrafos académicos concisos. Sin listas a menos que la pregunta lo requiera.

Respuesta:"""

    try:
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
        )
        answer = response.choices[0].message.content.strip()
        state["initial_answer"] = answer
        state["current_answer"] = answer
        print(f"Initial answer generated ({len(answer)} chars)")
    except Exception as e:
        print(f"Error generating initial answer: {e}")
        state["initial_answer"] = f"Error: {e}"
        state["current_answer"] = state["initial_answer"]

    return state


# ============================================================================
# PHASE 2: CONTEXTUALIZATION
# ============================================================================

def generate_contextualized_answer(state: AgentState) -> AgentState:
    """
    Takes the initial_answer + the contextualization paragraphs and produces a contextualized answer.
    The extra paragraphs are only used if they truly provide context for the question.
    """
    print("\n" + "="*80)
    print("PHASE 2: CONTEXTUALIZED ANSWER")
    print("="*80)

    question = state["original_question"]
    initial_answer = state.get("initial_answer", "")
    MAX_EXTRA = 10

    initial_texts = {p.get("text") for p in state.get("initial_phase_paragraphs", [])}
    all_docs = state.get("rag_documents", [])

    extra_docs = [d for d in all_docs
                if d.get("text") not in initial_texts]
    extra_docs = sorted(extra_docs,
                key=lambda x: x.get("total_score", 0),
                reverse=True)[:MAX_EXTRA]

    state["contextualization_phase_paragraphs"] = [
        {
            "author": d.get("author"),
            "title":  d.get("title"),
            "year":   d.get("year"),
            "page":   d.get("page"),
            "text":   d.get("text"),
            "score":  d.get("total_score", d.get("similarity", 0)),
            "sub_question_origin": d.get("sub_question_origin", ""),
        }
        for d in extra_docs
    ]

    if not extra_docs:
        # Nothing to add; the contextualized answer is the initial one
        state["contextualized_answer"] = initial_answer
        state["current_answer"] = initial_answer
        print("No extra contextualization paragraphs; keeping the initial answer.")
        return state

    ctx_extra = "\n\n".join([
        f"[{d.get('author','?')} - {d.get('title','?')} - {d.get('year','?')} - p.{d.get('page','?')}]\n{d.get('text','')}"
        for d in extra_docs
    ])

    # Additional resources already loaded by process_glossary_bibliography_currents
    glossary     = state.get("glossary_definitions", {}) or {}
    bibliography = state.get("bibliography_authors", {}) or {}
    currents     = state.get("currents_definitions", {}) or {}

    glossary_ctx = ""
    if glossary:
        glossary_ctx = "\n\n# GLOSARIO\n" + "\n".join(
            [f"- **{t}**: {d}" for t, d in glossary.items()]
        )
    bibliography_ctx = ""
    if bibliography:
        bibliography_ctx = "\n\n# BIOGRAFIAS\n" + "\n".join(
            [f"- **{n}**: {b}" for n, b in bibliography.items()]
        )
    currents_ctx = ""
    if currents:
        currents_ctx = "\n\n# CORRIENTES HISTORIOGRÁFICAS\n" + "\n".join(
            [f"- **{c}**: {d}" for c, d in currents.items()]
        )

    prompt = f"""Eres un experto historiador especializado en la Hispania visigoda.

# PREGUNTA ORIGINAL
{question}

# RESPUESTA INICIAL (a contextualizar SÓLO si es necesario)
{initial_answer}

# MATERIAL ADICIONAL PARA POSIBLE CONTEXTUALIZACIÓN
Recibes a continuación material que PODRÍA usarse para añadir encuadre a la
respuesta inicial. NO TIENES POR QUÉ USARLO TODO: úsalo SÓLO si realmente
aporta contexto necesario para entender mejor algún aspecto de la pregunta
original. Si no aporta nada útil, ignora ese material y devuelve la
respuesta inicial tal cual.

## Párrafos extra (hasta {MAX_EXTRA})
{ctx_extra}
{glossary_ctx}
{bibliography_ctx}
{currents_ctx}

# QUÉ CUENTA COMO CONTEXTO

- Marco histórico-político general en el que ocurre lo que la pregunta plantea (situación de la monarquía, estructuras de poder, relaciones entre poderes).
- Marco religioso, eclesiástico o doctrinal de la época si la pregunta lo toca.
- Encuadre geográfico (provincia, sede, frontera) sólo a nivel descriptivo, no para listar nombres exactos.
- Encuadre social o económico si es necesario para entender la pregunta.
- Antecedentes o consecuencias inmediatas que ayuden a entender el porqué de lo preguntado.

# REGLAS
1. CONTEXTUALIZA SÓLO SI ES NECESARIO. Si la respuesta inicial ya está bien encuadrada, devuelve la respuesta inicial tal cual.
2. NO CAMBIES EL FOCO DE LA PREGUNTA. Cada adición debe ser directamente necesaria para entender la pregunta original. Si algo es "interesante" pero no ayuda a responder la pregunta, no lo incluyas. En caso de duda, omite.
3. Parte siempre de la RESPUESTA INICIAL y consérvala donde ya esté bien; integra sin repetir lo que ya estaba.
4. Usa cualquiera de los materiales adicionales (párrafos, glosario, biografías, corrientes) SOLO para añadir encuadre.
5. Si algo del material no encuadra nada de la pregunta original, IGNÓRALO.
6. ATRIBUCIÓN ESTRICTA: No atribuyas a un autor ideas que no aparecen literalmente en sus fragmentos. Si usas comillas, el texto debe ser exactamente literal. Si no puedes garantizarlo, parafrasea sin comillas.
7. FUENTES: No menciones ni cites obras o autores que no estén en los fragmentos proporcionados. Si una fuente no aparece en el contexto, no existe.
8. Cita textualmente cuando sea relevante usando formato APA7: Se incluyen siempre de manera general el primer apellido del autor, el segundo apellido del autor (si lo hay, ya que en ocasiones solo tienen un apellido), la fecha de publicación y la página o páginas usadas con una “p.” si es solo una página o con “pp.” si son varias. Todo esto entre paréntesis y con un punto al final de la oración. Si citas varias páginas seguidas puedes unirlas con un guión (pp. 5-8). Si son varias páginas las usadas, pero son independientes entre sí, sepáralas con una coma “,” y la última de esas páginas con una “y” (pp. 5, 7-9 y 21). En la fecha de publicación es posible que te aparezcan a veces letras “a”, “b”, “c”… Si esto ocurre debes incluirlas en la cita justo después de la fecha (2021a), (1995b), (2024a) … EJEMPLOS: (Fernández Cadenas, 2021, p. 25) / (Menéndez-Reigada, 1939, pp. 6-7, 9-10 y 31) / (Godoy Fernández y Vilella Masana, 1991, pp. 121 y 124-125) / Contres autores o más: (Torres López et al., 1963, p. 230) / Citar una cita: Se inicia con el mismo formato estándar. Después se incluye la frase “como se citó en” y se cita la fuente donde aparece la cita. Ejemplo: (Nuñez Seixas, 2015, pp. 25-26, como se citó en García Sanjuan, 2023, p. 34) / Citar varias fuentes ejemplo: (Mazorriaga, 1946, p. 25; Orlandis Rovira, 1956, pp. 159 y 163)
9. Si mencionas una cita en latín u otro idioma, añade la traducción al español entre paréntesis SIEMPRE
10. Distingue fuente primaria de interpretación moderna; si dudas, dilo.
11. Sin siguientes pasos ni listas (salvo que la pregunta lo pida).
12. Si la fuente usa grafía medieval o latina (uos, Spiritu, fidelis en lugar de vos, Spíritu, fiel), cópiala exactamente. No modernices la ortografía bajo ningún concepto.

Respuesta contextualizada:"""

    try:
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
        )
        answer = response.choices[0].message.content.strip()
        state["contextualized_answer"] = answer
        state["current_answer"] = answer
        print(f"Contextualized answer generated ({len(answer)} chars)")
    except Exception as e:
        print(f"Error generating contextualized answer: {e}")
        state["contextualized_answer"] = initial_answer
        state["current_answer"] = initial_answer

    return state


def verify_and_refine_with_gaps(state: AgentState) -> AgentState:

    print("\n" + "="*80)
    print("NODE 8: VERIFICATION AND ITERATIVE REFINEMENT")
    print("="*80)

    question         = state["original_question"]
    keywords         = state.get("main_keywords", [])
    filter_terms     = state.get("filter_terms", [])
    base_year_ranges = state.get("year_ranges", [])
    base_db_authors  = state.get("db_authors_filter", [])
    seen_texts       = list(state.get("seen_texts", []))

    cycle = 0
    refinements: List[Dict] = []
    accumulated_gaps: List[str] = []

    while cycle < MAX_ITERATIONS:
        current_answer = state.get("current_answer", "")
        print(f"\n--- Refinement cycle {cycle + 1}/{MAX_ITERATIONS} ---")

        gaps = _identify_gaps(question, current_answer, previous_gaps=accumulated_gaps)
        accumulated_gaps.extend(gaps)

        if not gaps:
            print("No gaps identified. Answer complete.")
            state["is_complete"]     = True
            state["identified_gaps"] = []
            refinements.append({
                "cycle": cycle + 1,
                "input_answer": current_answer,
                "gaps": [],
                "fragments": [],
                "output_answer": current_answer,
                "complete": True,
            })
            break

        print(f"Identified gaps ({len(gaps)}):")
        for g in gaps:
            print(f"   - {g}")

        state["identified_gaps"] = gaps

        new_fragments: List[Dict] = []

        for i, gap in enumerate(gaps, 1):
            print(f"\nSearching fragments for gap {i}: {gap}...")

            gap_query = f"{question} {gap}"
            gap_terms, gap_ranges, gap_db_authors = build_filter_terms_for(
                gap,
                filter_terms,
                base_ranges=base_year_ranges,
                base_db_authors=base_db_authors,
            )

            gap_entities = extract_ner_entities(gap)
            gap_authors  = detect_biography_authors(gap)
            gap_entities_with_authors = {
                **gap_entities,
                "authors": [a.lower() for a in gap_authors]
            }

            gap_lower    = gap.lower()
            gap_keywords = [kw for kw in keywords if kw in gap_lower]

            try:
                gap_docs = semantic_similarity_search(
                    query=gap_query,
                    top_k=FRAGMENTS_PER_GAP,
                    filter_terms=gap_terms,
                    entities=gap_entities_with_authors,
                    exclude_texts=seen_texts,
                    keywords=gap_keywords,
                    year_ranges=gap_ranges,
                    db_authors=gap_db_authors,
                )

                for doc in gap_docs:
                    doc["origin_gap"]       = gap
                    doc["refinement_cycle"] = cycle + 1

                for doc in gap_docs:
                    if doc["text"] not in seen_texts:
                        seen_texts.append(doc["text"])
                        new_fragments.append(doc)

                print(f"   {len(gap_docs)} new fragments for this gap")

            except Exception as e:
                print(f"   Error: {e}")

        if not new_fragments:
            print("No new fragments found. Ending refinement.")
            state["is_complete"] = True
            break

        print(f"\nTotal new fragments this cycle: {len(new_fragments)}")

        new_answer = _generate_refined_answer(
            question=question,
            previous_answer=current_answer,
            new_fragments=new_fragments,
        )

        refinements.append({
            "cycle": cycle + 1,
            "input_answer": current_answer,
            "gaps": gaps,
            "fragments": [
                {
                    "author": d.get("author"),
                    "title":  d.get("title"),
                    "year":   d.get("year"),
                    "page":   d.get("page"),
                    "text":   d.get("text"),
                    "origin_gap": d.get("origin_gap"),
                    "score":  d.get("total_score", d.get("similarity", 0)),
                }
                for d in new_fragments
            ],
            "output_answer": new_answer,
            "complete": False,
        })

        state["current_answer"] = new_answer
        state["seen_texts"]     = seen_texts
        cycle += 1

    if cycle >= MAX_ITERATIONS:
        print(f"\nReached MAX_ITERATIONS ({MAX_ITERATIONS}). Ending.")
        state["is_complete"] = True

    state["refinements"]   = refinements
    state["current_cycle"] = cycle
    state["final_answer"]  = state["current_answer"]
    return state


def _identify_gaps(question: str, answer: str,
                   previous_gaps: List[str] = None) -> List[str]:

    previous_gaps = previous_gaps or []

    previous_section = ""
    if previous_gaps:
        gap_list = "\n".join(f"- {g}" for g in previous_gaps)
        previous_section = f"""
# GAPS YA INTENTADOS EN CICLOS ANTERIORES
Los siguientes gaps ya se buscaron en la base de datos y no se encontró
información adicional. NO los repitas ni generes variantes de ellos:
{gap_list}
"""

    prompt = f"""Analiza qué información concreta le FALTA a esta respuesta para responder completamente la pregunta.

# PREGUNTA
{question}

# RESPUESTA
{answer}

{previous_section}

Tu único objetivo es identificar datos específicos ausentes: cifras, fechas, nombres, procedimientos, condiciones, excepciones o cualquier detalle concreto que la pregunta requiere pero la respuesta no menciona o deja incompleto.

NO evalúes redacción, estructura ni calidad general. Solo busca información que falta.

Ejemplos de gaps NO útiles (evítalos):
- "La respuesta podría ser más detallada"
- "Falta contexto general"

IMPORTANTE: los gaps serán utilizados en un RAG, por lo que debes formularlos como preguntas que pueden ser respondidas basándose en similitud semántica con los documentos disponibles. Evita gaps que requieran razonamiento complejo o síntesis de múltiples fuentes.

REGLA DE FOCO ESTRICTA: cada gap debe ser un dato concreto que la pregunta pide EXPLÍCITAMENTE o que es imprescindible para responderla. Rechaza cualquier gap que:
- Introduzca un personaje, evento o concepto no directamente relevante para la pregunta
- Amplíe el contexto más allá de lo que la pregunta pide
- Sea un dato "interesante" pero que no cambie la respuesta a la pregunta concreta
- Describa algo que ya está bien cubierto en la respuesta actual

Si la respuesta actual responde lo que la pregunta pide aunque de forma breve, considera que está completa. La brevedad y el foco son preferibles a la amplitud.

JSON:
{{
  "esta_completa": true|false,
  "gaps_identificados": ["dato o información concreta que falta 1", "dato o información concreta que falta 2"]
}}"""

    try:
        response = client.chat.completions.create(
            model=CLASSIFICATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            response_format={"type": "json_object"}
        )

        eval_data = json.loads(response.choices[0].message.content)
        complete  = eval_data.get("esta_completa", False)
        gaps      = eval_data.get("gaps_identificados", [])

        if complete:
            return []
        return gaps

    except Exception as e:
        print(f"Error identifying gaps: {e}")
        return []


def _generate_refined_answer(
    question: str,
    previous_answer: str,
    new_fragments: List[Dict],
) -> str:

    print("\nGenerating refined answer...")

    fragments_ctx = "\n\n".join([
        f"[{doc['author']} - {doc['title']} - {doc['year']} - {doc['page']}]\n"
        f"(Recuperado para gap: {doc.get('origin_gap', 'desconocido')})\n"
        f"{doc['text']}"
        for doc in new_fragments
    ])

    prompt = f"""Eres un experto historiador especializado en la Hispania visigoda.

# PREGUNTA ORIGINAL
{question}

# RESPUESTA ANTERIOR (base a mejorar)
{previous_answer}

# NUEVOS FRAGMENTOS DE FUENTES (pueden ser útiles para contextualizar la pregunta o dar más detalles)
{fragments_ctx}

# INSTRUCCIONES
1. Toma la respuesta anterior como BASE: conserva todo lo que ya está bien respondido.
2. Usa los NUEVOS FRAGMENTOS para completar o corregir la pregunta con más detalles o contexto útil en caso de que así lo consideres.
3. Integra la nueva información de forma fluida en la respuesta, sin repetir lo que ya era correcto.
4. Usa SOLO información de las fuentes (respuesta anterior + nuevos fragmentos). No añadas conocimiento externo. NUNCA cites obras o autores que no estén en los fragmentos proporcionados.
4b. ATRIBUCIÓN ESTRICTA: Verifica que cada idea que atribuyes a un autor concreto aparece EXPLÍCITAMENTE en el fragmento de ese autor. No mezcles ideas de fragmentos de autores distintos aunque traten el mismo tema.
4c. Si usas comillas, el texto debe ser exactamente literal. Si no puedes garantizarlo, parafrasea sin comillas.
5. Incluye nombres propios completos, fechas exactas, lugares concretos.
6. Cita textualmente cuando sea relevante usando formato APA7: Se incluyen siempre de manera general el primer apellido del autor, el segundo apellido del autor (si lo hay, ya que en ocasiones solo tienen un apellido), la fecha de publicación y la página o páginas usadas con una “p.” si es solo una página o con “pp.” si son varias. Todo esto entre paréntesis y con un punto al final de la oración. Si citas varias páginas seguidas puedes unirlas con un guión (pp. 5-8). Si son varias páginas las usadas, pero son independientes entre sí, sepáralas con una coma “,” y la última de esas páginas con una “y” (pp. 5, 7-9 y 21). En la fecha de publicación es posible que te aparezcan a veces letras “a”, “b”, “c”… Si esto ocurre debes incluirlas en la cita justo después de la fecha (2021a), (1995b), (2024a) … EJEMPLOS: (Fernández Cadenas, 2021, p. 25) / (Menéndez-Reigada, 1939, pp. 6-7, 9-10 y 31) / (Godoy Fernández y Vilella Masana, 1991, pp. 121 y 124-125) / Contres autores o más: (Torres López et al., 1963, p. 230) / Citar una cita: Se inicia con el mismo formato estándar. Después se incluye la frase “como se citó en” y se cita la fuente donde aparece la cita. Ejemplo: (Nuñez Seixas, 2015, pp. 25-26, como se citó en García Sanjuan, 2023, p. 34) / Citar varias fuentes ejemplo: (Mazorriaga, 1946, p. 25; Orlandis Rovira, 1956, pp. 159 y 163)
7. Si los nuevos fragmentos no aportan información útil, no los menciones.
8. Si mencionas una cita en latín u otro idioma, añade la traducción al español entre paréntesis SIEMPRE
9. No propongas siguientes pasos.
10. Excepto que la pregunta requiera lo contrario, no menciones qué información falta en la respuesta final. Solo responde a la pregunta de forma completa.
11. La respuesta debe centrarse EXCLUSIVAMENTE en responder la pregunta. No añadas información de los fragmentos que no sea directamente necesaria para la pregunta, aunque sea relevante históricamente. Menos es más.
12. Deber tener cuidado al mencionar qué es fuente primaria y qué interpretación moderna. Si no lo tienes claro, menciónalo.

# FORMATO
Párrafos académicos concisos. Sin listas a menos que la pregunta lo requiera.

# IMPORTANTE
- NO ES NECESARIO USAR TODOS LOS FRAGMENTOS NUEVOS, añade sólo lo que realmente aporte información útil para responder mejor la pregunta. Si un fragmento no aporta nada relevante, ignóralo.
- Da todos los datos posibles: nombres, fechas, lugares, documentos, cifras
- La respuesta debe quedar contextualizada, pero sin llegar a ser demasiado general ni perder el foco de la pregunta, sólo añade el contexto si es realmente útil
- Es importante que expliques por qué llegas a cada conclusión y comparación basándote en las fuentes, sé específico en el razonamiento, debe quedar debidamente justificado
- Si hay contradicciones entre fuentes, mencionalo y cita las fuentes contradictorias
- Si las fuentes no son útiles para contestar a la pregunta, no las menciones de ninguna forma
- Al citar de forma explícita, revisa que usas la misma ortografía y exactamente las mismas palabras que en la fuente
- Es MUY IMPORTANTE que no elimines citas en formato APA de respuestas anteriores en sus partes correspondientes
- NUNCA hagas referencia a que esta respuesta viene de una anterior, debe parecer siempre una respuesta única y completa, sin importar cuántos ciclos de refinamiento haya habido
- Si la fuente usa grafía medieval o latina (uos, Spiritu, fidelis en lugar de vos, Spíritu, fiel), cópiala exactamente. No modernices la ortografía bajo ningún concepto.

Respuesta mejorada:"""

    try:
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=1
        )

        new_answer = response.choices[0].message.content.strip()
        return new_answer

    except Exception as e:
        print(f"Error generating refined answer: {e}")
        return previous_answer

def create_agent_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("prepare_question",         prepare_question)
    workflow.add_node("rag_main_question",        rag_main_question)
    # PHASE 1
    workflow.add_node("generate_initial_answer",  generate_initial_answer)
    # CONTEXTUALIZATION: sub-questions + sub-question rag + rerank + resources
    workflow.add_node("generate_sub_questions",   generate_sub_questions)
    workflow.add_node("rag_sub_questions",        rag_sub_questions)
    workflow.add_node("rerank",                   rerank_documents)
    workflow.add_node("process_resources",        process_glossary_bibliography_currents)
    # PHASE 2
    workflow.add_node("generate_contextualized_answer", generate_contextualized_answer)
    # PHASE 3
    workflow.add_node("verify_and_refine",        verify_and_refine_with_gaps)

    workflow.set_entry_point("prepare_question")

    workflow.add_edge("prepare_question",               "rag_main_question")
    workflow.add_edge("rag_main_question",              "generate_initial_answer")
    workflow.add_edge("generate_initial_answer",        "generate_sub_questions")
    workflow.add_edge("generate_sub_questions",         "rag_sub_questions")
    workflow.add_edge("rag_sub_questions",              "rerank")
    workflow.add_edge("rerank",                         "process_resources")
    workflow.add_edge("process_resources",              "generate_contextualized_answer")
    workflow.add_edge("generate_contextualized_answer", "verify_and_refine")
    workflow.add_edge("verify_and_refine",              END)

    return workflow.compile()

def run_agent(question: str) -> Dict:
    print("\n" + "="*80)
    print("RAG AGENT - WITH MULTILINGUAL GLOSSARY, LATIN BOOST, NER BOOST AND DB FILTERS")
    print("="*80)
    print(f"\nQUESTION: {question}\n")

    initial_state = {
        "messages":              [HumanMessage(content=question)],
        "original_question":     question,
        "sub_questions":         [],
        "main_keywords":         [],
        "rag_context":           "",
        "rag_documents":         [],
        "reranked_documents":    [],
        "seen_texts":            [],
        "latin_terms":           [],
        "technical_terms":       [],
        "proper_names":          [],
        "question_entities":     {"persons": [], "places": []},
        "filter_terms":          [],
        "mentioned_authors":     [],
        "year_ranges":           [],
        "db_authors_filter":     [],
        "needs_glossary":        False,
        "glossary_definitions":  {},
        "needs_bibliography":    False,
        "bibliography_authors":  {},
        "needs_currents":        False,
        "currents_definitions":  {},
        # Phase 1
        "initial_phase_paragraphs":           [],
        "initial_answer":                     "",
        # Phase 2
        "contextualization_phase_paragraphs": [],
        "contextualized_answer":              "",
        # Phase 3
        "refinements":     [],
        "current_answer":  "",
        "is_complete":     False,
        "identified_gaps": [],
        "current_cycle":   0,
        "final_answer":    "",
    }

    graph = create_agent_graph()

    try:
        result = graph.invoke(initial_state)
        print("\n" + "="*80)
        print("COMPLETED")
        print("="*80)
        return result

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return initial_state




def dump_state_to_json(state: Dict, output_path: str) -> None:
    """
    Dumps a structured summary of the state to a JSON suitable for evaluation.
    Includes:
      - Original question
      - Sub-questions
      - Answers from each phase (initial, contextualized, refinements x N)
      - Paragraphs used in each phase (with their metadata)
      - Gaps identified per refinement cycle
      - Metadata: confidence, complete, cycles
    """

    output = {
        "original_question": state.get("original_question", ""),
        "sub_questions":     state.get("sub_questions", []),
        "main_keywords":     state.get("main_keywords", []),
        "question_entities": state.get("question_entities", {}),
        "filter_terms":      state.get("filter_terms", []),
        "year_ranges":       state.get("year_ranges", []),
        "db_authors_filter": state.get("db_authors_filter", []),
        "initial_phase": {
            "paragraphs": state.get("initial_phase_paragraphs", []),
            "answer": state.get("initial_answer", ""),
        },
        "contextualization_phase": {
            "paragraphs": state.get("contextualization_phase_paragraphs", []),
            "answer": state.get("contextualized_answer", ""),
        },
        "refinement_phase": {
            "cycles": state.get("refinements", []),
            "final_answer": state.get("final_answer", ""),
            "cycles_done": state.get("current_cycle", 0),
            "complete": state.get("is_complete", False),
            "final_gaps": state.get("identified_gaps", []),
        },
        "used_resources": {
            "glossary":    state.get("glossary_definitions", {}),
            "biographies": state.get("bibliography_authors", {}),
            "currents":    state.get("currents_definitions", {}),
        }
    }
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        print(f"Full state dumped to: {output_path}")
    except Exception as e:
        print(f"Error dumping state to JSON: {e}")


def show_result(state: Dict):
    print("\n" + "="*100)
    print("PHASE 1 — INITIAL ANSWER")
    print("="*100)
    print(state.get("initial_answer", "(not generated)"))

    print("\n" + "="*100)
    print("PHASE 2 — CONTEXTUALIZED ANSWER")
    print("="*100)
    print(state.get("contextualized_answer", "(not generated)"))

    refinements = state.get("refinements", [])
    for r in refinements:
        print("\n" + "="*100)
        print(f"PHASE 3 — REFINEMENT CYCLE {r.get('cycle','?')}")
        print("="*100)
        gaps = r.get("gaps", [])
        if gaps:
            print(f"Identified gaps ({len(gaps)}):")
            for g in gaps:
                print(f"   - {g}")
        else:
            print("No gaps identified.")
        print(f"\nAnswer after this cycle:")
        print(r.get("output_answer", "(not generated)"))

    print("\n" + "="*100)
    print("FINAL ANSWER")
    print("="*100)
    print(state.get("final_answer", "Not generated"))

    print("\n" + "="*100)
    print("STATISTICS")
    print("="*100)

    print(f"Keywords: {state.get('main_keywords', [])}")

    entities = state.get("question_entities", {})
    if entities.get("persons"):
        print(f"Persons (NER): {entities['persons']}")
    if entities.get("places"):
        print(f"Places (NER): {entities['places']}")

    year_ranges = state.get("year_ranges", [])
    if year_ranges:
        summary = [
            (r["label"], r["year_min"], "exact" if r.get("exact") else f"–{r['year_max']}")
            for r in year_ranges
        ]
        print(f"Academic ranges: {summary}")

    db_authors = state.get("db_authors_filter", [])
    if db_authors:
        print(f"Filtered DB authors: {db_authors}")

    print(f"Sub-questions: {len(state.get('sub_questions', []))}")
    print(f"Retrieved docs: {len(state.get('rag_documents', []))}")
    print(f"Reranked docs: {len(state.get('reranked_documents', []))}")
    print(f"Seen texts (total): {len(state.get('seen_texts', []))}")

    latin_terms = state.get("latin_terms", [])
    if latin_terms:
        print(f"Latin terms in the question: {latin_terms}")

    print(f"Refinement cycles: {state.get('current_cycle', 0)}")
    print(f"Complete: {'Yes' if state.get('is_complete', False) else 'No'}")

    if state.get("identified_gaps"):
        print(f"\nLast gaps ({len(state['identified_gaps'])}):")
        for gap in state["identified_gaps"]:
            print(f"   - {gap}")

    if state.get("glossary_definitions"):
        print(f"\nGlossary: {len(state['glossary_definitions'])} terms")

    if state.get("bibliography_authors"):
        print(f"Bibliography: {len(state['bibliography_authors'])} authors")

    docs = state.get("reranked_documents", [])

    if docs and latin_terms:
        boosted_docs = [d for d in docs if d.get("latin_matches", 0) > 0]
        if boosted_docs:
            print(f"\nLATIN BOOST IMPACT:")
            print(f"   Docs with latin terms: {len(boosted_docs)}/{len(docs)}")
            avg_boost = sum(d.get("latin_boost", 0) for d in boosted_docs) / len(boosted_docs)
            print(f"   Average boost: +{avg_boost:.3f}")
            max_doc = max(boosted_docs, key=lambda x: x.get("latin_boost", 0))
            print(f"   Highest boost: +{max_doc.get('latin_boost', 0):.3f} ({max_doc.get('latin_matches', 0)} matches)")

    if docs:
        ner_docs = [d for d in docs if d.get("ner_matches", 0) > 0]
        if ner_docs:
            print(f"\n🔍 NER BOOST IMPACT:")
            print(f"   Docs with NER entities: {len(ner_docs)}/{len(docs)}")

            docs_with_persons = [d for d in ner_docs if d.get("person_hits")]
            docs_with_places  = [d for d in ner_docs if d.get("place_hits")]

            if docs_with_persons:
                avg_p = sum(d.get("person_boost", 0) for d in docs_with_persons) / len(docs_with_persons)
                print(f"   Average person boost: +{avg_p:.3f} ({len(docs_with_persons)} docs)")

            if docs_with_places:
                avg_l = sum(d.get("place_boost", 0) for d in docs_with_places) / len(docs_with_places)
                print(f"   Average place boost: +{avg_l:.3f} ({len(docs_with_places)} docs)")

            max_ner = max(ner_docs, key=lambda x: x.get("ner_boost", 0))
            print(f"   Highest NER boost: +{max_ner.get('ner_boost', 0):.3f} | "
                  f"persons={max_ner.get('person_hits', [])} | "
                  f"places={max_ner.get('place_hits', [])}")


def slug(text: str, maxlen: int = 60) -> str:
    """Generates a filesystem-safe slug from the text."""
    import unicodedata, re as _re
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = _re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_").lower()
    return t[:maxlen] or "question"


if __name__ == "__main__":
    import sys

    INPUT_PATH = "prompts.json"
    AGGREGATED_OUTPUT_PATH = "results.json"
    EVALUATION_DIR = "eval"

    os.makedirs(EVALUATION_DIR, exist_ok=True)

    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: File {INPUT_PATH} not found")
        sys.exit(1)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    if isinstance(input_data, list) and len(input_data) > 0:
        if isinstance(input_data[0], dict):
            question_list = [item["pregunta"] for item in input_data if "pregunta" in item]
        else:
            question_list = input_data
    else:
        print(f"The format of {INPUT_PATH} is not valid")
        sys.exit(1)

    all_results = []

    for i, question_text in enumerate(question_list, 1):
        print(f"\n\n{'#'*100}")
        print(f" PROCESSING QUESTION {i}/{len(question_list)}: {question_text}")
        print(f"{'#'*100}\n")

        final_state = run_agent(question_text)

        record = {
            "id": i,
            "original_question": question_text,
            "search_configuration": {
                "keywords": final_state.get("main_keywords"),
                "ner_entities": final_state.get("question_entities"),
                "year_filters": final_state.get("year_ranges"),
                "db_authors": final_state.get("db_authors_filter")
            },
            "process_phases": {
                "phase_1_initial_rag": {
                    "answer": final_state.get("initial_answer"),
                    "collected_context": [
                        {
                            "source": f"{p.get('author')} ({p.get('year')})",
                            "title": p.get("title"),
                            "text": p.get("text"),
                            "page": p.get("page")
                        } for p in final_state.get("initial_phase_paragraphs", [])
                    ]
                },
                "phase_2_contextualization": {
                    "answer": final_state.get("contextualized_answer"),
                    "sub_questions": final_state.get("sub_questions", []),
                    "additional_context": [
                        {
                            "origin": p.get("sub_question_text"),
                            "text": p.get("text")
                        } for p in final_state.get("contextualization_phase_paragraphs", [])
                    ],
                    "external_resources": {
                        "glossary_terms": final_state.get("glossary_definitions"),
                        "author_biographies": final_state.get("bibliography_authors"),
                        "historiographical_currents": final_state.get("currents_definitions")
                    }
                },
                "phase_3_iterative_refinements": [
                    {
                        "iteration": r.get("cycle"),
                        "detected_gaps": r.get("gaps"),
                        "answer_refined_in_this_step": r.get("output_answer"),
                        "new_context_for_this_iteration": [
                            {
                                "triggering_gap": f.get("origin_gap"),
                                "author": f.get("author"),
                                "fragment": f.get("text")
                            } for f in r.get("fragments", [])
                        ]
                    } for r in final_state.get("refinements", [])
                ]
            },
            "final_result": final_state.get("final_answer"),
            "agent_statistics": {
                "completed_cycles": final_state.get("current_cycle"),
                "total_sources_consulted": len(final_state.get("seen_texts", [])),
                "completion_status": "Complete" if final_state.get("is_complete") else "Incomplete/Max_Iter"
            }
        }

        all_results.append(record)

        file_name = f"{i}_{slug(question_text)}.json"
        dump_state_to_json(final_state, os.path.join(EVALUATION_DIR, file_name))

    with open(AGGREGATED_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)