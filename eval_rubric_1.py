import json
import statistics
import time

from openai import OpenAI

MODEL =  "gpt-4.1-mini"  # judge model 
N_PASSES = 3      # independent judge runs per answer, aggregated by median

# Keys of the JSON returned by the judge (defined in the Spanish prompt) -> English output names
DIMENSIONS = {
    "relevancia": "relevance",
    "completitud": "completeness",
    "coherencia": "coherence",
    "veracidad_factual": "factual_accuracy",
}


SYSTEM_PROMPT = """Eres un evaluador experto en historia medieval hispana.
Tu tarea es evaluar respuestas de un sistema de IA a preguntas sobre historia visigoda.
Sé riguroso, justo y basa TODAS las puntuaciones exclusivamente en la evidencia textual proporcionada.
Responde SIEMPRE con JSON válido. Ningún texto fuera del JSON."""


def build_prompt(question: str, ai_answer: str, ground_truth: str) -> str:
    return f"""Evalúa la siguiente respuesta en CUATRO dimensiones. Devuelve un único objeto JSON.

════════════════════════════════════════
PREGUNTA:
{question}

RESPUESTA DE LA IA (a evaluar):
{ai_answer}

RESPUESTA DE REFERENCIA (ground truth):
{ground_truth}
════════════════════════════════════════

INSTRUCCIONES POR DIMENSIÓN (escala 1-10):

─── RELEVANCIA (evalúa sin usar la ground truth) ───
¿La respuesta responde exactamente lo que se preguntó?
Rúbrica:
  1-2  No responde la pregunta o responde una distinta
  3-4  Responde parcialmente; desvíos importantes o foco equivocado
  5-6  Responde la pregunta pero incluye digresiones o pierde el foco en algún punto
  7-8  Responde bien con mínimas desviaciones
  9-10 Responde exactamente lo preguntado, enfocado y sin digresiones

─── COMPLETITUD (compara con la ground truth) ───
¿Cubre los puntos clave de la ground truth y además aporta contexto adicional?
REGLA CRÍTICA: omitir información ≠ contradecir. Penaliza SOLO por puntos clave ausentes, no por no contradecir.
Añadir contexto extra es positivo y debe subir la puntuación.
Rúbrica:
  1-2  Omite la mayoría de puntos clave de la ground truth
  3-4  Cubre menos de la mitad de los puntos clave
  5-6  Cubre los puntos principales pero omite detalles relevantes o autores específicos pedidos
  7-8  Cubre todos los puntos clave; añade algo de contexto
  9-10 Cubre completamente la ground truth y enriquece con contexto historiográfico valioso

─── COHERENCIA (evalúa sin usar la ground truth) ───
¿El razonamiento es internamente consistente? ¿No se contradice a sí misma? ¿Fluye con lógica?
Rúbrica:
  1-2  Contradicciones directas o razonamiento ilógico grave
  3-4  Inconsistencias importantes que confunden el argumento
  5-6  Generalmente coherente pero con saltos lógicos o imprecisiones menores
  7-8  Coherente y bien estructurada
  9-10 Perfectamente coherente, sin contradicciones, razonamiento fluido y claro

─── VERACIDAD FACTUAL (compara con la ground truth) ───
¿La respuesta CONTRADICE algún hecho de la ground truth?
REGLA CRÍTICA: omitir ≠ contradecir. Penaliza SOLO por afirmaciones que contradigan directamente hechos de la ground truth.
Ampliar, matizar o añadir información NUNCA baja la puntuación.
Rúbrica:
  1-2  Contradice directamente múltiples hechos clave de la ground truth
  3-4  Contradice algún hecho importante
  5-6  Mayormente correcto; alguna imprecisión menor que roza la contradicción
  7-8  No contradice la ground truth; puede ampliar sin errores
  9-10 Perfectamente alineado; ninguna contradicción directa

════════════════════════════════════════

Para CADA dimensión debes incluir:
- "puntuacion": número entero entre 1 y 10
- "evidencia": cita textual EXACTA de la respuesta IA (máx. 30 palabras) que justifica la puntuación
- "justificacion": explicación en máx. 2 oraciones de por qué esa puntuación

Para "completitud" añade también:
- "puntos_cubiertos": lista de puntos clave de la ground truth que SÍ aparecen en la respuesta IA
- "puntos_omitidos": lista de puntos clave de la ground truth que NO aparecen en la respuesta IA

Para "veracidad_factual" añade también:
- "contradicciones": lista de contradicciones directas encontradas (vacía [] si no hay ninguna)

Devuelve EXACTAMENTE este JSON (sin texto fuera de él):
{{
  "relevancia": {{
    "puntuacion": <1-10>,
    "evidencia": "<cita de la respuesta IA>",
    "justificacion": "<máx. 2 oraciones>"
  }},
  "completitud": {{
    "puntuacion": <1-10>,
    "evidencia": "<cita de la respuesta IA>",
    "puntos_cubiertos": ["..."],
    "puntos_omitidos": ["..."],
    "justificacion": "<máx. 2 oraciones>"
  }},
  "coherencia": {{
    "puntuacion": <1-10>,
    "evidencia": "<cita de la respuesta IA>",
    "justificacion": "<máx. 2 oraciones>"
  }},
  "veracidad_factual": {{
    "puntuacion": <1-10>,
    "evidencia": "<cita de la respuesta IA>",
    "contradicciones": ["..."],
    "justificacion": "<máx. 2 oraciones>"
  }}
}}"""


def judge_once(client: OpenAI, qa: dict) -> dict | None:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(
                    qa["pregunta"], qa["respuesta_ia"], qa["ground_truth"]
                )},
            ],
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"    Error in judge pass: {e}")
        return None


def valid_score(value) -> int | None:
    """Return the score as an int if it is within 1-10, else None."""
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    return score if 1 <= score <= 10 else None


def aggregate_passes(passes: list[dict]) -> dict:
    result = {}

    for dim_key, dim_name in DIMENSIONS.items():
        scored = []
        for p in passes:
            block = p.get(dim_key)
            if isinstance(block, dict):
                score = valid_score(block.get("puntuacion"))
                if score is not None:
                    scored.append((score, block))

        if not scored:
            result[dim_name] = {"score": None, "justification": "No valid data"}
            continue

        scores = [s for s, _ in scored]
        median = statistics.median(scores)
        # Representative pass: the one whose score is closest to the median
        _, rep = min(scored, key=lambda item: abs(item[0] - median))

        entry = {
            "score": median,
            "pass_scores": scores,
            "evidence": rep.get("evidencia", ""),
            "justification": rep.get("justificacion", ""),
        }
        if dim_key == "completitud":
            entry["covered_points"] = rep.get("puntos_cubiertos", [])
            entry["omitted_points"] = rep.get("puntos_omitidos", [])
        if dim_key == "veracidad_factual":
            entry["contradictions"] = rep.get("contradicciones", [])

        result[dim_name] = entry

    return result


def evaluate_answer(client: OpenAI, qa: dict) -> dict:
    qid = qa["id"]
    print(f"  Question {qid}: running {N_PASSES} passes...")

    passes = []
    for i in range(N_PASSES):
        print(f"    pass {i + 1}/{N_PASSES}...", end=" ", flush=True)
        res = judge_once(client, qa)
        if isinstance(res, dict):
            passes.append(res)
            print("ok")
        else:
            print("failed (skipped)")
        time.sleep(0.8)  # basic rate limiting

    if not passes:
        return {"id": qid, "question": qa["pregunta"], "evaluations": {},
                "mean_score": None, "n_valid_passes": 0}

    evaluations = aggregate_passes(passes)

    scores = [e["score"] for e in evaluations.values() if e.get("score") is not None]
    mean_score = round(sum(scores) / len(scores), 2) if scores else None

    return {
        "id": qid,
        "question": qa["pregunta"],
        "evaluations": evaluations,
        "mean_score": mean_score,
        "n_valid_passes": len(passes),
    }
