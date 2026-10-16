SYSTEM_PROMPT = """Eres un evaluador experto en historia medieval hispana.
Tu tarea es evaluar respuestas de un sistema de IA a preguntas sobre historia visigoda.
Sé riguroso, justo y basa TODAS las puntuaciones exclusivamente en la evidencia textual proporcionada.
Responde SIEMPRE con JSON válido. Ningún texto fuera del JSON."""


def build_prompt(pregunta: str, respuesta_ia: str, ground_truth: str) -> str:
    return f"""Evalúa la siguiente respuesta en CUATRO dimensiones. Devuelve un único objeto JSON.

════════════════════════════════════════
PREGUNTA:
{pregunta}

RESPUESTA DE LA IA (a evaluar):
{respuesta_ia}

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


def una_pasada(client: OpenAI, qa: dict) -> dict | None:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,         
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(
                    qa["pregunta"], qa["respuesta_ia"], qa["ground_truth"]
                )},
            ],
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception as e:
        print(f"    ⚠️  Error en pasada: {e}")
        return None


def mediana_entera(valores: list[int]) -> float:
    return statistics.median(valores)


def agregar_pasadas(pasadas: list[dict]) -> dict:

    dims = ["relevancia", "completitud", "coherencia", "veracidad_factual"]
    resultado = {}

    for dim in dims:
        puntuaciones = [p[dim]["puntuacion"] for p in pasadas if p and dim in p]
        if not puntuaciones:
            resultado[dim] = {"puntuacion": None, "justificacion": "Sin datos"}
            continue

        med = mediana_entera(puntuaciones)
        # pasada representativa: la más cercana a la mediana
        rep = min(pasadas, key=lambda p: abs(p[dim]["puntuacion"] - med) if p else float("inf"))

        entrada = {
            "puntuacion": med,
            "puntuaciones_pasadas": puntuaciones,
            "evidencia":      rep[dim].get("evidencia", ""),
            "justificacion":  rep[dim].get("justificacion", ""),
        }
        if dim == "completitud":
            entrada["puntos_cubiertos"] = rep[dim].get("puntos_cubiertos", [])
            entrada["puntos_omitidos"]  = rep[dim].get("puntos_omitidos", [])
        if dim == "veracidad_factual":
            entrada["contradicciones"] = rep[dim].get("contradicciones", [])

        resultado[dim] = entrada

    return resultado


def evaluar_respuesta(client: OpenAI, qa: dict) -> dict:
    pid = qa["id"]
    print(f"  Pregunta {pid} — ejecutando {N_PASADAS} pasadas...")

    pasadas = []
    for i in range(N_PASADAS):
        print(f"    pasada {i+1}/{N_PASADAS}...", end=" ", flush=True)
        res = una_pasada(client, qa)
        if res:
            pasadas.append(res)
            print("ok")
        else:
            print("error (se omite)")
        time.sleep(0.8)

    if not pasadas:
        return {"id": pid, "pregunta": qa["pregunta"],
                "evaluaciones": {}, "puntuacion_media": None}

    evaluaciones = agregar_pasadas(pasadas)

    puntuaciones = [evaluaciones[d]["puntuacion"] for d in evaluaciones
                    if evaluaciones[d].get("puntuacion") is not None]
    media = round(sum(puntuaciones) / len(puntuaciones), 2) if puntuaciones else None

    return {
        "id": pid,
        "pregunta": qa["pregunta"],
        "evaluaciones": evaluaciones,
        "puntuacion_media": media,
        "n_pasadas_validas": len(pasadas),
    }
