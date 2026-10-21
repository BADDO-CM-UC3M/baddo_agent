import json
import time

from openai import OpenAI

MODEL = "gpt-4.1-mini"
TEMPERATURE = 0
MAX_RETRIES = 3
RESULT_KEY = "LLMJudge_R2"

# Keys of the JSON returned by the judge (defined in the Spanish prompt) -> English output names
DIMENSIONS = {
    "exactitud_historica": "historical_accuracy",
    "cumplimiento_tarea": "task_fulfilment",
    "precision_epistemica": "epistemic_precision",
    "razonamiento_sintesis": "reasoning_synthesis",
    "relevancia_foco": "relevance_focus",
}


SYSTEM_PROMPT = """Vas a evaluar respuestas a preguntas de historia de nivel universitario sobre la Hispania visigoda y la historiografía que se ha ocupado de ella.

En cada caso tendrás la pregunta, una respuesta de referencia redactada por una persona experta en el tema y la respuesta que hay que evaluar. No sabrás qué sistema ha generado esa respuesta ni cómo funciona, y no hace falta que intentes adivinarlo.

Antes de puntuar, ten en cuenta lo siguiente.

La respuesta de referencia es el punto de comparación principal: indica qué contenido se espera y qué cuestiones tiene que tratar la respuesta. Pero no es una lista cerrada de todo lo que se puede decir. Si la respuesta incluye información que no está en la referencia, no la penalices, siempre que sea pertinente y no la contradiga. Tampoco la premies solo por aportar más contexto: el contexto adicional cuenta cuando ayuda de verdad a responder o a explicar lo que se pregunta, no por sí mismo.

No valores la extensión ni el estilo. Una respuesta no es mejor por ser larga o por estar escrita con mucha seguridad, ni por repetir las palabras de la referencia. Lo que importa es el contenido.

Ten presente la diferencia entre hechos, interpretaciones historiográficas e hipótesis. En este campo es habitual que una misma cuestión se lea de forma distinta según el autor, y una buena respuesta tiene que reflejarlo.

Puntúa por separado cada uno de estos cinco criterios, de 0 a 10.

1. Exactitud histórica y atribución
¿Son correctos los nombres, las fechas, los acontecimientos y las relaciones de causa? ¿Se atribuye a cada autor lo que realmente defiende? Baja la nota si hay errores de datos, si se pone en boca de un historiador una tesis que no es suya o si se presenta como un hecho probado lo que es una interpretación discutida, cuando esa diferencia importa para la pregunta.

2. Cumplimiento de la tarea
¿Responde a lo que se pregunta, incluidas todas sus partes? Para decidir qué tiene que aparecer, guíate por la propia pregunta y por los puntos de la referencia que la contestan. Una respuesta puede tener la nota máxima sin añadir contexto de fondo, así que no subas la nota por incluir más información de la necesaria.

3. Precisión epistémica
¿Distingue la respuesta entre lo que está establecido, lo que es la interpretación de un autor, lo que es una hipótesis y lo que no se sabe? Valora que matice las afirmaciones dudosas o debatidas y que no muestre una certeza o una precisión (cifras, fechas exactas) que las fuentes no permiten.

4. Razonamiento y síntesis
¿Relaciona los hechos y las posturas de los autores para construir una explicación coherente? Penaliza las inferencias que no se apoyan en nada, las contradicciones internas y las respuestas que se limitan a enumerar datos sin conectarlos.

5. Relevancia y foco
¿Se mantiene centrada en la pregunta y con el nivel de detalle adecuado? Penaliza las digresiones, el contexto que no hace falta, las repeticiones y el exceso de detalle que hace difícil encontrar la respuesta.

Como orientación para las puntuaciones:
  9-10  Excelente. Sin problemas de fondo, o solo detalles sin importancia.
  7-8   Buena. Correcta en conjunto, con alguna omisión o imprecisión menor.
  5-6   Irregular. Útil, pero con omisiones, errores, afirmaciones sin apoyo o fallos de razonamiento que se notan.
  3-4   Deficiente. Errores u omisiones importantes que debilitan claramente la respuesta.
  0-2   Incorrecta en lo fundamental, sin apoyo o no responde a la pregunta.

Marca "error_grave" como true si la respuesta tiene al menos un error que llevaría a un lector a una conclusión equivocada sobre lo que se pregunta (por ejemplo, una fecha o una atribución errónea en el punto central de la pregunta).

Justifica cada puntuación en una o dos frases, señalando aspectos concretos de la respuesta. No hace falta que expliques tu razonamiento paso a paso.

Responde solo con este JSON, con puntuaciones enteras:

{
  "exactitud_historica": {"puntuacion": 0, "justificacion": ""},
  "cumplimiento_tarea": {"puntuacion": 0, "justificacion": ""},
  "precision_epistemica": {"puntuacion": 0, "justificacion": ""},
  "razonamiento_sintesis": {"puntuacion": 0, "justificacion": ""},
  "relevancia_foco": {"puntuacion": 0, "justificacion": ""},
  "error_grave": false,
  "valoracion_general": ""
}"""

USER_TEMPLATE = """PREGUNTA:
{question}

RESPUESTA DE REFERENCIA:
{reference_answer}

RESPUESTA A EVALUAR:
{candidate_answer}"""


def parse_scores(raw: dict) -> dict:
    """Check that every dimension is present with an integer score in 0-10."""
    evaluations = {}
    for dim_key, dim_name in DIMENSIONS.items():
        block = raw.get(dim_key)
        if not isinstance(block, dict) or "puntuacion" not in block:
            raise ValueError(f"missing dimension: {dim_key}")
        score = block["puntuacion"]
        if isinstance(score, float) and score.is_integer():
            score = int(score)
        if not isinstance(score, int) or not 0 <= score <= 10:
            raise ValueError(f"invalid score for {dim_key}: {score!r}")
        evaluations[dim_name] = {
            "score": score,
            "justification": block.get("justificacion", ""),
        }
    return evaluations


def judge(client: OpenAI, question: str, answer: str, ground_truth: str) -> dict:
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(
                        question=question,
                        reference_answer=ground_truth,
                        candidate_answer=answer,
                    )},
                ],
            )
            raw = json.loads(response.choices[0].message.content)
            evaluations = parse_scores(raw)
            scores = [e["score"] for e in evaluations.values()]
            return {
                "mean_score": round(sum(scores) / len(scores), 2),
                "serious_error": bool(raw.get("error_grave", False)),
                "overall_assessment": raw.get("valoracion_general", ""),
                "evaluations": evaluations,
            }
        except Exception as e:
            last_error = e
            print(f"    attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(2 * attempt)
    return {"mean_score": None, "error": str(last_error)}


def evaluate_file(client: OpenAI, path: str) -> None:
    with open(path, encoding="utf-8") as fh:
        items = json.load(fh)

    for i, item in enumerate(items, 1):
        print(f"  {i}/{len(items)}", end=" ", flush=True)
        result = judge(client, item["pregunta"], item["respuesta_generada"],
                       item["ground_truth"])
        item[RESULT_KEY] = result
        print(result["mean_score"] if result["mean_score"] is not None else "ERROR")
        time.sleep(0.8)  # basic rate limiting

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=2)
