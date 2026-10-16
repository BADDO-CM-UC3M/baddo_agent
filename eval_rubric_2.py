import argparse
import glob
import hashlib
import json
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# CONFIGURACIÓN — idéntica a evaluador_v2.py (solo cambia la rúbrica)
# ---------------------------------------------------------------------------

MODEL = "gpt-4.1-mini"
TEMPERATURE = 0
MAX_REINTENTOS = 3
CLAVE_RESULTADO = "LLMJudge_R2"
PATRON_FICHEROS = "resultados_ragas_*_run*.json"

DIMS = [
    "exactitud_historica",
    "cumplimiento_tarea",
    "precision_epistemica",
    "razonamiento_sintesis",
    "relevancia_foco",
]

# ---------------------------------------------------------------------------
# RÚBRICA 2 — texto literal. Debe quedar CONGELADO antes de ejecutar.
# Si los coautores la modifican, se pega aquí la versión final; el hash cambia
# y queda registrado en cada resultado.
# ---------------------------------------------------------------------------

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

RUBRICA_SHA256 = hashlib.sha256(
    (SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE).encode("utf-8")
).hexdigest()


# ---------------------------------------------------------------------------
# LÓGICA DE EVALUACIÓN
# ---------------------------------------------------------------------------

def _validar(raw: dict) -> dict:
    """Comprueba la estructura y que cada puntuación sea un entero 0-10."""
    evaluaciones = {}
    for dim in DIMS:
        bloque = raw.get(dim)
        if not isinstance(bloque, dict) or "puntuacion" not in bloque:
            raise ValueError(f"falta la dimensión {dim}")
        score = bloque["puntuacion"]
        if isinstance(score, float) and score.is_integer():
            score = int(score)
        if not isinstance(score, int) or not 0 <= score <= 10:
            raise ValueError(f"puntuación inválida en {dim}: {score!r}")
        evaluaciones[dim] = {
            "puntuacion": score,
            "justificacion": bloque.get("justificacion", ""),
        }
    return evaluaciones


def evaluar_rubrica2(client: OpenAI, pregunta: str, respuesta: str,
                     ground_truth: str) -> dict:
    """Una pasada con la rúbrica 2. Devuelve el bloque a guardar en LLMJudge_R2."""
    ultimo_error = None
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(
                        question=pregunta,
                        reference_answer=ground_truth,
                        candidate_answer=respuesta,
                    )},
                ],
            )
            raw = json.loads(response.choices[0].message.content.strip())
            evaluaciones = _validar(raw)
            scores = [evaluaciones[d]["puntuacion"] for d in DIMS]
            return {
                "puntuacion_media": round(sum(scores) / len(scores), 2),
                "error_grave": bool(raw.get("error_grave", False)),
                "valoracion_general": raw.get("valoracion_general", ""),
                "evaluaciones": evaluaciones,
                "modelo": MODEL,
                "temperature": TEMPERATURE,
                "rubrica_sha256": RUBRICA_SHA256,
                "intentos": intento,
            }
        except Exception as e:  # noqa: BLE001 — se reintenta y se registra
            ultimo_error = e
            print(f"    ⚠️  intento {intento}/{MAX_REINTENTOS}: {e}")
            time.sleep(2 * intento)
    return {"puntuacion_media": None, "error": str(ultimo_error),
            "rubrica_sha256": RUBRICA_SHA256}


# ---------------------------------------------------------------------------
# BATCH SOBRE LOS RUNS
# ---------------------------------------------------------------------------

def nombre_config(ruta: str) -> str:
    """resultados_ragas_gpt5mini_ragsimple_run2.json -> gpt5mini_ragsimple"""
    m = re.match(r"resultados_ragas_(.+)_run\d+\.json$", os.path.basename(ruta))
    return m.group(1) if m else os.path.basename(ruta)


def procesar_fichero(client, ruta, limite, force, dry_run):
    with open(ruta, encoding="utf-8") as fh:
        datos = json.load(fh)

    items = datos[:limite] if limite else datos
    hechos = saltados = errores = 0
    for i, item in enumerate(items, 1):
        previo = item.get(CLAVE_RESULTADO) or {}
        if (not force and previo.get("puntuacion_media") is not None
                and previo.get("rubrica_sha256") == RUBRICA_SHA256):
            saltados += 1
            continue

        print(f"  P{i:>2}/{len(items)} ...", end=" ", flush=True)
        res = evaluar_rubrica2(client, item["pregunta"],
                               item["respuesta_generada"], item["ground_truth"])
        if res.get("puntuacion_media") is None:
            errores += 1
            print("ERROR")
        else:
            hechos += 1
            print(f"R2={res['puntuacion_media']:.2f}  "
                  f"(R1={item.get('LLMJudge', {}).get('puntuacion_media')})")

        if dry_run:
            print(json.dumps(res, ensure_ascii=False, indent=2)[:1500])
            continue

        item[CLAVE_RESULTADO] = res
        # guardado incremental (reanudable)
        tmp = ruta + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(datos, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, ruta)
        time.sleep(0.8)

    print(f"  → evaluadas {hechos}, ya existentes {saltados}, errores {errores}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="*",
                    help="Subconjunto de configuraciones (p.ej. gpt5mini_ragsimple)")
    ap.add_argument("--limite", type=int, default=None,
                    help="Evaluar solo las N primeras preguntas de cada fichero")
    ap.add_argument("--force", action="store_true",
                    help="Re-evaluar aunque ya exista LLMJudge_R2")
    ap.add_argument("--dry-run", action="store_true",
                    help="Llama al juez pero no escribe en los JSON")
    ap.add_argument("--exportar-rubrica", action="store_true",
                    help="Escribe rubrica2_prompt.txt (para publicarla) y sale")
    args = ap.parse_args()

    if args.exportar_rubrica:
        with open("rubrica2_prompt.txt", "w", encoding="utf-8") as fh:
            fh.write(f"# Modelo juez: {MODEL} | temperature: {TEMPERATURE} | "
                     f"response_format: json_object | 1 pasada\n")
            fh.write(f"# SHA-256: {RUBRICA_SHA256}\n\n")
            fh.write("=== PROMPT DE SISTEMA ===\n\n" + SYSTEM_PROMPT + "\n\n")
            fh.write("=== PLANTILLA DEL MENSAJE DE USUARIO ===\n\n" + USER_TEMPLATE + "\n")
        print(f"Rúbrica exportada a rubrica2_prompt.txt (sha256 {RUBRICA_SHA256[:12]}…)")
        return

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY no encontrada en .env ni en el entorno")
    client = OpenAI()

    ficheros = sorted(glob.glob(PATRON_FICHEROS))
    if args.configs:
        ficheros = [f for f in ficheros if nombre_config(f) in args.configs]
    if not ficheros:
        raise SystemExit("No hay ficheros que evaluar.")

    print(f"Rúbrica 2 | juez {MODEL} | temperature {TEMPERATURE} | "
          f"sha256 {RUBRICA_SHA256[:12]}…")
    print(f"{len(ficheros)} ficheros\n")
    for ruta in ficheros:
        print(f"[{nombre_config(ruta)}] {ruta}")
        procesar_fichero(client, ruta, args.limite, args.force, args.dry_run)
        print()


if __name__ == "__main__":
    main()
