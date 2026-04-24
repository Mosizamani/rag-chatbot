"""
RocketRide DAP client helpers: engine check, RAG (AWS D1.1 / ICC), vision weld
defects via `agents.pipe` and `vision_agent.pipe`, and orchestrated Field Inspection
Reports (RAG + vision synthesis and Agent 3 formatting on `agents.pipe`).
"""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from rocketride import RocketRideClient
from rocketride.schema import Question

load_dotenv()

# Local engine URL used by the VS Code / CLI RocketRide DAP server
_DEFAULT_ROCKETRIDE_URI = "http://localhost:5565"

# Pipeline files (same directory as this module)
_DIR = Path(__file__).resolve().parent
AGENTS_RAG_PIPE = _DIR / "agents.pipe"
VISION_PIPE = _DIR / "vision_agent.pipe"

_DEFAULT_RAG_PIPE_TARGET = "RAG_Node"
_DEFAULT_VISION_PIPE_TARGET = "Vision_Node"

_CODE_STANDARDS = {
    "aws_d11": "AWS D1.1",
    "icc": "International Building Code (ICC/IBC family)",
}

_USE_CHAT_RAG = (os.getenv("RAG_PIPE_USE_CHAT") or "").lower() in ("1", "true", "yes")


def _auth() -> str:
    return (os.getenv("ROCKETRIDE_APIKEY") or os.getenv("ROCKETRIDE_API_KEY") or "").strip()


def _uri() -> str:
    u = (os.getenv("ROCKETRIDE_URI") or _DEFAULT_ROCKETRIDE_URI).strip()
    return u.rstrip("/")


def _rag_target_provider() -> str:
    if _USE_CHAT_RAG:
        return "chat"
    return (os.getenv("RAG_PIPE_PROVIDER") or _DEFAULT_RAG_PIPE_TARGET).strip() or "chat"


def _vision_target_provider() -> str:
    if (os.getenv("VISION_PIPE_USE_WEBHOOK") or "").lower() in ("1", "true", "yes"):
        return "webhook"
    return (os.getenv("VISION_PIPE_PROVIDER") or _DEFAULT_VISION_PIPE_TARGET).strip() or "webhook"


def _connect_client() -> RocketRideClient:
    return RocketRideClient(uri=_uri(), auth=_auth())


def _load_text_snippets_for_codes(user_notes: str, limit: int = 6) -> List[Dict[str, Any]]:
    """Pre-fetch Mongo text relevant to AWS D1.1 / ICC using the same SRV as setup_mongo.py."""
    srv = (os.getenv("MONGODB_ATLAS_SRV") or "").strip()
    if not srv:
        return []
    try:
        from pymongo import MongoClient
    except ImportError:
        return []
    db_name = (os.getenv("MONGODB_DB_NAME") or "weld_inspection").strip()
    coll_name = (os.getenv("MONGODB_COLLECTION") or "documents").strip()
    client = MongoClient(srv, serverSelectionTimeoutMS=8_000)
    try:
        coll = client[db_name][coll_name]
        code_regex = "AWS|D1\\.1|D1-1|ICC|IBC|IBC-2021|IBC-2018|building\\s*code|IBC/ICC"
        filters: List[Dict[str, Any]] = [
            {"$or": [
                {"metadata.code_standard": {"$regex": code_regex, "$options": "i"}},
                {"metadata.title": {"$regex": code_regex, "$options": "i"}},
                {"page_content": {"$regex": code_regex, "$options": "i"}},
            ]}
        ]
        if (user_notes or "").strip():
            n = re.escape(user_notes.strip()[: 4000])
            filters[0]["$or"].append({"page_content": {"$regex": n, "$options": "i"}})
        cur = (
            coll.find(
                filters[0],
                {"page_content": 1, "metadata": 1, "score": 1},
            )
            .limit(limit)
        )
        return [
            {
                "page_content": d.get("page_content") or "",
                "metadata": d.get("metadata") or {},
            }
            for d in cur
        ]
    except Exception:  # noqa: BLE001 — return empty on any connection/query issue
        return []
    finally:
        try:
            client.close()
        except Exception:
            pass


def _format_mongo_brief(docs: List[Dict[str, Any]]) -> str:
    if not docs:
        return ""
    parts: List[str] = []
    for i, d in enumerate(docs, 1):
        m = d.get("metadata") or {}
        head = f"[{i}] " + (m.get("title") or m.get("code_standard") or m.get("parent") or "chunk")
        body = (d.get("page_content") or "")[: 1200]
        parts.append(f"{head}\n{body}\n")
    return "\n".join(parts).strip()


def _normalize_answers_to_json(body: Optional[Dict[str, Any]]) -> Any:
    """Return first structured answer, or a plain dict of the full body for JSON-only callers."""
    if not body:
        return None
    if "answers" in body and body["answers"] is not None:
        a = body["answers"]
        if isinstance(a, list) and a:
            if len(a) == 1 and a[0] is not None:
                return a[0]
            return a
    return body


def _as_structured_error(stage: str, err: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "stage": stage,
        "error": err,
    }


def _envelope_ok(payload: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": True, **extra, "result": payload}


async def get_rag_context(
    query: str,
    user_notes: str,
) -> Dict[str, Any]:
    """
    Run `agents.pipe` and send a structured Question to the RAG path (RAG_Node).
    `user_notes` and retrieved MongoDB snippets (AWS D1.1, ICC) are added as
    context so the vector path scopes to those building and welding code topics.
    """
    mongo_hits = _load_text_snippets_for_codes(user_notes)
    notes_context = f"User / inspector notes: {user_notes}\n" if (user_notes or "").strip() else ""
    code_scope = (
        f"Relevant code scope: { _CODE_STANDARDS['aws_d11'] } and { _CODE_STANDARDS['icc'] }."
    )
    brief = _format_mongo_brief(mongo_hits)
    if brief:
        code_scope = code_scope + "\n\nExcerpts from the MongoDB code corpus:\n" + brief

    q = Question(expectJson=True)
    q.addInstruction(
        "Citations and scope",
        "Ground answers in AWS D1.1 and ICC/IBC building code material when applicable. "
        "If a topic is not covered, say so explicitly.",
    )
    if notes_context or brief:
        q.addContext(notes_context + code_scope)
    else:
        q.addContext(code_scope)
    q.addExample(
        "List base-plate requirements from notes",
        {
            "citations": [{"code": "AWS D1.1", "clause": "…"}],
            "summary": "…",
            "limitations": "…",
        },
    )
    q.addQuestion(query)

    if not AGENTS_RAG_PIPE.is_file():
        return _as_structured_error("get_rag_context", f"Missing pipeline file: {AGENTS_RAG_PIPE}")

    body: Any = None
    client = _connect_client()
    prov = _rag_target_provider()
    try:
        await client.connect()
        use_result = await client.use(filepath=str(AGENTS_RAG_PIPE))
        token = use_result.get("token")
        if not token:
            return _as_structured_error("get_rag_context", "use() did not return a task token")
        objinfo: Dict[str, Any] = {"name": "rag_context_query", "size": 1}
        pipe = await client.pipe(
            token,
            objinfo,
            "application/rocketride-question",
            provider=prov,
        )
        try:
            await pipe.open()
            await pipe.write(bytes(q.model_dump_json(), "utf-8"))
            body = await pipe.close() or {}
        except Exception as exc:  # noqa: BLE001
            if pipe.is_opened:
                try:
                    await pipe.close()
                except Exception:
                    pass
            return _as_structured_error("get_rag_context", str(exc))
    except Exception as exc:  # noqa: BLE001
        return _as_structured_error("get_rag_context", str(exc))
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass

    parsed = _normalize_answers_to_json(body) if isinstance(body, dict) else body
    return _envelope_ok(
        {"query": query, "user_notes": user_notes, "code_standards": list(_CODE_STANDARDS.values())},
        {
            "pipeline": "agents.pipe",
            "target_provider": prov,
            "mongo_hits": mongo_hits,
            "raw": body,
            "answers_parsed": parsed,
        },
    )


async def detect_weld_defects(image_path: str) -> Dict[str, Any]:
    """
    Encode a local image as base64 (returned in JSON) and send raw image bytes
    to `vision_agent.pipe` (Vision_Node / webhook) for Gemini-2.0–based analysis
    of porosity, undercut, and slag. Final CWI JSON comes from the pipeline LLM
    (gemini-2_0-flash) in `vision_agent.pipe`.
    """
    path = Path(image_path)
    if not path.is_file():
        return _as_structured_error("detect_weld_defects", f"Not a file: {image_path}")

    raw = path.read_bytes()
    b64 = base64.standard_b64encode(raw).decode("ascii")
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "application/octet-stream"

    if not VISION_PIPE.is_file():
        return _as_structured_error("detect_weld_defects", f"Missing pipeline file: {VISION_PIPE}")

    client = _connect_client()
    prov = _vision_target_provider()
    size = max(len(raw), 1)
    body: Any = None
    try:
        await client.connect()
        use_result = await client.use(filepath=str(VISION_PIPE))
        token = use_result.get("token")
        if not token:
            return _as_structured_error("detect_weld_defects", "use() did not return a task token")
        objinfo: Dict[str, Any] = {
            "name": path.name,
            "size": size,
        }
        pipe = await client.pipe(
            token,
            objinfo,
            mime,
            provider=prov,
        )
        try:
            await pipe.open()
            await pipe.write(raw)
            body = await pipe.close() or {}
        except Exception as exc:  # noqa: BLE001
            if pipe.is_opened:
                try:
                    await pipe.close()
                except Exception:
                    pass
            return _as_structured_error("detect_weld_defects", str(exc))
    except Exception as exc:  # noqa: BLE001
        return _as_structured_error("detect_weld_defects", str(exc))
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass

    parsed: Any
    if isinstance(body, dict):
        parsed = _normalize_answers_to_json(body)
        try:
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    else:
        parsed = body
    return _envelope_ok(
        {
            "defects_analysis": parsed,
        },
        {
            "pipeline": "vision_agent.pipe",
            "image_path": str(path.resolve()),
            "mime_type": mime,
            "image_base64": b64,
            "target_provider": prov,
            "cwi_model": "gemini-2_0-flash",
            "defects_focus": ["porosity", "undercut", "slag"],
            "raw": body,
        },
    )


def _context_blob(label: str, data: Any, max_chars: int = 32_000) -> str:
    if data is None:
        return f"{label}: (none)\n"
    if isinstance(data, (dict, list)):
        try:
            s = json.dumps(data, default=str, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            s = str(data)
    else:
        s = str(data)
    if len(s) > max_chars:
        s = s[: max_chars - 3] + "..."
    return f"{label}:\n{s}\n\n"


def _llm_unwrap_text(answers_parsed: Any) -> str:
    """Return a string answer from a parsed pipe response (string, dict, or first list item)."""
    if answers_parsed is None:
        return ""
    if isinstance(answers_parsed, str):
        t = answers_parsed.strip()
        if t.startswith("{") or t.startswith("["):
            try:
                parsed = json.loads(answers_parsed)
                if isinstance(parsed, str):
                    return parsed.strip()
                if isinstance(parsed, dict) and "text" in parsed:
                    return str(parsed["text"]).strip()
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return t
    if isinstance(answers_parsed, dict):
        for key in (
            "polished_report",
            "field_inspection_report",
            "report",
            "text",
            "answer",
        ):
            v = answers_parsed.get(key)
            if v is not None and str(v).strip():
                return str(v).strip()
        return json.dumps(answers_parsed, default=str, ensure_ascii=False, indent=2)
    if isinstance(answers_parsed, list) and answers_parsed:
        return _llm_unwrap_text(answers_parsed[0])
    return str(answers_parsed).strip()


async def _run_agents_pipe(
    q: Question,
    *,
    stage: str,
) -> Dict[str, Any]:
    """Run a `Question` through `agents.pipe` (same path as RAG: chat → embed → Mongo → LLM)."""
    if not AGENTS_RAG_PIPE.is_file():
        return _as_structured_error(stage, f"Missing pipeline file: {AGENTS_RAG_PIPE}")
    prov = _rag_target_provider()
    body: Any = None
    client = _connect_client()
    try:
        await client.connect()
        use_result = await client.use(filepath=str(AGENTS_RAG_PIPE))
        token = use_result.get("token")
        if not token:
            return _as_structured_error(stage, "use() did not return a task token")
        objinfo: Dict[str, Any] = {"name": stage, "size": 1}
        pipe = await client.pipe(
            token,
            objinfo,
            "application/rocketride-question",
            provider=prov,
        )
        try:
            await pipe.open()
            await pipe.write(bytes(q.model_dump_json(), "utf-8"))
            body = await pipe.close() or {}
        except Exception as exc:  # noqa: BLE001
            if pipe.is_opened:
                try:
                    await pipe.close()
                except Exception:
                    pass
            return _as_structured_error(stage, str(exc))
    except Exception as exc:  # noqa: BLE001
        return _as_structured_error(stage, str(exc))
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass
    ap = _normalize_answers_to_json(body) if isinstance(body, dict) else body
    if isinstance(ap, str):
        try:
            j = json.loads(ap)
        except (json.JSONDecodeError, TypeError, ValueError):
            j = ap
        ap = j
    return {
        "ok": True,
        "stage": stage,
        "target_provider": prov,
        "pipeline": "agents.pipe",
        "raw": body,
        "answers_parsed": ap,
    }


def _orchestrator_parse_json(answers_parsed: Any) -> Optional[Dict[str, Any]]:
    d: Any = answers_parsed
    if isinstance(d, str):
        try:
            d = json.loads(d)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    if not isinstance(d, dict):
        return None
    return d


async def generate_final_report(
    rag_data: Any,
    vision_data: Any,
    inspector_notes: str,
) -> Dict[str, Any]:
    """
    Orchestrator: compare vision weld findings to RAG regulatory limits (AWS D1.1 / ICC),
    then Agent 3 (separate template) for grammar and formatting. Returns a professional
    Field Inspection Report: project header, visual findings, code references, and
    a final line that must read ``NON-COMPLIANT`` when a vision metric exceeds
    the applicable RAG-stated limit.
    """
    notes = (inspector_notes or "").strip()
    bundle = _context_blob("RAG and regulatory data", rag_data) + _context_blob("Vision (image) analysis", vision_data)
    if notes:
        bundle = f"Inspector / site notes:\n{notes}\n\n" + bundle

    q1 = Question(expectJson=True)
    q1.addInstruction(
        "Role: Field inspection orchestrator (high-reasoning)",
        "You are a certified welding/structural field-inspection lead. "
        "Compare the Vision agent findings (measured or scored defects: porosity, undercut, slag, etc.) "
        "against the limits, tolerances, and requirements stated or implied in the RAG / regulatory payload "
        f"({ _CODE_STANDARDS['aws_d11'] } and { _CODE_STANDARDS['icc'] }). "
        "If the Vision data indicates a defect, dimension, or severity that exceeds a limit established "
        "by the RAG material (or, when RAG is silent on a limit, you must not invent numeric limits; "
        "state the gap and use NEEDS_FURTHER_REVIEW or INCONCLUSIVE as appropriate), "
        "set compliance_status to the string 'NON-COMPLIANT' and explain exactly why. "
        "If Vision findings are within the RAG-sourced limits, use 'COMPLIANT'. "
        "Use clear professional language suitable for a formal report.",
    )
    q1.addInstruction(
        "Field Inspection Report (structure)",
        "In 'field_inspection_report', output Markdown with these sections in order: "
        "## Header (Project Information) — use any project/site info present in the inputs or state 'Not provided'; "
        "## Visual Findings — from the vision analysis, tables where helpful; "
        "## Regulatory Reference (AWS/ICC) — cite clauses/sections that apply from the RAG payload, not invented citations; "
        "## Final Compliance Status — one line beginning with: **COMPLIANT** or **NON-COMPLIANT** or **NEEDS_FURTHER_REVIEW** "
        "and echo compliance_status. "
        "The comparison rationale must make clear whether a vision-reported condition exceeds a RAG limit.",
    )
    q1.addContext(bundle)
    q1.addExample(
        "Synthesis",
        {
            "compliance_status": "NON-COMPLIANT",
            "rationale": "Porosity / aggregate defect severity X exceeds the acceptance limit Y stated under AWS D1.1 …",
            "compared_against": "E.g. AWS D1.1 table/clause or ICC/IBC reference as given in RAG (paraphrased, not fake clause numbers if absent).",
            "field_inspection_report": (
                "# Field Inspection Report\n\n## Header (Project Information)\n…\n\n## Visual Findings\n…\n\n"
                "## Regulatory Reference (AWS/ICC)\n…\n\n## Final Compliance Status\n**NON-COMPLIANT** — …\n"
            ),
        },
    )
    q1.addQuestion(
        "Produce the JSON only: 'compliance_status' (one of: COMPLIANT, NON-COMPLIANT, NEEDS_FURTHER_REVIEW, INCONCLUSIVE), "
        "optional 'rationale' and 'compared_against', and 'field_inspection_report' (full Markdown for the four sections).",
    )

    o = await _run_agents_pipe(q1, stage="orchestrate_report")
    if not o.get("ok"):
        return o

    parsed1 = o.get("answers_parsed")
    odict = _orchestrator_parse_json(parsed1) or {}
    if not odict.get("field_inspection_report") and _llm_unwrap_text(parsed1):
        odict = {
            "compliance_status": "INCONCLUSIVE",
            "rationale": "Unstructured model output was wrapped into the report field.",
            "compared_against": "",
            "field_inspection_report": _llm_unwrap_text(parsed1),
        }
    draft = (odict.get("field_inspection_report") or "").strip() or _llm_unwrap_text(parsed1)
    if not draft:
        return _as_structured_error("orchestrate_report", "Orchestrator returned no field_inspection_report text")
    com = (odict.get("compliance_status") or "").strip() or "INCONCLUSIVE"
    com_u = com.upper()
    if com_u == "NON-COMPLIANT" and "NON-COMPLIANT" not in draft and "Non-compliant" not in draft:
        draft = draft + "\n\n## Final Compliance Status\n**NON-COMPLIANT** — (See compliance_status in structured output.)\n"

    q2 = Question(expectJson=False)
    q2.addInstruction(
        "Agent 3: Grammar, clarity, and formatting (do not re-judge compliance)",
        "You are a technical editor and document formatter. Polish the given Field Inspection Report for professional tone, "
        "correct grammar, consistent headings, and clear Markdown. "
        "Do not change the factual or technical content: do not add measurements, do not remove NON-COMPLIANT flags, "
        "and do not weaken any stated code references. Do not re-run compliance; preserve **COMPLIANT** / **NON-COMPLIANT** "
        "/ **NEEDS_FURTHER_REVIEW** in the final section. Keep the four top-level section headings. "
        "Return only the polished report text, no preface or explanation.",
    )
    q2.addContext(draft)
    q2.addQuestion("Return the polished report only, as clean Markdown.")

    p = await _run_agents_pipe(q2, stage="format_report")
    if not p.get("ok"):
        return _envelope_ok(
            {
                "compliance_status": com,
                "field_inspection_report_draft": draft,
                "field_inspection_report": draft,
                "orchestrator_structured": odict or parsed1,
                "polish_failed": True,
                "polish_error": p,
            },
            {
                "pipeline": "agents.pipe",
                "orchestrator_target_provider": o.get("target_provider"),
                "orchestrator_raw": o.get("raw"),
            },
        )

    polished = _llm_unwrap_text(p.get("answers_parsed")).strip() or draft
    if not polished.strip():
        polished = draft
    return _envelope_ok(
        {
            "compliance_status": com,
            "orchestrator_structured": odict or parsed1,
            "field_inspection_report_draft": draft,
            "field_inspection_report": polished,
        },
        {
            "pipeline": "agents.pipe",
            "orchestrator_target_provider": o.get("target_provider"),
            "orchestrator_raw": o.get("raw"),
            "polish_target_provider": p.get("target_provider"),
            "polish_raw": p.get("raw"),
        },
    )


async def check_engine() -> bool:
    """Connect, ping, and disconnect. Returns True if the engine responds."""
    client = _connect_client()
    try:
        await client.connect()
        await client.ping()
        return True
    finally:
        await client.disconnect()


async def main() -> None:
    uri = _uri()
    print(f"Target RocketRide DAP: {uri}")
    try:
        ok = await check_engine()
    except Exception as exc:  # noqa: BLE001
        print(f"RocketRide engine is not reachable on {uri!r} ({type(exc).__name__}: {exc})")
    else:
        if ok:
            print(f"RocketRide engine is reachable on {uri!r}.")


if __name__ == "__main__":
    asyncio.run(main())
