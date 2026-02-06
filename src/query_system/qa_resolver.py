from typing import List, Dict, Tuple


def _build_context(retrieved_results: List[Dict], top_k: int = 3) -> str:
    top = sorted(retrieved_results, key=lambda r: r["similarity_score"], reverse=True)[:top_k]
    return "\n\n".join([f"Clause: {t['clause']}" for t in top])


def _extract_answer_from_context(query: str, retrieved_results: List[Dict]) -> str:
    q = query.lower()
    top = sorted(retrieved_results, key=lambda r: r["similarity_score"], reverse=True)[:3]
    clauses = [t["clause"] for t in top]
    text = " ".join(clauses)
    if any(k in q for k in ["liability", "limit", "cap"]):
        for c in clauses:
            if "liability" in c.lower():
                return c
    if any(k in q for k in ["terminate", "termination", "convenience", "notice"]):
        for c in clauses:
            if "terminate" in c.lower() or "termination" in c.lower():
                return c
    if any(k in q for k in ["warranty", "warranties", "disclaim"]):
        for c in clauses:
            if "warrant" in c.lower() or "disclaim" in c.lower():
                return c
    return clauses[0] if clauses else ""


def _confidence(answer: str, retrieved_results: List[Dict]) -> float:
    context = _build_context(retrieved_results, top_k=3)
    if not context:
        return 0.5
    overlap = sum(1 for w in answer.split() if w.lower() in context.lower().split())
    ratio = overlap / max(len(answer.split()), 1)
    top_sim = max((r["similarity_score"] for r in retrieved_results), default=0.5)
    return float(0.5 * ratio + 0.5 * top_sim)


def generate_answer(query: str, retrieved_results: List[Dict], full_contract_text: str | None = None) -> Tuple[str, float]:
    context = _build_context(retrieved_results, top_k=3)
    if not context:
        return "I do not have sufficient information in the contract to answer this question.", 0.5
    answer = _extract_answer_from_context(query, retrieved_results)
    if not answer or len(answer.split()) < 5:
        return "I do not have sufficient information in the contract to answer this question.", 0.6
    conf = _confidence(answer, retrieved_results)
    return answer, conf
