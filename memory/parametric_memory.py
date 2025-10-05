import json
import argparse
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModel

from train_memory_retriever import MemoryRetrieverClassifier, _parse_plan, _pretty_plan


class CaseRetriever:
    def __init__(
        self,
        model_path: str,
        model_name: str = "princeton-nlp/sup-simcse-roberta-base",
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
        backbone = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
        self.model = MemoryRetrieverClassifier(backbone).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    @torch.inference_mode()
    def _score_batch(self, natural: List[str], icl: List[str]) -> torch.Tensor:
        t1 = self.tokenizer(icl, padding=True, truncation=True, return_tensors="pt")
        t2 = self.tokenizer(natural, padding=True, truncation=True, return_tensors="pt")
        ids1, mask1 = t1["input_ids"].to(self.device), t1["attention_mask"].to(self.device)
        ids2, mask2 = t2["input_ids"].to(self.device), t2["attention_mask"].to(self.device)
        logits = self.model(ids1, mask1, ids2, mask2)
        return torch.softmax(logits, dim=1)[:, 1]

    def retrieve(self, natural_prompt: str, icl_pool: List[str], metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        probs = self._score_batch([natural_prompt] * len(icl_pool), icl_pool)
        results = []
        for i, (prompt, score, meta) in enumerate(zip(icl_pool, probs, metadata)):
            results.append({
                "prompt": prompt,
                "score": float(score),
                "index": i,
                "case_label": meta.get("case_label", "unknown"),
                "case": meta.get("case", ""),
                "plan": meta.get("plan", None)
            })
        return results


def build_icl_text(case: str, plan) -> str:
    parts = ["[CASE]", str(case)]
    if plan is not None:
        pobj = _parse_plan(plan)
        parts += ["[PLAN]", _pretty_plan(pobj) if pobj is not None else str(plan)]
    return "\n".join(parts).strip()


def load_pool(path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    pool = []
    metadata = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            case = obj.get("case")
            if case is None:
                raise ValueError("Each line in pool jsonl must contain 'case' field")
            plan = obj.get("plan", None)
            pool.append(build_icl_text(case, plan))
            metadata.append({
                "case": case,
                "plan": plan,
                "case_label": obj.get("case_label", "unknown")
            })
    if not pool:
        raise ValueError("Pool is empty")
    return pool, metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--pool_jsonl", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    retriever = CaseRetriever(model_path=args.model_path)
    icl_pool, metadata = load_pool(args.pool_jsonl)
    ranked = retriever.retrieve(args.query, icl_pool, metadata)
    ranked.sort(key=lambda x: x["score"], reverse=True)

    topk = ranked[:args.topk] if 0 < args.topk < len(ranked) else ranked
    for i, item in enumerate(topk, 1):
        print(f"[{i}] score={item['score']:.4f} idx={item['index']} label={item['case_label']}")
        print(f"{item['prompt']}\n" + "-" * 60)


if __name__ == "__main__":
    main()