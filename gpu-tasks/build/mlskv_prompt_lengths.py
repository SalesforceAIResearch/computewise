"""Prompt token lengths of the MLS-Bench KV workloads, built with the evaluators' own templates.

usage: <python with tokenizers==0.22.2> -I build/mlskv_prompt_lengths.py <download dir> <out json>

Output kept at survey/mlskv-prompt-lengths-2026-10-23.json; build/gen_mlskv_serving.py reads it for the records' memory.
The download dir holds Qwen/Qwen2.5-3B-Instruct tokenizer.json (as qwen-tokenizer.json) at aa8e7253, THUDM/LongBench
data.zip (as longbench_data.zip) at 5e628be4, THUDM/LongBench-v2 data.json (as lbv2.json) at 2b48e494 and
opencompass/NeedleBench PaulGrahamEssays.jsonl at 651d7c8f: the revisions the task stages.

The downloaded files are untrusted data: this script lives outside their directory and only reads them. Templates are copied
verbatim from custom_selection_eval.py / custom_quant_eval.py (registry artifacts 236d78dcda10 and e4a833bd5279); NeedleBench's
prompt is build_needlebench_examples's; the selection evaluator truncates to max_position_embeddings - max_new_tokens - 1, the
quant evaluator does not truncate (prefill_prompt tokenises the whole prompt). GSM8K prompts are short and are not measured here.
"""
import io
import json
import sys
import zipfile

from tokenizers import Tokenizer

D, OUT = sys.argv[1], sys.argv[2]
tok = Tokenizer.from_file(f"{D}/qwen-tokenizer.json")
MAXPOS = 32768

LB = {
    "hotpotqa_e": ("Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
                   "The following are given passages.\n{context}\n\nQuestion: {input}\nAnswer:"),
    "passage_retrieval_en_e": ("Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the "
                               "abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number "
                               "of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", "
                               "\"Paragraph 2\", etc.\n\nThe answer is: "),
    "repobench-p_e": "Please complete the code given below.\n{context}{input}Next line of code:\n",
}
V2 = ("Please read the following text and answer the question below.\n\n{context}\n\nWhat is the correct answer to this question: "
      "{question}\nChoices:\n(A) {choice_A}\n(B) {choice_B}\n(C) {choice_C}\n(D) {choice_D}\nFormat your response as follows: "
      "\"The correct answer is (insert answer here)\".")
NEW = {"hotpotqa_e": 32, "passage_retrieval_en_e": 32, "repobench-p_e": 64, "longbench_v2": 128, "needlebench_niah": 32}
NEEDLE = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
NQ = "The best thing to do in San Francisco is: "


def stats(lengths, new):
    lim = MAXPOS - new - 1
    return {"n": len(lengths), "max": max(lengths), "mean": round(sum(lengths) / len(lengths), 1),
            "p50": sorted(lengths)[len(lengths) // 2], "over_selection_limit": sum(1 for x in lengths if x > lim),
            "selection_limit": lim, "max_after_selection_truncation": min(max(lengths), lim)}


out = {}
with zipfile.ZipFile(f"{D}/longbench_data.zip") as z:
    for name, tpl in LB.items():
        member = next(m for m in z.namelist() if m.endswith(f"{name}.jsonl"))
        rows = [json.loads(l) for l in io.TextIOWrapper(z.open(member), encoding="utf-8") if l.strip()]
        lens = [len(tok.encode(tpl.format(context=r["context"], input=r["input"])).ids) for r in rows]
        out[name] = stats(lens, NEW[name])
        print(name, out[name], flush=True)
rows = json.load(open(f"{D}/lbv2.json"))
lens = [len(tok.encode(V2.format(context=str(r["context"]).strip(), question=str(r["question"]).strip(),
                                  choice_A=str(r["choice_A"]).strip(), choice_B=str(r["choice_B"]).strip(),
                                  choice_C=str(r["choice_C"]).strip(), choice_D=str(r["choice_D"]).strip())).ids) for r in rows]
out["longbench_v2"] = stats(lens, NEW["longbench_v2"])
print("longbench_v2", out["longbench_v2"], flush=True)
hay = []
for l in open(f"{D}/PaulGrahamEssays.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    t = next((r[k] for k in ("text", "content", "English", "english") if isinstance(r.get(k), str) and r[k].strip()), None)
    if t is None:
        t = next((v for v in r.values() if isinstance(v, str) and v.strip()), "")
    if t:
        hay.append(t)
lens = []
for h in hay:
    for depth in (0.10, 0.50, 0.90):
        s = int(len(h) * depth)
        ctx = h[:s] + "\n" + NEEDLE + "\n" + h[s:]
        p = ("A single relevant sentence is hidden in the following long document. Read the document carefully and answer the "
             f"retrieval question with the exact phrase.\n\n{ctx}\n\nQuestion: {NQ}\nAnswer:")
        lens.append(len(tok.encode(p).ids))
out["needlebench_niah"] = stats(lens, NEW["needlebench_niah"])
print("needlebench_niah", out["needlebench_niah"], flush=True)
json.dump(out, open(OUT, "w"), indent=1)
