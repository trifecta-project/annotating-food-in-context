import pandas as pd

gold_file = "human_annotation" #file human annotation
pred_file = "llm_annotation" #file LLM annotation

gold = pd.read_csv(gold_file, sep="\t")
pred = pd.read_csv(pred_file, sep="\t")


if len(gold) != len(pred):
    print(f"different tokens: gold={len(gold)}, pred={len(pred)}")


misaligned = []

for i, (g_tok, p_tok) in enumerate(zip(gold["token"], pred["token"])):
    if g_tok != p_tok:
        misaligned.append((i, g_tok, p_tok))

if misaligned:
    print(f"{len(misaligned)}tokens not aligned:\n")
    for idx, g_tok, p_tok in misaligned:
        print(f"line {idx+1}: gold='{g_tok}'  pred='{p_tok}'")
else:
    print("all tokens aligned")