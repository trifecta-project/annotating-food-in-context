import re
import pandas as pd
import argparse

def parse_annotations(text):
    frames = ["FOOD", "DESCR", "COOKING_CREATION", "CURE", "INGESTION", "INGR", "PR"]
    pattern = r"(\S+?)<([^>]*)>"
    matches = re.findall(pattern, text)

    rows = []

    for token, tags in matches:
        ann = {frame: "O" for frame in frames}
        if tags != "O":
            for tag in tags.split(","):
                tag = tag.strip()
                if not tag:
                    continue

                prefix = tag[:2]  # B- or I-

                # FOOD
                if "FOOD_" in tag or tag.startswith(("B-FOOD", "I-FOOD")):
                    fe = tag.split("FOOD_", 1)[-1] if "FOOD_" in tag else ""
                    ann["FOOD"] = prefix + "FOOD" + ("_" + fe if fe else "")
                # DESCR
                elif "DESCR_" in tag or tag.startswith(("B-DESCR", "I-DESCR")):
                    fe = tag.split("DESCR_", 1)[-1] if "DESCR_" in tag else ""
                    ann["DESCR"] = prefix + "DESCR" + ("_" + fe if fe else "")
                # PR
                elif "PR_" in tag or tag.endswith(("B-PR_", "I-PR")):
                    fe = tag.split("PR_", 1)[-1] if "PR_" in tag else ""
                    ann["PR"] = prefix + "PR" + ("_" + fe if fe else "")
                # COOK
                    elif "COOKING_CREATION_" in tag or tag.startswith(("B-COOKING_CREATION", "I-COOKING_CREATION")):
                    fe = tag.split("COOKING_CREATION_", 1)[-1] if "COOKING_CREATION_" in tag else ""
                    ann["CURE"] = prefix + "CURE" + ("_" + fe if fe else "")
                # CURE   
                elif "CURE_" in tag or tag.startswith(("B-CURE", "I-CURE")):
                    fe = tag.split("CURE_", 1)[-1] if "CURE_" in tag else ""
                    ann["CURE"] = prefix + "CURE" + ("_" + fe if fe else "")
                # INGESTION
                elif "INGESTION_" in tag or tag.startswith(("B-INGESTION", "I-INGESTION")):
                    fe = tag.split("INGESTION_", 1)[-1] if "INGESTION_" in tag else ""
                    ann["INGESTION"] = prefix + "INGESTION" + ("_" + fe if fe else "")
                # INGR
                elif "INGR_" in tag or tag.startswith(("B-INGR", "I-INGR")):
                    fe = tag.split("INGR_", 1)[-1] if "INGR_" in tag else ""
                    ann["INGR"] = prefix + "INGR" + ("_" + fe if fe else "")

        row = {"token": token}
        row.update(ann)
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Convert annotated text into TSV format with FEs")
    parser.add_argument("--input", required=True, help="Input filen(format <token><tag>)")
    parser.add_argument("--output", required=True, help="TSV output")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    df = parse_annotations(text)
    df.to_csv(args.output, sep="\t", index=False)
    print(f"tsv saved as: {args.output}")


if __name__ == "__main__":
    main()
