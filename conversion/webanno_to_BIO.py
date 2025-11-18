import os
import sys

def convert_to_bio(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as out_f:
        prev_entities = {}

        columns = ['FOOD', 'DESCR', 'COOKING_CREATION', 'CURE',  'INGESTION', 'INGR', 'PR']
        out_f.write("TOKEN\t" + "\t".join(columns) + "\n")

        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = line.split('\t')
            if len(parts) < 4:
                continue
            
            token = parts[2]
            labels = parts[3].split('|')

        
            label_columns = {col: [] for col in columns}

            for label in labels:
                if label == "_" or label == "O":
                    continue

                if "[" in label:
                    base_label = label.split("[")[0]
                    entity_id = label.split("[")[1].rstrip("]")
                    prefix = "B-" if entity_id not in prev_entities else "I-"
                    prev_entities[entity_id] = True
                    bio_label = f"{prefix}{base_label}"
                else:
                    base_label = label
                    bio_label = f"B-{base_label}"

                
                if base_label.startswith("FOOD"):
                    label_columns['FOOD'].append(bio_label)
                elif base_label.startswith("PR"):
                    label_columns['PR'].append(bio_label)
                elif base_label.startswith("INGR"):
                    label_columns['INGR'].append(bio_label)
                elif base_label.startswith("INGESTION"):
                    label_columns['INGESTION'].append(bio_label)
                elif base_label.startswith("CURE"):
                    label_columns['CURE'].append(bio_label)
                elif base_label.startswith("DESCR"):
                    label_columns['DESCR'].append(bio_label)
                elif base_label.startswith("COOKING_CREATION"):
                    label_columns['COOKING_CREATION'].append(bio_label)

            
            final_labels = []
            for col in columns:
                if label_columns[col]:
                    final_labels.append("|".join(label_columns[col]))
                else:
                    final_labels.append("O")

            out_f.write(f"{token}\t" + "\t".join(final_labels) + "\n")


def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    files = [f for f in os.listdir(input_folder) if f.endswith(".tsv")]
    print(f"Found {len(files)} .tsv files in {input_folder}")
    
    for filename in files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".tsv", "_bio.tsv"))
        convert_to_bio(input_path, output_path)
        print(f"Converted: {filename} -> {output_path}")


input_folder = 'webanno_text'
output_folder = 'BIO_human_annotation'
process_folder(input_folder, output_folder)