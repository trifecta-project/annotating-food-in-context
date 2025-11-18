import pandas as pd
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from collections import defaultdict

gold_file = "BIO_human_annotation/all_human.tsv"
pred_file = "BIO_deep_V3/all_deep.tsv"

gold = pd.read_csv(gold_file, sep="\t")
pred = pd.read_csv(pred_file, sep="\t")

frames = ["FOOD", "DESCR", "COOKING_CREATION", "CURE", "INGESTION", "INGR", "PR"]

def expand_multi_labels(label):
    if pd.isna(label) or label == 'O':
        return ['O']

    labels = [l.strip() for l in str(label).split('|') if l.strip()]
    return labels if labels else ['O']

def split_into_sentences(df, frame_col):
    sentences = []
    current_sentence = []
    
    for idx, row in df.iterrows():
        token = row['token']
        label = row[frame_col]
        current_sentence.append(label)
        
        if token in ['.', '!', '?', ';']:
            sentences.append(current_sentence)
            current_sentence = []
    
    
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences

def extract_label_types(labels_list):

    label_types = set()
    for sentence in labels_list:
        for label in sentence:
            
            expanded = expand_multi_labels(label)
            for single_label in expanded:
                if single_label != 'O':
                    
                    label_type = single_label.split('-', 1)[1] if '-' in single_label else single_label
                    label_types.add(label_type)
    return sorted(label_types)

def filter_labels_by_type(labels_list, target_type):
   
    filtered = []
    for sentence in labels_list:
        filtered_sentence = []
        for label in sentence:
            # Espandi le label multiple
            expanded = expand_multi_labels(label)
            
            
            matching_label = None
            for single_label in expanded:
                if single_label != 'O':
                    prefix = single_label.split('-')[0]  # B o I
                    label_type = single_label.split('-', 1)[1] if '-' in single_label else single_label
                    
                    if label_type == target_type:
                        matching_label = single_label
                        break
            
            
            filtered_sentence.append(matching_label if matching_label else 'O')
        
        filtered.append(filtered_sentence)
    return filtered

def count_entities_by_type(labels_list, target_type=None):
    
    count = 0
    for sentence in labels_list:
        in_entity = False
        for label in sentence:
            expanded = expand_multi_labels(label)
            
            
            has_begin = False
            for single_label in expanded:
                if single_label != 'O':
                    prefix = single_label.split('-')[0]
                    label_type = single_label.split('-', 1)[1] if '-' in single_label else single_label
                    
                    if target_type is None or label_type == target_type:
                        if prefix == 'B':
                            has_begin = True
                            break
            
            if has_begin:
                count += 1
                in_entity = True
            elif not any(l.startswith('I-') for l in expanded if l != 'O'):
                in_entity = False
    
    return count


results = {}
detailed_results = defaultdict(dict)


print("EVALUATION RESULTS")


for frame in frames:
    print(f"# {frame} FRAME")
    
    
    gold_labels = split_into_sentences(gold, frame)
    pred_labels = split_into_sentences(pred, frame)
    
    
    if len(gold_labels) != len(pred_labels):
        print(f"error --> Different number of sentences - Gold: {len(gold_labels)}, Pred: {len(pred_labels)}")
        min_len = min(len(gold_labels), len(pred_labels))
        gold_labels = gold_labels[:min_len]
        pred_labels = pred_labels[:min_len]
    
    
    print(f"--- Overall {frame} Frame Metrics ---")
    
    
    gold_entities = count_entities_by_type(gold_labels)
    pred_entities = count_entities_by_type(pred_labels)
    print(f"Total entities: Gold={gold_entities}, Pred={pred_entities}")
    
    try:
        p = precision_score(gold_labels, pred_labels)
        r = recall_score(gold_labels, pred_labels)
        f1 = f1_score(gold_labels, pred_labels)
        
        results[frame] = {"precision": p, "recall": r, "f1": f1}
        
        print(f"Precision: {p:.4f}")
        print(f"Recall:    {r:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        results[frame] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    
    print(f"\n--- Per-Label Metrics for {frame} ---")
    
    
    gold_label_types = extract_label_types(gold_labels)
    pred_label_types = extract_label_types(pred_labels)
    all_label_types = sorted(set(gold_label_types) | set(pred_label_types))
    
    if not all_label_types:
        print("No labels found in this frame.")
        continue
    
    print(f"Labels found: {', '.join(all_label_types)}\n")
    
    for label_type in all_label_types:
        
        gold_entities_type = count_entities_by_type(gold_labels, label_type)
        pred_entities_type = count_entities_by_type(pred_labels, label_type)
        
        
        gold_filtered = filter_labels_by_type(gold_labels, label_type)
        pred_filtered = filter_labels_by_type(pred_labels, label_type)
        
        try:
            p_label = precision_score(gold_filtered, pred_filtered)
            r_label = recall_score(gold_filtered, pred_filtered)
            f1_label = f1_score(gold_filtered, pred_filtered)
            
            detailed_results[frame][label_type] = {
                "precision": p_label,
                "recall": r_label,
                "f1": f1_label,
                "gold_entities": gold_entities_type,
                "pred_entities": pred_entities_type
            }
            
            print(f"  {label_type}:")
            print(f"    Entities: Gold={gold_entities_type}, Pred={pred_entities_type}")
            print(f"    Precision: {p_label:.4f}")
            print(f"    Recall:    {r_label:.4f}")
            print(f"    F1-Score:  {f1_label:.4f}")
            
        except Exception as e:
            print(f"  {label_type}: Error - {e}")
            detailed_results[frame][label_type] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "gold_entities": gold_entities_type,
                "pred_entities": pred_entities_type
            }


print("SUMMARY - OVERALL FRAME METRICS")
print(f"{'Frame':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
for frame, scores in results.items():
    print(f"{frame:<20} {scores['precision']:<12.4f} {scores['recall']:<12.4f} {scores['f1']:<12.4f}")



avg_precision = sum(s['precision'] for s in results.values()) / len(results)
avg_recall = sum(s['recall'] for s in results.values()) / len(results)
avg_f1 = sum(s['f1'] for s in results.values()) / len(results)
print(f"{'AVERAGE (Macro)':<20} {avg_precision:<12.4f} {avg_recall:<12.4f} {avg_f1:<12.4f}")



print("DETAILED SUMMARY - PER-LABEL METRICS")


for frame in frames:
    if frame in detailed_results and detailed_results[frame]:
        print(f"\n{frame} Frame:")
        print(f"{'Label':<35} {'Gold':<8} {'Pred':<8} {'Prec':<10} {'Rec':<10} {'F1':<10}")
        
        
        for label_type, scores in detailed_results[frame].items():
            print(f"{label_type:<35} {scores['gold_entities']:<8} {scores['pred_entities']:<8} "
                  f"{scores['precision']:<10.4f} {scores['recall']:<10.4f} {scores['f1']:<10.4f}")


results_df = pd.DataFrame(results).T
results_df.to_csv("mistr_evaluation_results_overall.csv")
print("\n\nOverall results saved to 'mistr_evaluation_results_overall.csv'")


detailed_rows = []
for frame, labels in detailed_results.items():
    for label_type, scores in labels.items():
        detailed_rows.append({
            'Frame': frame,
            'Label': label_type,
            'Gold_Entities': scores['gold_entities'],
            'Pred_Entities': scores['pred_entities'],
            'Precision': scores['precision'],
            'Recall': scores['recall'],
            'F1-Score': scores['f1']
        })

detailed_df = pd.DataFrame(detailed_rows)
detailed_df.to_csv("evaluation_results_detailed.csv", index=False)
print("Detailed per-label results saved to 'evaluation_results_detailed.csv'")
