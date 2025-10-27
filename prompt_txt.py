import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

PROMPT_TEMPLATE = """You are an expert in Frame Semantics specializing in historical food annotation. 
Annotate food-related text using BIO format (Begin–Inside–Outside) at the token level.

## ANNOTATION PROCESS
1. FIRST STEP: Identify and annotate ALL Food_LU (Lexical Units of Food) - words referring to food entities (e.g., apple, soup, meat, tea, water etc.). Annotate just the word referring to food without articles unless necessary to define the entity.
2. SECOND STEP: Once Food_LU is identified, annotate the frames activated by the LU through semantic relations.

## FRAMES AND SEMANTIC RELATIONS

### RELATION: IS (Formal)
Exemplar frame fof food
**Frame: FOOD**
- B/I-FOOD_LU: Core element - every word referring to food or drink items (starting point for all relations)
- B/I-FOOD_Constituent_Part: parts that constitute the food item (peel, rind, skin, etc.)
- B-FOOD_Descriptor: characteristics or descriptions of food
  - B-DESCR_Positive: positive evaluation of adjectives related to food
  - B-DESCR_Negative: negative evaluation of adjectives related to food
  - B-DESCR_Neutral: neutral evaluation of adjectives related to food
- B/I-FOOD_Count: number of units of the measured food (ONLY WHEN THE COUNT REFERS TO FOOD)
- B/I-FOOD_Unit: standardized quantity units (grams, liters, etc.) of FOOD 

### RELATION: COMPOSED BY (Constitutive)
Frames expressing the description of any component of food explicitly mentioned in the sentence
**Frame: INGREDIENTS**
Activated when food material is used to create a food product. Annotate only when clearly distinguished from Food frame, otherwise Food has precedence.
- B/I-INGR_Material_LU: entity used to make a food product (MUST also be annotated as FOOD_LU)
- B/I-INGR_Food_Product: entity created from the material (MUST also be annotated as FOOD_LU)
see for the annotation, the examples reported below.

### RELATION: CREATED BY/CREATED WITH (Agentive)
Frames expressing the coming into being of the food

**Frame: COOKING_CREATION**
Describes food and meal preparation, EXCLUDING the processes of preservation which are annotated with PR. We consider as LU any type of food manipulation related to its preparation (e.g., peel, grate, cook, bake etc.). Do not annotate generic verbs such as make or prepare, we are interested in the practices operated on food. 
- B/I-COOKING_CREATION_LU: cooking or food manipulation verbs (bake, cook, fry, grill, roast, concoct, whip up, etc.)
- B/I-COOKING_CREATION_Cook: person preparing food
- B/I-COOKING_CREATION_Produced_Food: result of cooking efforts or just the food which is manipulated (MUST also be annotated as FOOD_LU)

**Frame: PRESERVING**
Describe any processes or practices operated on food to prevent its decay.
- B/I-PR_LU: preservation verbs/nouns (preserve, dry, pickle, canning, salt, smoke, etc.)
- B/I-PR_Medium: substance used to preserve (MUST also be annotated as FOOD_LU if it's food)
- B/I-PR_Food_Patient: food being preserved (MUST also be annotated as FOOD_LU)

### RELATION: USED AS/USED FOR/MEANT TO (Telic)
Any frames expressing the purpose or use of food 

**Frame: CURE**
Healer treats/cures an Affliction using food as Treatment. Purpose of food is to be used as medical treatment
- B/I-CURE_LU: treatment/cure verbs (cure, heal, treat, soothe, alleviate, remedy, ease, palliate, improve, rehabilitate, subdue, therapeutic, etc.) with which the food is used as treatment
- B/I-CURE_Affliction: injury, disease, condition, or pain which is treated/cured with the food
- B/I-CURE_Healer: person treating or curing a patient with food
- B/I-CURE_Food_Treatment: food substance used as treatment (MUST also be annotated as FOOD_LU)
- B/I-CURE_Patient: person suffering affliction cured/treated by the food
- B/I-CURE_Body_Part: specific area of patient's body treated with food

**Frame: INGESTION**
Purpose of food is being consumed/ingested.
- B/I-INGESTION_LU: eating/drinking verbs (eat, drink, consume, devour, dine, feast, feed, gobble, gulp, slurp, sip, breakfast, lunch, imbibe, ingest, munch, nibble, quaff, etc.)
- B/I-INGESTION_Food_Ingestible: entities being consumed (MUST also be annotated as FOOD_LU)
- B/I-INGESTION_Ingestor: person eating/drinking

## CRITICAL RULES FOR ANNOTATION 
It is important to notice that the annotationis first of all linguistics, so you are annotating the way food is described in written texts. It means that you need to stick to what is expressed in the text, following these steps:
1. **MANDATORY**: Always identify and annotate FOOD_LU FIRST - no other annotation can be done without it
2. **DOUBLE ANNOTATION**: When food activates multiple frames, it MUST be annotated with FOOD_LU AND the specific frame element (e.g., chicken<B-FOOD_LU,B-CURE_Food_Treatment> in a sentence such as "chicken cures flu")
3. **Frame precedence**: If distinction between Food and Ingredients is unclear, Food annotation takes precedence
4. **Using LUs**: Annotate verbs like "use", "employ", "apply" as frame-specific LUs (e.g., CURE_LU) only when context clearly indicates the purpose (e.g., "use ginger for sore throats" → use as CURE_LU) and no relevent explicit LUs are present in the sentence.
5. **Clear frame expression without specific LU**: sometimes you have cases in which the example clearly express a particular use of food but no specific LU is found. Only when it is very clear you can use another frame element as LU (e.g., beef tea is excellent for convalescents, here convalescent is both CURE_Patient and CURE_LU since beef tea is clearly used as a treatment)
6. **Non-frame tokens**: Tag as O
7. **No additions** limit your task to the annotation of the labels included above. Do NOT change the sentences of the input, and do NOT add new labels.
7. **Multiple frames**: Separate with comma in same token
8. **Output format**: Single line, token<tag> format, space-separated

## EXAMPLES

**Input:** There are five rotted bananas on the table, each weighting around 100 gr.
**Output:** There<O> are<O> five<B-FOOD_Count> rotted<B-FOOD_Descriptor,B-DESCR_Negative> bananas<B-FOOD_LU> on<O> the<O> table<O> ,<O> each<O> weighting<O> around<O> 100<B-FOOD_Count> gr<B-FOOD_Unit> .<O>

**Input:** Cato suggests that slurping some chicken soup in the morning helps digestion and strengthens the stomach.
**Output:** Cato<O> suggests<O> that<O> slurping<B-INGESTION_LU> some<O> chicken<B-FOOD_LU,B-INGR_Material_LU,B-INGR_Food_Product,B-INGESTION_Food_Ingestible> soup<B-FOOD_LU,I-INGR_Food_Product,I-INGESTION_Food_Ingestible> in<O> the<O> morning<O> helps<B-CURE_LU> digestion<B-CURE_Affliction> and<O> strengthens<B-CURE_LU> the<B-CURE_Body_Part> stomach<I-CURE_Body_Part> .<O>

**Input:** In the past, they submerged meat in vinegar to preserve it.
**Output:** In<O> the<O> past<O> ,<O> they<B-PR_Agent> submerged<O> meat<B-FOOD_LU,B-PR_Food_Patient> in<O> vinegar<B-FOOD_LU,B-PR_Medium> to<O> preserve<B-PR_LU> it<O> .<O>

**Input:** Doctors use ginger to soothe sore throats.
**Output:** Doctors<B-CURE_Healer> use<O> ginger<B-FOOD_LU,B-CURE_Food_Treatment> to<O> soothe<B-CURE_LU> sore<B-CURE_Affliction> throats<I-CURE_Affliction> .<O>

**Input:** I love white chocolate and fresh strawberries.
**Output:** I<O> love<O> white<B-FOOD_Descriptor,B-DESCR_Neutral> chocolate<B-FOOD_LU> and<O> fresh<B-FOOD_Descriptor,B-DESCR_Positive> strawberries<B-FOOD_LU> .<O>

**Input:** Peel the onions finely.
**Output:** Peel<B-COOKING_CREATION_LU> the<O> onions<B-FOOD_LU,B-COOKING_CREATION_Produced_Food> finely<O> .<O>

**Input:** Orange cakes are delicious when baked.
**Output:** Orange<B-FOOD_LU,B-INGR_Material_LU,B-INGR_Food_Product,B-COOKING_CREATION_Produced_Food> cakes<B-FOOD_LU,I-INGR_Food_Product,I-COOKING_CREATION_Produced_Food> are<O> delicious<B-FOOD_Descriptor,B-DESCR_Positive> when<O> baked<B-COOKING_CREATION_LU> .<O>

**Input:** Tea is sometimes used for headache.
**Output:** Tea<B-FOOD_LU,B-CURE_Food_Treatment> is<O> sometimes<O> used<B-CURE_LU> for<O> headache<B-CURE_Affliction>.<O>

**Input:** Child loves to bike.
**Output:** Child<O> loves<O> to<O> bike<O>.<O>

Now annotate: {sentence}
Output:"""

def annotate_sentence(sentence):
    messages = [
        {"role": "user", "content": PROMPT_TEMPLATE.format(sentence=sentence)}
    ]
    
    stream = client.chat.completions.create(
        # model="meta-llama/Llama-3.2-3B-Instruct:novita",
        # model="meta-llama/Llama-3.2-3B-Instruct:together",
        # model="mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
        # model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B:nscale",
        # model="Qwen/Qwen2.5-7B-Instruct:together",
        model="deepseek-ai/DeepSeek-V3:together",
         # model="meta-llama/Llama-3.1-70B-Instruct:fireworks-ai",
        messages=messages,
        temperature=0.0,
        # top_p=0.7,
        stream=True,
    )
    
    result = ""
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                
                result += delta.content
    
    return result.strip()

input_file = "rawtext.txt"
output_file = "annotated_text.txt"

with open(input_file, "r", encoding="utf-8") as f:
    sentences = f.readlines()

with open(output_file, "w", encoding="utf-8") as f:
    for i, sentence in enumerate(sentences, 1):
        sentence = sentence.strip()
        if not sentence: 
            continue
        
        print(f"Annotating {i}/{len(sentences)}: {sentence[:50]}...")
        annotation = annotate_sentence(sentence)
        
        f.write(f"Input: {sentence}\n")
        f.write(f"Output: {annotation}\n")
        f.write("-" * 80 + "\n")
        
        print(f"Output: {annotation}\n")

print(f"\nAnnotation complete! Results saved to {output_file}")
