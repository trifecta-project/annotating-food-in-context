import os
import re
import xml.dom.minidom
from colorama import init, Fore, Style
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

init(autoreset=True)

load_dotenv("key.env")
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def extract_xml(text):
    match = re.search(r'<xml_annotation>.*?</xml_annotation>', text, re.DOTALL)
    return match.group(0) if match else None

def pretty_print_xml(xml_string):
    tag_colors = {
        'FOOD_Food_LU': Fore.GREEN,
        'FOOD_Descriptor': Fore.BLUE,
        'FOOD_Count': Fore.YELLOW,
        'FOOD_Unit': Fore.MAGENTA,
        'CURE_LU': Fore.LIGHTRED_EX,
        'CURE_Food_Treatment': Fore.BLACK,
        'CURE_Affliction': Fore.LIGHTCYAN_EX, 
        'CURE_Patient': Fore.LIGHTGREEN_EX,
        'CURE_Healer': Fore.CYAN,
        'PR_LU': Fore.RED,
        'PR_Food_Patient': Fore.LIGHTMAGENTA_EX,
        'PR_Medium': Fore.WHITE,
        'PR_Agent': Fore.LIGHTYELLOW_EX,
        'xml_annotation': Fore.LIGHTWHITE_EX
    }
    try:
        dom = xml.dom.minidom.parseString(xml_string)
        pretty_xml = dom.toprettyxml(indent="  ")
        for tag, color in tag_colors.items():
            pretty_xml = re.sub(f'<({tag}[^>]*)>', f'{color}<\\1>{Style.RESET_ALL}', pretty_xml)
            pretty_xml = re.sub(f'</({tag})>', f'{color}</\\1>{Style.RESET_ALL}', pretty_xml)
        return pretty_xml
    except Exception as e:
        print(f"Error formatting XML: {e}")
        return xml_string

def main():
    input_file = input("Insert txt path: ").strip()

    if not os.path.isfile(input_file):
        print(f"file not found: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        print("empty file")
        return

    with open('prompt.txt', "r") as f:
        system_prompt = f.read()

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

    # print("Annotating...")

    response = client.chat.completions.create(
        model="o4-mini",
        messages=messages,
        temperature=0
    )

    annotated_text = response.choices[0].message.content
    xml_part = extract_xml(annotated_text)

    # if xml_part:
    #     print("\n=== Annotated XML ===\n")
    #     print(pretty_print_xml(xml_part))
    # else:
    #     print("Nessuna annotazione XML trovata nella risposta.")

    output_file = "annotation.xml"

    try:
        with open(output_file, "w", encoding="utf-8") as out_f:
            out_f.write(annotated_text)
        print(f"\nOutput saved to: {output_file}")
    except Exception as e:
        print(f"Error saving output to file: {e}")

if __name__ == "__main__":
    main()