import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import xml.etree.ElementTree as ET
import argparse
import random

max_findings_length = 0
max_impression_length = 0

def parse_agrs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--xml_folder', type=str, help='the folder containing the XML files.')
    parser.add_argument('--output_json_path', type=str, default='output.json', help='the path to the output file.')

    args = parser.parse_args()

    return args

def extract_report_from_xml(xml_file, split="train"):
    global max_findings_length, max_impression_length
    tree = ET.parse(xml_file)
    root = tree.getroot()

    indication = ""
    comparison = ""
    findings = ""
    impression = ""
    image_ids = []

    for elem in root.iter():
        if elem.tag == "AbstractText" and "Label" in elem.attrib:
            if elem.attrib["Label"] == "COMPARISON":
                comparison = elem.text.strip() if elem.text else ""
            elif elem.attrib["Label"] == "INDICATION":
                indication = elem.text.strip() if elem.text else ""
            elif elem.attrib["Label"] == "FINDINGS":
                findings = elem.text.strip() if elem.text else ""
            elif elem.attrib["Label"] == "IMPRESSION":
                impression = elem.text.strip() if elem.text else ""
        elif elem.tag == "parentImage" and "id" in elem.attrib:
            image_id = elem.attrib["id"]
            image_ids.append(image_id)

    # if len(image_ids) < 2:
    #     # print(xml_file, image_ids)
    #     return None
    # if len(impression) > 60:
    # if len(findings) > 60:
    #     # print(xml_file)
    #     return None
    
    max_findings_length = max(max_findings_length, len(findings))
    max_impression_length = max(max_impression_length, len(impression)) 


    # report_text = findings if findings else impression
    # report_text = impression if impression else findings
    report_text = comparison + indication + findings + impression
    study_id, png1, png2 = "", "", ""
    if len(image_ids) >= 2:
        study_id = image_ids[0].rsplit("-", 1)[0]
        png1 = image_ids[0].rsplit("-", 1)[1]
        png2 = image_ids[1].rsplit("-", 1)[1]
    return {
        "id": study_id,
        "report": report_text,
        "image_path": [f"{study_id}/{png1}.png", f"{study_id}/{png2}.png"],
        "split": split
    }

def process_all_xml(xml_folder):
    files = os.listdir(xml_folder)
    dataset = {"train": [], "val": [], "test": []}
    random.shuffle(files)  # Randomize the order
    total = len(files)
    train_cutoff = int(0.7 * total)
    val_cutoff = int(0.9 * total)  # 70% + 20% = 90%

    split = "train"
    for i, filename in enumerate(files):
        if i < train_cutoff:
            split = "train"
        elif i < val_cutoff:
            split = "val"
        else:
            split = "test"
        
        if filename.endswith(".xml"):
            file_path = os.path.join(xml_folder, filename)
            result = extract_report_from_xml(file_path, split)
            if result:
                dataset[split].append(result)

    print(f"Max findings length: {max_findings_length}") 
    print(f"Max impression length: {max_impression_length}")
    tot_report = len(dataset['train']) + len(dataset['val']) + len(dataset['test'])

    return dataset, tot_report

if __name__ == "__main__":
    args = parse_agrs()
    result, tot_report = process_all_xml(args.xml_folder)
    with open(args.output_json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {tot_report} reports to {args.output_json_path}")
