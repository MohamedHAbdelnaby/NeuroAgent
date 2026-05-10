
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

API_KEY    = os.getenv("NCBI_API_KEY", "")
BASE_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BATCH_SIZE = 200
MAX_PER_QUERY = 400
DELAY      = 0.35 if not API_KEY else 0.11

DATA_DIR   = Path(__file__).parent
FIELDS     = ["pmid", "title", "year", "journal", "authors", "abstract", "mesh", "doi"]

QUERIES: dict[str, list[str]] = {
    "tumor": [
        '"WHO classification" AND "central nervous system" AND (2016[pdat] OR 2021[pdat]:3000[pdat])',
        'glioblastoma[MeSH] AND (IDH OR EGFR OR MGMT OR TERT) AND prognosis',
        'brain neoplasms[MeSH] AND (grade OR grading) AND MRI[tiab]',
        'glioblastoma[MeSH] AND "ring enhancement" AND MRI',
        '"butterfly glioma"[tiab] OR "corpus callosum" AND glioblastoma',
        'brain tumor[tiab] AND (mass effect OR herniation OR hydrocephalus)',
        '"awake craniotomy"[tiab] AND (language OR motor OR mapping)',
        '"eloquent cortex"[tiab] AND (tumor OR glioma OR resection)',
        'corticospinal tract[tiab] AND (tumor OR glioma) AND DTI',
        'BraTS[tiab] AND brain tumor AND segmentation',
        '"brain tumor segmentation"[tiab] AND (deep learning OR convolutional)',
        'meningioma[MeSH] AND (grading OR resection OR outcome)',
        '"low grade glioma"[tiab] AND (IDH mutation OR treatment OR outcome)',
        '"posterior fossa"[tiab] AND tumor AND (cerebellum OR brainstem)',
        'vasogenic edema[tiab] AND brain tumor AND (dexamethasone OR treatment)',
    ],
    "stroke": [
        '"TOAST classification"[tiab] OR "stroke etiology" AND ischemic',
        'ischemic stroke[MeSH] AND (large artery OR small vessel OR cardioembolic)',
        '"middle cerebral artery"[tiab] AND stroke AND (aphasia OR hemiplegia OR neglect)',
        '"anterior cerebral artery"[tiab] AND stroke AND (weakness OR personality)',
        '"posterior cerebral artery"[tiab] AND stroke AND (visual field OR amnesia)',
        '"basilar artery"[tiab] AND stroke AND ("locked-in" OR coma)',
        '"PICA"[tiab] AND stroke AND Wallenberg',
        '"voxel-based lesion symptom mapping"[tiab] OR VLSM[tiab]',
        '"lesion symptom mapping"[tiab] AND (aphasia OR neglect OR motor)',
        'ATLAS[tiab] AND stroke AND (lesion OR T1 MRI)',
        '"lacunar infarct"[tiab] AND (pure motor OR pure sensory OR internal capsule)',
        '"internal capsule"[tiab] AND stroke AND (hemiplegia OR motor)',
        'stroke[MeSH] AND NIHSS AND functional outcome',
        'stroke[MeSH] AND "modified Rankin" AND prediction',
        '"chronic stroke"[tiab] AND T1 MRI AND (lesion OR atrophy)',
        '"Wallerian degeneration"[tiab] AND stroke AND corticospinal',
    ],
    "adhd": [
        '"frontostriatal"[tiab] AND ADHD AND (dopamine OR connectivity OR volume)',
        'attention deficit hyperactivity disorder[MeSH] AND caudate AND volume',
        '"default mode network"[tiab] AND ADHD AND (suppression OR deactivation OR connectivity)',
        '"ENIGMA"[tiab] AND ADHD AND (subcortical OR volume OR thickness)',
        'ADHD[tiab] AND "cortical thickness" AND (children OR adults)',
        '"corpus callosum"[tiab] AND ADHD AND volume',
        'ADHD[tiab] AND "resting state" AND fMRI AND connectivity',
        '"posterior cingulate"[tiab] AND ADHD AND (DMN OR "default mode")',
        '"right inferior frontal"[tiab] AND ADHD AND (inhibition OR stop-signal)',
        '"stop-signal"[tiab] AND ADHD AND (IFG OR caudate OR inhibition)',
        'ADHD[tiab] AND "response inhibition" AND (brain OR neuroimaging)',
        '"ADHD-200"[tiab] OR "ADHD 200"[tiab] AND neuroimaging',
        'methylphenidate[MeSH] AND ADHD AND (brain activation OR fMRI OR connectivity)',
        'ADHD[tiab] AND (inattentive OR combined) AND neuroimaging AND subtype',
        '"adult ADHD"[tiab] AND neuroimaging AND (frontal OR connectivity)',
    ],
}

def _params(**kwargs) -> dict:
    p = {}
    if API_KEY:
        p["api_key"] = API_KEY
    p.update(kwargs)
    return p

def esearch(query: str, max_records: int) -> tuple[str, str, int]:
    r = requests.get(
        f"{BASE_URL}/esearch.fcgi",
        params=_params(db="pubmed", term=query, usehistory="y", retmax=0, retmode="xml"),
        timeout=30,
    )
    r.raise_for_status()
    root = ET.fromstring(r.text)
    count    = int(root.findtext("Count") or 0)
    webenv   = root.findtext("WebEnv") or ""
    querykey = root.findtext("QueryKey") or ""
    return webenv, querykey, min(count, max_records)

def efetch_batch(webenv: str, querykey: str, retstart: int, retmax: int) -> ET.Element:
    r = requests.get(
        f"{BASE_URL}/efetch.fcgi",
        params=_params(
            db="pubmed", query_key=querykey, WebEnv=webenv,
            retstart=retstart, retmax=retmax, retmode="xml", rettype="abstract",
        ),
        timeout=60,
    )
    r.raise_for_status()
    return ET.fromstring(r.text)

def parse_article(el: ET.Element) -> dict:
    def text(path: str) -> str:
        node = el.find(path)
        return (node.text or "").strip() if node is not None else ""

    pmid  = text(".//PMID")
    title = text(".//ArticleTitle")
    year  = text(".//PubDate/Year") or text(".//PubDate/MedlineDate")[:4]

    parts = el.findall(".//AbstractText")
    if parts:
        chunks = []
        for p in parts:
            label = p.get("Label")
            body  = (p.text or "").strip()
            chunks.append(f"{label}: {body}" if label else body)
        abstract = " ".join(chunks)
    else:
        abstract = ""

    author_els = el.findall(".//Author")
    authors = []
    for a in author_els[:6]:
        last  = (a.findtext("LastName") or "").strip()
        first = (a.findtext("ForeName") or "").strip()
        if last:
            authors.append(f"{last} {first}".strip())
    if len(author_els) > 6:
        authors.append("et al.")

    journal = text(".//Journal/Title") or text(".//MedlineTA")

    mesh_els = el.findall(".//MeshHeading/DescriptorName")
    mesh     = [m.text for m in mesh_els if m.text]

    doi = ""
    for id_el in el.findall(".//ArticleId"):
        if id_el.get("IdType") == "doi":
            doi = (id_el.text or "").strip()
            break

    return {
        "pmid":     pmid,
        "title":    title,
        "year":     year,
        "journal":  journal,
        "authors":  "; ".join(authors),
        "abstract": abstract,
        "mesh":     mesh,
        "doi":      doi,
    }

def download_condition(
    condition: str,
    queries: list[str],
    out_jsonl: Path,
    out_csv: Path,
) -> int:
    seen: set[str] = set()
    total_saved = 0

    with (
        open(out_jsonl, "w", encoding="utf-8") as jf,
        open(out_csv,   "w", encoding="utf-8", newline="") as cf,
    ):
        writer = csv.DictWriter(cf, fieldnames=FIELDS)
        writer.writeheader()

        for q_idx, query in enumerate(queries, start=1):
            print(f"  [{q_idx}/{len(queries)}] {query[:80]}...")
            try:
                webenv, querykey, total = esearch(query, MAX_PER_QUERY)
            except Exception as e:
                print(f"    esearch failed: {e}")
                time.sleep(2)
                continue

            print(f"    {total} records (capped at {MAX_PER_QUERY})")
            new_this_query = 0

            for start in range(0, total, BATCH_SIZE):
                batch_n = min(BATCH_SIZE, total - start)
                try:
                    root = efetch_batch(webenv, querykey, start, batch_n)
                except Exception as e:
                    print(f"    efetch error at offset {start}: {e}")
                    time.sleep(2)
                    continue

                for article_el in root.findall(".//PubmedArticle"):
                    rec = parse_article(article_el)
                    if not rec["pmid"] or rec["pmid"] in seen:
                        continue
                    if not rec["abstract"]:
                        continue
                    seen.add(rec["pmid"])
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    csv_rec = {**rec, "mesh": " | ".join(rec["mesh"])}
                    writer.writerow(csv_rec)
                    new_this_query += 1

                time.sleep(DELAY)

            total_saved += new_this_query
            print(f"    +{new_this_query} new records  (total so far: {total_saved})")
            time.sleep(DELAY)

    return total_saved

def main():
    parser = argparse.ArgumentParser(description="NeuroAgent PubMed Downloader")
    parser.add_argument(
        "--condition", choices=["tumor", "stroke", "adhd", "all"], default="all",
    )
    args = parser.parse_args()

    conditions = (
        list(QUERIES.keys())
        if args.condition == "all"
        else [args.condition]
    )

    if not API_KEY:
        print("Set NCBI_API_KEY env var to increase rate limit from 3 to 10 req/s.")
        print("Get a free key at https://www.ncbi.nlm.nih.gov/account/\n")

    for condition in conditions:
        print(f"\nCondition: {condition.upper()} ({len(QUERIES[condition])} queries)")

        out_jsonl = DATA_DIR / f"{condition}_papers.jsonl"
        out_csv   = DATA_DIR / f"{condition}_papers.csv"

        n = download_condition(condition, QUERIES[condition], out_jsonl, out_csv)
        print(f"\n  Saved {n} unique records to {out_jsonl}")

    print("\nNext step:")
    print("  python data/build_knowledge_base.py --build")

if __name__ == "__main__":
    main()
