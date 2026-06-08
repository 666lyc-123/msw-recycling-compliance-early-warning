from __future__ import annotations

import csv
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path


BASE = Path(r"D:\VC code\MDPI2")
REF_CSV = BASE / "outputs" / "tables" / "table21_30_core_references_for_waste_management.csv"
VERIFIED_CSV = BASE / "outputs" / "tables" / "table21_30_verified_crossref_references.csv"


def ascii_clean(text: str) -> str:
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2010", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text).strip()


def get_year(item: dict) -> str:
    for key in ["published-print", "published-online", "issued"]:
        parts = item.get(key, {}).get("date-parts")
        if parts and parts[0]:
            return str(parts[0][0])
    return ""


def format_author(author: dict) -> str:
    family = ascii_clean(author.get("family", ""))
    given = ascii_clean(author.get("given", ""))
    initials = ""
    for part in re.split(r"[\s.-]+", given):
        if part:
            initials += part[0].upper() + "."
    return f"{family}, {initials}" if initials else family


def format_authors(authors: list[dict], max_authors: int = 6) -> str:
    if not authors:
        return ""
    formatted = [format_author(a) for a in authors]
    if len(formatted) > max_authors:
        return ", ".join(formatted[:max_authors]) + ", et al."
    return ", ".join(formatted)


def fetch_crossref(doi: str) -> dict:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (reference verification; mailto:lanyoucheng@mail.sdu.edu.cn)"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)["message"]


def format_reference(item: dict, doi: str) -> str:
    authors = format_authors(item.get("author", []))
    year = get_year(item)
    title = ascii_clean(item.get("title", [""])[0]).rstrip(".")
    journal = ascii_clean(item.get("container-title", [""])[0])
    volume = ascii_clean(str(item.get("volume", "")))
    issue = ascii_clean(str(item.get("issue", "")))
    page = ascii_clean(str(item.get("page", "")))
    article_number = ascii_clean(str(item.get("article-number", "")))

    vol_issue = volume
    if issue:
        vol_issue = f"{volume}({issue})" if volume else f"({issue})"

    locator = page or article_number
    tail = journal
    if vol_issue:
        tail += f" {vol_issue}"
    if locator:
        tail += f", {locator}"
    title_end = "" if title.endswith(("?", "!", ".")) else "."
    return f"{authors}, {year}. {title}{title_end} {tail}. https://doi.org/{doi}"


def extract_doi(reference: str) -> str:
    match = re.search(r"https://doi\.org/(\S+)", reference)
    if not match:
        raise ValueError(f"No DOI found in reference: {reference}")
    return match.group(1).rstrip(".")


def main() -> None:
    rows = list(csv.DictReader(REF_CSV.open("r", encoding="utf-8-sig")))
    out_rows = []
    for row in rows:
        doi = extract_doi(row["Reference"])
        item = fetch_crossref(doi)
        formatted = format_reference(item, doi)
        out_rows.append(
            {
                "No.": row["No."],
                "Reference": formatted,
                "Purpose in revised introduction": row.get("Purpose in revised introduction", ""),
                "A-journal rationale": row.get("A-journal rationale", ""),
                "DOI": doi,
                "Crossref title": ascii_clean(item.get("title", [""])[0]),
                "Crossref journal": ascii_clean(item.get("container-title", [""])[0]),
                "Crossref year": get_year(item),
                "Verification source": "Crossref DOI metadata",
            }
        )
        print(row["No."], doi, "OK")
        time.sleep(0.1)

    headers = [
        "No.",
        "Reference",
        "Purpose in revised introduction",
        "A-journal rationale",
        "DOI",
        "Crossref title",
        "Crossref journal",
        "Crossref year",
        "Verification source",
    ]
    for path in [REF_CSV, VERIFIED_CSV]:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(out_rows)
        print(path)


if __name__ == "__main__":
    main()
