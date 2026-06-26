from pathlib import Path
import json
import time
from typing import Dict, Any

import pandas as pd
import requests


RAW_OUTPUT_DIR = Path("data/raw/validation")
PROCESSED_OUTPUT_DIR = Path("data/processed/validation")

RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def validate_cuad_huggingface() -> Dict[str, Any]:
    print("\n[1/5] Validating CUAD Hugging Face dataset...")

    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "theatticusproject/cuad",
            split="train",
            verification_mode="no_checks",
        )

        sample_size = min(20, len(dataset))
        df = dataset.select(range(sample_size)).to_pandas()

        output_path = RAW_OUTPUT_DIR / "cuad_sample.csv"
        df.to_csv(output_path, index=False)

        print(f"SUCCESS: CUAD loaded with {sample_size} sample rows.")
        return {
            "source": "CUAD Hugging Face",
            "status": "success",
            "rows_loaded": sample_size,
            "columns": list(df.columns),
            "output_file": str(output_path),
        }

    except Exception as error:
        print(f"FAILED: CUAD error: {error}")
        return {
            "source": "CUAD Hugging Face",
            "status": "failed",
            "error": str(error),
        }


def validate_usaspending_api() -> Dict[str, Any]:
    print("\n[2/5] Validating USAspending API...")

    url = "https://api.usaspending.gov/api/v2/search/spending_by_award/"

    payload = {
        "filters": {
            "time_period": [
                {
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                }
            ],
            "award_type_codes": ["A", "B", "C", "D"],
        },
        "fields": [
            "Award ID",
            "Recipient Name",
            "Award Amount",
            "Start Date",
            "End Date",
            "Awarding Agency",
            "Awarding Sub Agency",
            "Award Type",
        ],
        "page": 1,
        "limit": 10,
        "sort": "Award Amount",
        "order": "desc",
    }

    try:
        response = requests.post(url, json=payload, timeout=40)
        response.raise_for_status()
        data = response.json()

        save_json(data, RAW_OUTPUT_DIR / "usaspending_awards_sample.json")

        results = data.get("results", [])
        df = pd.DataFrame(results)

        csv_path = RAW_OUTPUT_DIR / "usaspending_awards_sample.csv"
        df.to_csv(csv_path, index=False)

        print(f"SUCCESS: USAspending returned {len(df)} sample rows.")
        return {
            "source": "USAspending API",
            "status": "success",
            "rows_loaded": len(df),
            "columns": list(df.columns),
            "output_file": str(csv_path),
        }

    except Exception as error:
        print(f"FAILED: USAspending error: {error}")
        return {
            "source": "USAspending API",
            "status": "failed",
            "error": str(error),
        }


def validate_gdelt_api() -> Dict[str, Any]:
    print("\n[3/5] Validating GDELT API...")

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": "IBM",
        "mode": "artlist",
        "format": "json",
        "maxrecords": 10,
    }

    try:
        response = requests.get(url, params=params, timeout=40)
        response.raise_for_status()

        try:
            data = response.json()
        except json.JSONDecodeError:
            text_path = RAW_OUTPUT_DIR / "gdelt_non_json_response.txt"
            text_path.write_text(response.text[:2000], encoding="utf-8")

            return {
                "source": "GDELT API",
                "status": "failed_non_json_response",
                "http_status": response.status_code,
                "output_file": str(text_path),
                "error": "GDELT returned non-JSON response. Saved response text for debugging.",
            }

        save_json(data, RAW_OUTPUT_DIR / "gdelt_vendor_news_sample.json")

        articles = data.get("articles", [])
        df = pd.DataFrame(articles)

        csv_path = RAW_OUTPUT_DIR / "gdelt_vendor_news_sample.csv"
        df.to_csv(csv_path, index=False)

        print(f"SUCCESS: GDELT returned {len(df)} news articles.")
        return {
            "source": "GDELT API",
            "status": "success",
            "rows_loaded": len(df),
            "columns": list(df.columns),
            "output_file": str(csv_path),
        }

    except Exception as error:
        print(f"FAILED: GDELT error: {error}")
        return {
            "source": "GDELT API",
            "status": "failed",
            "error": str(error),
        }


def validate_sec_edgar_api() -> Dict[str, Any]:
    print("\n[4/5] Validating SEC EDGAR API...")

    cik = "0000320193"  # Apple Inc.
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"

    headers = {
        "User-Agent": "Portfolio Project contact@example.com",
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }

    try:
        response = requests.get(url, headers=headers, timeout=40)
        response.raise_for_status()
        data = response.json()

        save_json(data, RAW_OUTPUT_DIR / "sec_company_sample_apple.json")

        recent_filings = data.get("filings", {}).get("recent", {})
        recent_df = pd.DataFrame(recent_filings)

        csv_path = RAW_OUTPUT_DIR / "sec_recent_filings_sample.csv"
        recent_df.head(20).to_csv(csv_path, index=False)

        print(f"SUCCESS: SEC returned company data for {data.get('name')}.")
        return {
            "source": "SEC EDGAR API",
            "status": "success",
            "company_name": data.get("name"),
            "cik": data.get("cik"),
            "recent_filings_loaded": min(20, len(recent_df)),
            "output_file": str(csv_path),
        }

    except Exception as error:
        print(f"FAILED: SEC EDGAR error: {error}")
        return {
            "source": "SEC EDGAR API",
            "status": "failed",
            "error": str(error),
        }


def validate_opensanctions_api() -> Dict[str, Any]:
    print("\n[5/5] Validating OpenSanctions API search...")

    url = "https://api.opensanctions.org/search/default"
    params = {
        "q": "IBM",
        "limit": 10,
    }

    try:
        response = requests.get(url, params=params, timeout=40)

        if response.status_code in [401, 403]:
            print("OPTIONAL: OpenSanctions hosted API requires API key. Will use fallback later.")
            return {
                "source": "OpenSanctions API",
                "status": "optional_api_key_required",
                "http_status": response.status_code,
                "message": "Hosted OpenSanctions API requires API key. Compliance module will use fallback/sample data later.",
            }

        response.raise_for_status()
        data = response.json()

        save_json(data, RAW_OUTPUT_DIR / "opensanctions_search_sample.json")

        results = data.get("results", [])
        df = pd.DataFrame(results)

        csv_path = RAW_OUTPUT_DIR / "opensanctions_search_sample.csv"
        df.to_csv(csv_path, index=False)

        print(f"SUCCESS: OpenSanctions returned {len(df)} search results.")
        return {
            "source": "OpenSanctions API",
            "status": "success",
            "rows_loaded": len(df),
            "columns": list(df.columns),
            "output_file": str(csv_path),
        }

    except Exception as error:
        print(f"FAILED: OpenSanctions error: {error}")
        return {
            "source": "OpenSanctions API",
            "status": "failed",
            "error": str(error),
        }


def main() -> None:
    print("Starting real dataset/API validation...")

    validation_results = []

    checks = [
        validate_cuad_huggingface,
        validate_usaspending_api,
        validate_gdelt_api,
        validate_sec_edgar_api,
        validate_opensanctions_api,
    ]

    for check in checks:
        result = check()
        validation_results.append(result)
        time.sleep(1)

    summary_df = pd.DataFrame(validation_results)
    summary_path = PROCESSED_OUTPUT_DIR / "data_source_validation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nValidation completed.")
    print(f"Summary saved to: {summary_path}")
    print("\nValidation Summary:")
    print(summary_df[["source", "status"]])


if __name__ == "__main__":
    main()
