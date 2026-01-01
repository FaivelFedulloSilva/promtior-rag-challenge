import json
import trafilatura

JSONL_PATH = "data/promtior_docs.jsonl"  # tu archivo real

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    line = f.readline()
    data = json.loads(line)

raw_text = data["text"]

print("\n================ RAW TEXT ================\n")
print(raw_text[:2000])  # solo preview

print("\n================ CLEANED TEXT ================\n")

cleaned = trafilatura.extract(
    raw_text,
    include_links=False,
    include_images=False,
    include_tables=False,
    favor_recall=False
)

print(cleaned)

url = "https://www.promtior.ai/post/building-conversational-agents-with-their-own-identity"  # o mejor un /post/...
downloaded = trafilatura.fetch_url(url)
extracted = trafilatura.extract(downloaded, include_links=False, include_images=False)

print(extracted)