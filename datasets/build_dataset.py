#!/usr/bin/env python3
"""Create clean and synthetic Vietnamese spelling-correction datasets.

This standard-library-only pipeline downloads the public benchmark separately,
extracts a clean corpus from the official Vietnamese Wikipedia dump, excludes
benchmark page ids, injects aligned synthetic errors, and builds vocabularies.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import html
import json
import random
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
RAW, INTERIM, PROCESSED = ROOT / "raw", ROOT / "interim", ROOT / "processed"
WIKI_URL = "https://dumps.wikimedia.org/viwiki/latest/viwiki-latest-pages-articles.xml.bz2"
TEST_URL = "https://raw.githubusercontent.com/heraclex12/Viwiki-spelling/main/spelling_test.json"
PAD, UNK = "<pad>", "<unk>"
TONE = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}
SHAPE = {"\u0302", "\u0306", "\u031b"}
VI_CORE = set("ăâđêôơưĂÂĐÊÔƠƯ")
WORD = re.compile(r"^([^A-Za-zÀ-ỹĐđ]*)([A-Za-zÀ-ỹĐđ]+)([^A-Za-zÀ-ỹĐđ]*)$")
SENTENCE = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÀ-ỴĐ])")
SPACE = re.compile(r"\s+")


def mkdirs() -> None:
    for path in (RAW, INTERIM, PROCESSED):
        path.mkdir(parents=True, exist_ok=True)


def records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{number}") from error


def write(handle: Any, item: dict[str, Any]) -> None:
    handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def download(url: str, output: Path, force: bool) -> None:
    """Download to .part first; resume only when the server supports Range."""
    mkdirs()
    if output.exists() and not force:
        print(f"Already present: {output}")
        return
    part = output.with_suffix(output.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "vi-spell-dataset-builder/1.0"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    print(f"Downloading {url}\n        -> {output}")
    with urllib.request.urlopen(request) as response:
        append = bool(offset and response.status == 206)
        if not append:
            offset = 0
        done, report_at = offset, offset + 64 * 2**20
        with part.open("ab" if append else "wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
                done += len(chunk)
                if done >= report_at:
                    print(f"  downloaded {done / 2**20:.0f} MiB")
                    report_at += 64 * 2**20
    part.replace(output)
    print(f"Saved {output} ({output.stat().st_size / 2**20:.1f} MiB)")


def lname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def child_text(element: ET.Element, wanted: str) -> str:
    for child in element:
        if lname(child.tag) == wanted:
            return child.text or ""
    return ""


def page_revision_text(page: ET.Element) -> str:
    for child in page:
        if lname(child.tag) == "revision":
            return child_text(child, "text")
    return ""


def test_page_ids(path: Path) -> set[str]:
    return {str(item["page_id"]) for item in records(path) if item.get("page_id") is not None}


def prepare_benchmark(args: argparse.Namespace) -> None:
    """Convert public document-level offset labels to token-aligned test chunks."""
    resolved = unresolved = chunks = 0
    with args.output.open("w", encoding="utf-8") as target:
        for document in records(args.input):
            text = document["text"]
            spans = [(match.start(), match.end(), match.group()) for match in re.finditer(r"\S+", text)]
            noisy = [item[2] for item in spans]
            canonical, labels = list(noisy), [0] * len(noisy)
            accepted: list[list[str] | None] = [None] * len(noisy)
            for mistake in document.get("mistakes", []):
                start = int(mistake["start_offset"])
                token_index = next((index for index, (left, right, _) in enumerate(spans) if left <= start < right), None)
                suggestions = [value for value in mistake.get("suggest", []) if value and not any(char.isspace() for char in value)]
                if token_index is None or not suggestions:
                    unresolved += 1
                    continue
                # Keep only labels whose offset points to the annotated typo.
                if not text.startswith(mistake["text"], start):
                    unresolved += 1
                    continue
                labels[token_index] = 1
                accepted[token_index] = suggestions
                canonical[token_index] = suggestions[0]
                resolved += 1
            for first in range(0, len(noisy), args.max_tokens):
                last = min(first + args.max_tokens, len(noisy))
                if not any(labels[first:last]):
                    continue
                write(target, {
                    "id": f"viwiki-{document['_id']}-{first // args.max_tokens}",
                    "noisy_tokens": noisy[first:last],
                    "canonical_tokens": canonical[first:last],
                    "detection_labels": labels[first:last],
                    "accepted_corrections": accepted[first:last],
                    "source": "viwiki_spelling_test",
                    "page_id": document.get("page_id"),
                    "title": document.get("_id"),
                })
                chunks += 1
    print(f"Prepared {chunks} test chunks; resolved {resolved} mistakes; unresolved {unresolved}")


def clean_markup(text: str) -> str:
    """Conservative cleanup; remaining markup is rejected by quality filters."""
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"<ref\b[^>/]*?/\s*>", " ", text, flags=re.I)
    text = re.sub(r"<ref\b[^>]*>.*?</ref\s*>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.S)
    for _ in range(8):
        changed = re.sub(r"\{\{[^{}]*\}\}", " ", text, flags=re.S)
        if changed == text:
            break
        text = changed
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\]", " ", text)
    text = re.sub(r"(?m)^\s*=+.*?=+\s*$", " ", text)
    text = re.sub(r"(?m)^\s*[*#:;]+", " ", text)
    text = re.sub(r"\[\d+\]", " ", text)
    return SPACE.sub(" ", html.unescape(text.replace("'''", "").replace("''", ""))).strip()


def acceptable(sentence: str, min_tokens: int, max_tokens: int) -> bool:
    tokens = sentence.split()
    if not min_tokens <= len(tokens) <= max_tokens or not 24 <= len(sentence) <= 900:
        return False
    if any(markup in sentence for markup in ("{{", "}}", "[[", "]]")):
        return False
    letters = sum(char.isalpha() for char in sentence)
    if letters / max(1, len(sentence)) < 0.52:
        return False
    nfd = unicodedata.normalize("NFD", sentence)
    if not (any(char in VI_CORE for char in sentence) or any(char in TONE | SHAPE for char in nfd)):
        return False
    symbols = sum(char in "{}[]|<>_=*/\\" for char in sentence)
    return symbols / len(sentence) < 0.015


def extract_wikipedia(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise FileNotFoundError(f"Missing dump: {args.input}; run download-wikipedia first.")
    excluded = test_page_ids(args.exclude_test) if args.exclude_test.exists() else set()
    seen: set[str] = set()
    written = pages = skipped = written_bytes = 0
    max_output_bytes = int(args.max_output_gib * 2**30) if args.max_output_gib else None
    print(f"Extracting clean sentences from {args.input}")
    with bz2.open(args.input, "rb") as source, args.output.open("w", encoding="utf-8") as target:
        for _, page in ET.iterparse(source, events=("end",)):
            if lname(page.tag) != "page":
                continue
            page_id, namespace, title = child_text(page, "id"), child_text(page, "ns"), child_text(page, "title")
            redirect = any(lname(child.tag) == "redirect" for child in page)
            if namespace != "0" or redirect:
                page.clear()
                continue
            if page_id in excluded:
                skipped += 1
                page.clear()
                continue
            pages += 1
            for candidate in SENTENCE.split(clean_markup(page_revision_text(page))):
                sentence = unicodedata.normalize("NFC", SPACE.sub(" ", candidate).strip())
                if not acceptable(sentence, args.min_tokens, args.max_tokens):
                    continue
                digest = hashlib.sha256(sentence.casefold().encode()).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                record = {"text": sentence, "source": "viwiki", "page_id": page_id, "title": title}
                write(target, record)
                written += 1
                written_bytes += len(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 1
                if (args.max_sentences and written >= args.max_sentences) or (max_output_bytes and written_bytes >= max_output_bytes):
                    print(f"Reached {written:,} sentences / {written_bytes / 2**30:.2f} GiB; excluded {skipped} benchmark pages")
                    return
            page.clear()
            if pages % 20_000 == 0:
                print(f"  pages={pages:,}; clean sentences={written:,}")
    print(f"Finished {written:,} sentences from {pages:,} pages; excluded {skipped} benchmark pages")


def api_query(parameters: dict[str, str]) -> dict[str, Any]:
    query = {**parameters, "action": "query", "format": "json", "formatversion": "2"}
    url = "https://vi.wikipedia.org/w/api.php?" + urllib.parse.urlencode(query)
    request = urllib.request.Request(url, headers={"User-Agent": "vi-spell-dataset-builder/1.0"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request) as response:
                payload = json.load(response)
            time.sleep(0.25)  # polite rate limit, including between paired queries
            return payload
        except urllib.error.HTTPError as error:
            if error.code not in {429, 503} or attempt == 5:
                raise
            wait = 2 ** attempt
            print(f"  API returned {error.code}; retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def api_revision_content(page: dict[str, Any]) -> str:
    revision = (page.get("revisions") or [{}])[0]
    return revision.get("slots", {}).get("main", {}).get("content", revision.get("*", ""))


def collect_wikipedia_api(args: argparse.Namespace) -> None:
    """Create a small clean seed corpus without waiting for the 1.14 GB dump.

    This uses the official MediaWiki API in modest batches. It is intentionally
    a bootstrap path; the XML dump remains the reproducible full-corpus path.
    """
    excluded = test_page_ids(args.exclude_test) if args.exclude_test.exists() else set()
    seen_pages: set[str] = set()
    accepted_pages = written = skipped = 0
    with args.output.open("w", encoding="utf-8") as target:
        while accepted_pages < args.pages and (not args.max_sentences or written < args.max_sentences):
            random_pages = api_query({"generator": "random", "grnnamespace": "0", "grnlimit": str(args.batch_size)})
            ids = [str(page["pageid"]) for page in random_pages.get("query", {}).get("pages", [])]
            if not ids:
                continue
            details = api_query({"pageids": "|".join(ids), "prop": "revisions", "rvprop": "content", "rvslots": "main"})
            for page in details.get("query", {}).get("pages", []):
                page_id = str(page.get("pageid", ""))
                if page_id in seen_pages or page_id in excluded:
                    skipped += 1
                    continue
                seen_pages.add(page_id)
                raw = api_revision_content(page)
                if not raw:
                    continue
                accepted_pages += 1
                for candidate in SENTENCE.split(clean_markup(raw)):
                    sentence = unicodedata.normalize("NFC", SPACE.sub(" ", candidate).strip())
                    if acceptable(sentence, args.min_tokens, args.max_tokens):
                        write(target, {"text": sentence, "source": "viwiki_api", "page_id": page_id, "title": page.get("title", "")})
                        written += 1
                        if args.max_sentences and written >= args.max_sentences:
                            break
                if accepted_pages >= args.pages or (args.max_sentences and written >= args.max_sentences):
                    break
            if accepted_pages % 200 < args.batch_size:
                print(f"  pages={accepted_pages:,}; clean sentences={written:,}")
            time.sleep(args.delay_seconds)
    print(f"Collected {written:,} sentences from {accepted_pages:,} API pages; skipped {skipped} pages")


def parts(token: str) -> tuple[str, str, str] | None:
    found = WORD.match(token)
    if not found or len(found.group(2)) < 2:
        return None
    return found.groups()


def undiacritic(text: str, all_marks: bool) -> str:
    out: list[str] = []
    for char in text:
        if char in "đĐ":
            out.append("d" if char == "đ" else "D")
            continue
        decomposed = unicodedata.normalize("NFD", char)
        base, marks = decomposed[0], decomposed[1:]
        # Full diacritic removal must remove every combining mark, including
        # marks from imported names such as Turkish capital İ.
        kept = [] if all_marks else [mark for mark in marks if mark not in TONE]
        out.append(unicodedata.normalize("NFC", base + "".join(kept)))
    return "".join(out)


def telex(text: str, vni: bool = False) -> str:
    shapes = {"\u0306": "8" if vni else "w", "\u0302": "6" if vni else None, "\u031b": "7" if vni else "w"}
    tones = {"\u0301": "1" if vni else "s", "\u0300": "2" if vni else "f", "\u0309": "3" if vni else "r", "\u0303": "4" if vni else "x", "\u0323": "5" if vni else "j"}
    out: list[str] = []
    tone_keys: list[str] = []
    for char in text:
        if char in "đĐ":
            out.append(("D9" if vni else "DD") if char == "Đ" else ("d9" if vni else "dd"))
            continue
        decomposed = unicodedata.normalize("NFD", char)
        base, marks = decomposed[0], decomposed[1:]
        shaped = base
        for mark in marks:
            if mark == "\u0302":
                shaped = base + ("6" if vni else base.lower())
            elif mark in shapes:
                shaped = base + str(shapes[mark])
        out.append(shaped)
        tone_keys.extend(tones[mark] for mark in marks if mark in tones)
    # Telex/VNI tone keys are normally entered after the syllable, e.g.
    # bão -> baox and bộ -> booj / bo65, rather than baxo / bo6j.
    return "".join(out + tone_keys)


NEIGHBORS = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wrsd", "f": "drtgvc", "g": "ftyhbv",
    "h": "gyujnb", "i": "uokl", "j": "huikmn", "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm",
    "o": "iklp", "p": "ol", "q": "wa", "r": "etfd", "s": "awedxz", "t": "ryfg", "u": "yihj",
    "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tugh", "z": "asx",
}
CONFUSIONS = (("tr", "ch"), ("ch", "tr"), ("s", "x"), ("x", "s"), ("d", "gi"), ("gi", "d"), ("r", "gi"), ("gi", "r"), ("n", "l"), ("l", "n"))
OPERATORS = ("telex", "vni", "drop_tone", "drop_diacritic", "keyboard", "delete", "transpose", "consonant")
WEIGHTS = (0.22, 0.10, 0.16, 0.12, 0.14, 0.10, 0.08, 0.08)


def corrupt_core(core: str, operator: str, rng: random.Random) -> str:
    if operator == "telex":
        return telex(core)
    if operator == "vni":
        return telex(core, vni=True)
    if operator == "drop_tone":
        return undiacritic(core, all_marks=False)
    if operator == "drop_diacritic":
        return undiacritic(core, all_marks=True)
    if operator == "consonant":
        lower = core.casefold()
        choices = [(old, new) for old, new in CONFUSIONS if lower.startswith(old)]
        if not choices:
            return core
        old, new = rng.choice(choices)
        replacement = new.upper() if core[:len(old)].isupper() else (new.capitalize() if core[:1].isupper() else new)
        return replacement + core[len(old):]
    plain = undiacritic(core, all_marks=True)
    if len(plain) < 2:
        return core
    if operator == "delete":
        index = rng.randrange(len(plain))
        return plain[:index] + plain[index + 1:]
    if operator == "transpose":
        index = rng.randrange(len(plain) - 1)
        chars = list(plain)
        chars[index], chars[index + 1] = chars[index + 1], chars[index]
        return "".join(chars)
    if operator == "keyboard":
        choices = [index for index, char in enumerate(plain.lower()) if char in NEIGHBORS]
        if not choices:
            return core
        index = rng.choice(choices)
        replacement = rng.choice(NEIGHBORS[plain[index].lower()])
        return plain[:index] + (replacement.upper() if plain[index].isupper() else replacement) + plain[index + 1:]
    raise ValueError(operator)


def corrupt_token(token: str, rng: random.Random) -> tuple[str, str] | None:
    split = parts(token)
    if split is None:
        return None
    prefix, core, suffix = split
    chosen = rng.choices(OPERATORS, weights=WEIGHTS, k=1)[0]
    fallback = [operator for operator in OPERATORS if operator != chosen]
    rng.shuffle(fallback)
    order = [chosen] + fallback
    for operator in order:
        corrupted = prefix + corrupt_core(core, operator, rng) + suffix
        if corrupted != token and corrupted.strip():
            return corrupted, operator
    return None


def noisy_sentence(text: str, rng: random.Random, rate: float, max_errors: int, clean_rate: float) -> tuple[list[str], list[int], list[str]]:
    target = text.split()
    noisy, labels, kinds = list(target), [0] * len(target), ["none"] * len(target)
    if rng.random() < clean_rate:
        return noisy, labels, kinds
    indexes = list(range(len(target)))
    rng.shuffle(indexes)
    errors = 0
    for index in indexes:
        if errors >= max_errors or rng.random() >= rate:
            continue
        result = corrupt_token(target[index], rng)
        if result:
            noisy[index], kinds[index] = result
            labels[index], errors = 1, errors + 1
    if errors == 0:
        for index in indexes:
            result = corrupt_token(target[index], rng)
            if result:
                noisy[index], kinds[index], labels[index] = result[0], result[1], 1
                break
    return noisy, labels, kinds


def validation_page(page_id: str, fraction: float) -> bool:
    digest = int(hashlib.sha256(page_id.encode()).hexdigest()[:12], 16)
    return digest / 16**12 < fraction


def generate(args: argparse.Namespace) -> None:
    rng, train_count, valid_count = random.Random(args.seed), 0, 0
    with args.train_output.open("w", encoding="utf-8") as train, args.validation_output.open("w", encoding="utf-8") as valid:
        for source in records(args.input):
            text = source["text"]
            page_id = str(source.get("page_id") or source.get("source_id") or text)
            destination = valid if validation_page(page_id, args.validation_fraction) else train
            for variant in range(args.variants_per_sentence):
                noisy, labels, kinds = noisy_sentence(text, rng, args.token_error_rate, args.max_errors, args.clean_sentence_rate)
                target = text.split()
                assert len(noisy) == len(target) == len(labels) == len(kinds)
                item = {
                    "id": hashlib.sha256(f"{page_id}|{text}|{variant}".encode()).hexdigest()[:20],
                    "noisy_text": " ".join(noisy), "clean_text": text,
                    "noisy_tokens": noisy, "clean_tokens": target,
                    "detection_labels": labels, "error_types": kinds,
                    "source": source.get("source", "unknown"), "page_id": source.get("page_id"), "title": source.get("title"),
                }
                write(destination, item)
                if destination is train:
                    train_count += 1
                else:
                    valid_count += 1
                if args.max_examples and train_count + valid_count >= args.max_examples:
                    print(f"Generated train={train_count:,}; validation={valid_count:,}")
                    return
    print(f"Generated train={train_count:,}; validation={valid_count:,}")


def build_vocab(args: argparse.Namespace) -> None:
    words: Counter[str] = Counter()
    chars: Counter[str] = Counter()
    for item in records(args.input):
        for token in item["clean_tokens"]:
            words[token] += 1
            chars.update(token)
    word_tokens = [PAD, UNK] + [token for token, _ in words.most_common(args.size - 2) if token not in {PAD, UNK}]
    char_tokens = [PAD, UNK] + [token for token, _ in chars.most_common(args.char_size - 2) if token not in {PAD, UNK}]
    # Keep ids compatible with Compact-1M even for a small bootstrap corpus.
    word_tokens.extend(f"<unused_word_{index}>" for index in range(args.size - len(word_tokens)))
    char_tokens.extend(f"<unused_char_{index}>" for index in range(args.char_size - len(char_tokens)))
    word_vocab = {token: index for index, token in enumerate(word_tokens[:args.size])}
    char_vocab = {token: index for index, token in enumerate(char_tokens[:args.char_size])}
    args.word_output.write_text(json.dumps(word_vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    args.char_output.write_text(json.dumps(char_vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(word_vocab):,} word entries and {len(char_vocab):,} character entries")


def verify(args: argparse.Namespace) -> None:
    examples = errors = 0
    kinds: Counter[str] = Counter()
    for path in args.paths:
        for item in records(path):
            noisy, target, labels, types = item["noisy_tokens"], item["clean_tokens"], item["detection_labels"], item["error_types"]
            if not len(noisy) == len(target) == len(labels) == len(types):
                raise ValueError(f"Token misalignment: {path}:{item.get('id')}")
            for noisy_token, target_token, label, kind in zip(noisy, target, labels, types):
                if int(noisy_token != target_token) != label:
                    raise ValueError(f"Incorrect label: {path}:{item.get('id')}")
                errors += label
                if label:
                    kinds[kind] += 1
            examples += 1
    print(f"Verified {examples:,} examples, {errors:,} synthetic errors")
    print(dict(kinds.most_common()))


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download-test"); p.add_argument("--output", type=Path, default=RAW / "viwiki_spelling_test.json"); p.add_argument("--force", action="store_true")
    p = sub.add_parser("prepare-benchmark", help="Convert public offset labels into token-aligned test data"); p.add_argument("--input", type=Path, default=RAW / "viwiki_spelling_test.json"); p.add_argument("--output", type=Path, default=PROCESSED / "viwiki_test.jsonl"); p.add_argument("--max-tokens", type=int, default=192)
    p = sub.add_parser("download-wikipedia"); p.add_argument("--output", type=Path, default=RAW / "viwiki-latest-pages-articles.xml.bz2"); p.add_argument("--force", action="store_true")
    p = sub.add_parser("extract-wikipedia"); p.add_argument("--input", type=Path, default=RAW / "viwiki-latest-pages-articles.xml.bz2"); p.add_argument("--output", type=Path, default=INTERIM / "clean_viwiki.jsonl"); p.add_argument("--exclude-test", type=Path, default=RAW / "viwiki_spelling_test.json"); p.add_argument("--max-sentences", type=int); p.add_argument("--max-output-gib", type=float); p.add_argument("--min-tokens", type=int, default=5); p.add_argument("--max-tokens", type=int, default=192)
    p = sub.add_parser("collect-wikipedia-api", help="Build a small seed corpus from the official API"); p.add_argument("--output", type=Path, default=INTERIM / "clean_viwiki_api.jsonl"); p.add_argument("--exclude-test", type=Path, default=RAW / "viwiki_spelling_test.json"); p.add_argument("--pages", type=int, default=1000); p.add_argument("--batch-size", type=int, default=20); p.add_argument("--max-sentences", type=int); p.add_argument("--min-tokens", type=int, default=5); p.add_argument("--max-tokens", type=int, default=192); p.add_argument("--delay-seconds", type=float, default=.5)
    p = sub.add_parser("generate"); p.add_argument("--input", type=Path, default=INTERIM / "clean_viwiki.jsonl"); p.add_argument("--train-output", type=Path, default=PROCESSED / "train.jsonl"); p.add_argument("--validation-output", type=Path, default=PROCESSED / "validation.jsonl"); p.add_argument("--seed", type=int, default=20260808); p.add_argument("--validation-fraction", type=float, default=.02); p.add_argument("--token-error-rate", type=float, default=.13); p.add_argument("--clean-sentence-rate", type=float, default=.15); p.add_argument("--max-errors", type=int, default=4); p.add_argument("--variants-per-sentence", type=int, default=2); p.add_argument("--max-examples", type=int)
    p = sub.add_parser("build-vocab"); p.add_argument("--input", type=Path, default=PROCESSED / "train.jsonl"); p.add_argument("--word-output", type=Path, default=PROCESSED / "word_vocab.json"); p.add_argument("--char-output", type=Path, default=PROCESSED / "char_vocab.json"); p.add_argument("--size", type=int, default=9500); p.add_argument("--char-size", type=int, default=400)
    p = sub.add_parser("verify"); p.add_argument("paths", nargs="*", type=Path, default=[PROCESSED / "train.jsonl", PROCESSED / "validation.jsonl"])
    return parser.parse_args()


def main() -> None:
    args = cli(); mkdirs()
    if args.command == "download-test": download(TEST_URL, args.output, args.force)
    elif args.command == "prepare-benchmark": prepare_benchmark(args)
    elif args.command == "download-wikipedia": download(WIKI_URL, args.output, args.force)
    elif args.command == "extract-wikipedia": extract_wikipedia(args)
    elif args.command == "collect-wikipedia-api": collect_wikipedia_api(args)
    elif args.command == "generate": generate(args)
    elif args.command == "build-vocab": build_vocab(args)
    else: verify(args)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, urllib.error.URLError) as exception:
        print(f"ERROR: {exception}", file=sys.stderr)
        raise SystemExit(2)
