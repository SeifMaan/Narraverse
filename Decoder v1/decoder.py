"""
SAGA Decoder — Full 4-Stage Sequel Generation Pipeline
=======================================================
Stage 1 : Context Compilation   (no LLM)
Stage 2 : Blueprint Generation  (1 LLM call)
Stage 3 : Chapter Outlines      (1 LLM call per chapter, batched)
Stage 4 : Scene Prose           (1 LLM call per scene, batched)
+ Consistency Pass              (rule-based, no LLM)

LLM: Mistral API  (minimised calls, rate-limit safe)
"""

import json
import time
import re
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import requests


# ===========================================================================
# CONFIG
# ===========================================================================

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "jXOD1ZX2TXyI9qGJtyXeEQ9k5s3YhL6I")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = (
    "mistral-large-latest"  # best quality; swap to mistral-small to save quota
)

# Rate-limit / retry settings
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2  # seconds — doubles each retry (exponential back-off)
REQUEST_TIMEOUT = 120  # seconds per call

# Generation settings
TARGET_CHAPTERS = 25  # approximate book length
SCENES_PER_CHAP = 3  # scenes generated per chapter


# ===========================================================================
# MISTRAL CLIENT  (minimal, no SDK dependency)
# ===========================================================================


def call_mistral(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """
    Single Mistral call with exponential back-off on rate limits (429)
    and transient errors (5xx).  Raises RuntimeError after MAX_RETRIES.
    """
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    delay = RETRY_BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                MISTRAL_API_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            # Rate limited
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", delay))
                wait = max(retry_after, delay)
                print(
                    f"  [rate-limit] waiting {wait}s (attempt {attempt}/{MAX_RETRIES})"
                )
                time.sleep(wait)
                delay *= 2
                continue

            # Transient server error
            if resp.status_code >= 500:
                print(f"  [server-error {resp.status_code}] retrying in {delay}s")
                time.sleep(delay)
                delay *= 2
                continue

            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

        except requests.exceptions.Timeout:
            print(f"  [timeout] retrying in {delay}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(delay)
            delay *= 2
        except requests.exceptions.ConnectionError:
            print(f"  [connection-error] retrying in {delay}s")
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(f"Mistral call failed after {MAX_RETRIES} retries.")


def parse_json_response(raw: str) -> dict | list:
    """Strip markdown fences and parse JSON from LLM response with repair attempts."""
    # Remove markdown code blocks
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

    # Attempt 1: Direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Attempt 2: Try to fix unterminated strings (common truncation issue)
        if "Unterminated string" in str(e):
            # Add missing closing quote and brace
            if cleaned.rstrip().endswith('"'):
                cleaned = cleaned + '"}'
            else:
                # Find the last quote and close it properly
                lines = cleaned.split("\n")
                last_line = lines[-1]
                if '"' in last_line and not last_line.rstrip().endswith('"'):
                    # Add closing quote before the end
                    cleaned = cleaned + '"'
                cleaned = cleaned + "}"

            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

        # Attempt 3: Try to find complete JSON by counting braces
        brace_count = 0
        in_string = False
        escape_next = False
        valid_end = 0

        for i, ch in enumerate(cleaned):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if not in_string:
                if ch == "{":
                    brace_count += 1
                elif ch == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        valid_end = i + 1
                        break

        if valid_end > 0:
            truncated_json = cleaned[:valid_end]
            try:
                return json.loads(truncated_json)
            except json.JSONDecodeError:
                pass

        raise ValueError(f"LLM returned invalid JSON: {e}\n\nRaw:\n{raw[:500]}")


# ===========================================================================
# STAGE 1 — CONTEXT COMPILATION  (no LLM)
# ===========================================================================


def compile_context(retrieval_json: dict, user_prompt: str) -> dict:
    """
    Transform the raw retrieval JSON into a lean, structured story-bible
    that every downstream stage will read.

    No LLM call — pure data transformation.
    This is where we decide what matters and discard the noise.
    """
    print("[Stage 1] Compiling context from retrieval data...")

    meta = retrieval_json.get("meta", {})
    ending = retrieval_json.get("story_ending", {})
    char_states = retrieval_json.get("character_states", [])
    relations = retrieval_json.get("relationship_summary", [])
    threads = retrieval_json.get("unresolved_threads", [])
    chains = retrieval_json.get("causal_chains", [])
    flexible = retrieval_json.get("flexible_events", [])
    trajectories = retrieval_json.get("character_trajectories", [])

    # --- Characters: keep only the most relevant fields, cap to top 10 ---
    slim_characters = []
    for ch in char_states[:10]:
        canon = ch.get("canon_state", {})
        last_trans = (
            ch.get("state_transitions", [])[-3:] if ch.get("state_transitions") else []
        )
        slim_characters.append(
            {
                "name": ch["name"],
                "descriptions": ch.get("descriptions", [])[:2],  # top 2 descriptions
                "canon_state": canon,
                "recent_changes": last_trans,  # last 3 state transitions
                "aliases": ch.get("aliases", []),
            }
        )

    # --- Relationships: only the most recent / significant ---
    slim_relations = [
        {
            "between": f"{r['entity_a']} ↔ {r['entity_b']}",
            "type": r.get("relationship_type"),
            "latest": r.get("latest_change"),
            "evidence": r.get("evidence"),
        }
        for r in relations[:15]
    ]

    # --- Unresolved threads: sorted by divergence potential ---
    slim_threads = [
        {
            "event": t.get("event_description"),
            "decision": t.get("decision_made"),
            "alternatives": t.get("alternatives"),
            "potential": t.get("divergence_potential"),
        }
        for t in sorted(
            threads, key=lambda x: x.get("divergence_potential", 0), reverse=True
        )[:8]
    ]

    # --- Causal chains: keep description + story function, skip raw events ---
    slim_chains = [
        {
            "id": c.get("chain_id"),
            "description": c.get("description"),
            "type": c.get("chain_type"),
            "function": c.get("story_function"),
            "event_count": len(c.get("events", [])),
        }
        for c in chains
    ]

    # --- Story ending anchor ---
    last_scene = ending.get("last_scene", {})
    critical_tail = ending.get("critical_path_tail", [])

    story_ending_summary = {
        "last_scene_summary": last_scene.get("summary", ""),
        "entities_present": [e["name"] for e in last_scene.get("entities_present", [])],
        "location": (
            last_scene.get("location", {}).get("name")
            if last_scene.get("location")
            else None
        ),
        "critical_events": [
            e.get("description") for e in critical_tail[-5:]
        ],  # last 5 critical beats
    }

    # --- Flexible events: creative latitude zones ---
    slim_flexible = [
        {"description": f.get("description"), "score": f.get("flexibility_score")}
        for f in flexible[:5]
    ]

    compiled = {
        "book_title": meta.get("book_title", "Unknown"),
        "user_prompt": user_prompt,
        "story_ending": story_ending_summary,
        "characters": slim_characters,
        "relationships": slim_relations,
        "unresolved_threads": slim_threads,
        "causal_chains": slim_chains,
        "flexible_events": slim_flexible,
        "character_trajectories": [
            {
                "character": t["character"],
                "last_events": [e.get("description") for e in t.get("last_events", [])],
            }
            for t in trajectories
        ],
    }

    print(
        f"  -> Compiled: {len(slim_characters)} characters, "
        f"{len(slim_threads)} threads, {len(slim_chains)} causal chains"
    )
    return compiled


# ===========================================================================
# STAGE 2 — BLUEPRINT GENERATION  (1 LLM call)
# ===========================================================================

BLUEPRINT_SYSTEM = """
You are a master story architect. You will be given a compiled story bible 
from an existing book and a user's creative direction. Your job is to design 
the blueprint for a full sequel novel.

RULES:
- Do NOT impose a rigid 3-act structure. Infer the right structure from the 
  source material's own narrative shape and the characters' open arcs.
- Respect all critical canon events — these are ground truth.
- The sequel must flow naturally from where the source book ended.
- Weight the user's creative direction when deciding which unresolved threads 
  to activate as the central conflict.
- Output ONLY valid JSON, no markdown fences, no preamble.

OUTPUT SCHEMA:
{
  "title": "proposed sequel title",
  "premise": "2-3 sentence summary of what this book is about",
  "structure_type": "e.g. linear, episodic, dual-timeline, etc.",
  "total_chapters": <integer>,
  "central_conflict": "the main conflict driving the story",
  "primary_arcs": [
    {
      "arc_name": "e.g. Nesta's Redemption",
      "character": "character name",
      "starts_at": "where this arc begins emotionally/situationally",
      "ends_at": "where it resolves",
      "key_turning_point": "the single event that changes everything for this arc"
    }
  ],
  "acts": [
    {
      "label": "e.g. Part One / Act 1 / Opening",
      "chapter_range": "e.g. 1-7",
      "narrative_goal": "what this section must accomplish",
      "ends_with": "the event or revelation that closes this section",
      "dominant_arcs": ["arc names active here"]
    }
  ],
  "world_threads_activated": ["list of unresolved threads being used"],
  "tone": "the emotional register of this book"
}
"""


def generate_blueprint(compiled_context: dict) -> dict:
    """Stage 2: One LLM call to generate the full book blueprint."""
    print("[Stage 2] Generating book blueprint...")

    user_prompt = f"""
STORY BIBLE:
{json.dumps(compiled_context, indent=2, ensure_ascii=False)}

USER DIRECTION: "{compiled_context['user_prompt']}"

Design the sequel blueprint now. Output only valid JSON.
"""

    raw = call_mistral(
        system_prompt=BLUEPRINT_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.8,
        max_tokens=3000,
    )

    blueprint = parse_json_response(raw)
    print(
        f"  -> Blueprint: '{blueprint.get('title')}' | "
        f"{blueprint.get('total_chapters')} chapters | "
        f"{len(blueprint.get('acts', []))} acts"
    )
    return blueprint


# ===========================================================================
# STAGE 3 — CHAPTER OUTLINES  (1 LLM call per chapter)
# ===========================================================================

OUTLINE_SYSTEM = """
You are a narrative planner. Given a book blueprint, the current world state, 
and previous chapter summaries, produce a detailed outline for the NEXT chapter.

RULES:
- Stay consistent with the current world state (character positions, 
  relationship dynamics, active conflicts).
- Advance at least one primary arc meaningfully.
- Each chapter must end on a beat that pulls the reader forward.
- Output ONLY valid JSON, no markdown fences.

OUTPUT SCHEMA:
{
  "chapter_number": <int>,
  "chapter_title": "string",
  "pov_character": "whose perspective",
  "location": "where this takes place",
  "scenes": [
    {
      "scene_number": <int>,
      "summary": "what happens in 2-3 sentences",
      "characters_present": ["names"],
      "purpose": "what narrative work this scene does",
      "ends_on": "the beat or image this scene closes on"
    }
  ],
  "arc_progress": {
    "arc_name": "what changes in this arc this chapter"
  },
  "world_state_changes": [
    "concise description of any state/relationship change that occurs"
  ],
  "chapter_closes_on": "the final beat of the chapter"
}
"""


def generate_chapter_outline(
    blueprint: dict,
    compiled_context: dict,
    world_state: dict,
    previous_summaries: list[str],
    chapter_number: int,
) -> dict:
    """Stage 3: Generate outline for one chapter with retry on JSON failure."""

    # Find which act this chapter belongs to
    current_act = {}
    for act in blueprint.get("acts", []):
        rng = act.get("chapter_range", "")
        parts = rng.replace(" ", "").split("-")
        if len(parts) == 2:
            try:
                if int(parts[0]) <= chapter_number <= int(parts[1]):
                    current_act = act
                    break
            except ValueError:
                pass

    recent_summaries = previous_summaries[-3:] if previous_summaries else []

    user_prompt = f"""
BLUEPRINT:
{json.dumps(blueprint, indent=2)}

CURRENT WORLD STATE:
{json.dumps(world_state, indent=2)}

RECENT CHAPTER SUMMARIES (last {len(recent_summaries)}):
{json.dumps(recent_summaries, indent=2)}

CURRENT ACT CONTEXT:
{json.dumps(current_act, indent=2)}

Now generate the outline for CHAPTER {chapter_number}.
Output only valid JSON.
"""

    max_parse_retries = 3
    for parse_attempt in range(max_parse_retries):
        raw = call_mistral(
            system_prompt=OUTLINE_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.75,
            max_tokens=3000,  # Increased from 1500
        )

        try:
            outline = parse_json_response(raw)
            break
        except ValueError as e:
            print(
                f"  [!] JSON parse failed (attempt {parse_attempt + 1}/{max_parse_retries})"
            )
            if parse_attempt == max_parse_retries - 1:
                raise
            # Retry with a more explicit prompt about JSON format
            user_prompt += "\n\nIMPORTANT: Your response must be complete, valid JSON. Do not cut off mid-sentence. Ensure all strings are properly closed with quotes."

    print(
        f"  -> Chapter {chapter_number}: '{outline.get('chapter_title')}' | "
        f"{len(outline.get('scenes', []))} scenes | POV: {outline.get('pov_character')}"
    )
    return outline


# ===========================================================================
# STAGE 4 — SCENE PROSE  (1 LLM call per scene)
# ===========================================================================

PROSE_SYSTEM = """
You are a fiction writer continuing a novel. You will be given a scene outline, 
the characters involved, the current world state, and the previous scene's 
closing lines. Write the full prose for this scene.

RULES:
- Match the tone and style of the source book (fantasy romance, emotional depth,
  close third-person or first-person POV matching the POV character).
- Stay true to each character's voice and established personality.
- Do NOT introduce new major characters or contradict canon facts.
- Do NOT summarise — write scene prose in full.
- End the scene on the beat described in the outline.
- Length: 600-1200 words per scene.
- Output ONLY the prose, no JSON, no titles, no commentary.
"""


def generate_scene_prose(
    scene_outline: dict,
    chapter_outline: dict,
    world_state: dict,
    previous_scene_ending: str,
    book_title: str,
) -> str:
    """Stage 4: Generate prose for one scene."""

    # Pull only the characters present in this scene
    present_names = scene_outline.get("characters_present", [])
    relevant_chars = [
        c for c in world_state.get("characters", []) if c["name"] in present_names
    ]
    relevant_rels = [
        r
        for r in world_state.get("relationships", [])
        if any(name in r["between"] for name in present_names)
    ]

    user_prompt = f"""
SOURCE BOOK: {book_title}
CHAPTER: {chapter_outline.get('chapter_number')} — {chapter_outline.get('chapter_title')}
POV CHARACTER: {chapter_outline.get('pov_character')}
LOCATION: {scene_outline.get('location', chapter_outline.get('location', 'unknown'))}

SCENE OUTLINE:
{json.dumps(scene_outline, indent=2)}

CHARACTERS IN THIS SCENE:
{json.dumps(relevant_chars, indent=2)}

RELEVANT RELATIONSHIPS:
{json.dumps(relevant_rels, indent=2)}

PREVIOUS SCENE ENDED WITH:
\"\"\"{previous_scene_ending}\"\"\"

Write the full prose for this scene now.
"""

    prose = call_mistral(
        system_prompt=PROSE_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.85,
        max_tokens=3000,
    )
    return prose


# ===========================================================================
# WORLD STATE MANAGER
# ===========================================================================


def initialise_world_state(compiled_context: dict) -> dict:
    """
    The world state starts as the compiled context and evolves
    chapter by chapter as we apply the outlined changes.
    """
    return {
        "characters": compiled_context["characters"].copy(),
        "relationships": compiled_context["relationships"].copy(),
        "active_threads": compiled_context["unresolved_threads"].copy(),
        "events_so_far": [],
    }


def update_world_state(world_state: dict, chapter_outline: dict) -> dict:
    """
    Rule-based consistency pass — apply the world_state_changes
    from the chapter outline without any LLM call.
    """
    changes = chapter_outline.get("world_state_changes", [])
    if changes:
        world_state["events_so_far"].extend(changes)

    # Keep events list from growing unbounded — keep last 50
    if len(world_state["events_so_far"]) > 50:
        world_state["events_so_far"] = world_state["events_so_far"][-50:]

    return world_state


def chapter_summary_from_outline(outline: dict) -> str:
    """
    Derive a compact chapter summary from the outline for the rolling
    context window — no extra LLM call needed.
    """
    scenes = outline.get("scenes", [])
    scene_text = " ".join(s.get("summary", "") for s in scenes)
    return (
        f"Chapter {outline.get('chapter_number')} — "
        f"{outline.get('chapter_title')}: {scene_text} "
        f"[Closes on: {outline.get('chapter_closes_on', '')}]"
    )


# ===========================================================================
# CONSISTENCY CHECKER  (rule-based, no LLM)
# ===========================================================================


def check_consistency(
    prose: str, chapter_outline: dict, world_state: dict
) -> list[str]:
    """
    Lightweight rule-based consistency check.
    Returns a list of warning strings (empty = all clear).
    """
    warnings = []
    known_names = {c["name"].lower() for c in world_state.get("characters", [])}

    # Check that POV character appears in the prose
    pov = chapter_outline.get("pov_character", "")
    if pov and pov.lower() not in prose.lower():
        warnings.append(f"POV character '{pov}' not found in prose")

    # Check none of the known critical canon events are being contradicted
    # (simple keyword presence check — extend this as encoder matures)
    for event in world_state.get("events_so_far", [])[-5:]:
        pass  # placeholder for deeper checks when encoder provides richer data

    return warnings


# ===========================================================================
# OUTPUT WRITER
# ===========================================================================


def save_chapter(
    output_dir: Path,
    chapter_number: int,
    chapter_title: str,
    scenes_prose: list[str],
) -> Path:
    """Write a chapter's prose to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"chapter_{chapter_number:02d}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"CHAPTER {chapter_number}\n{chapter_title}\n\n")
        f.write("\n\n---\n\n".join(scenes_prose))
    return filename


def save_progress(output_dir: Path, state: dict) -> None:
    """Save pipeline progress so it can be resumed if interrupted."""
    progress_file = output_dir / "progress.json"
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)


def load_progress(output_dir: Path) -> Optional[dict]:
    """Load existing progress to resume an interrupted run."""
    progress_file = output_dir / "progress.json"
    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ===========================================================================
# MAIN PIPELINE ORCHESTRATOR
# ===========================================================================


def generate_sequel(
    retrieval_json_path: str,
    user_prompt: str,
    output_dir: str = "output/sequel",
) -> Path:
    """
    Full 4-stage sequel generation pipeline.

    Args:
        retrieval_json_path : path to the JSON produced by retrieval.py
        user_prompt         : user's creative direction e.g. "focus on Nesta and Cassian"
        output_dir          : where to write the generated chapters

    Returns:
        Path to the output directory
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- Load retrieval data ---
    with open(retrieval_json_path, "r", encoding="utf-8") as f:
        retrieval_json = json.load(f)

    # --- Check for existing progress (resume support) ---
    progress = load_progress(out_path)
    start_chapter = 1
    previous_summaries = []
    world_state = None
    blueprint = None
    compiled = None

    if progress:
        print(
            f"[Resume] Found existing progress — resuming from chapter "
            f"{progress.get('last_completed_chapter', 0) + 1}"
        )
        blueprint = progress["blueprint"]
        compiled = progress["compiled_context"]
        world_state = progress["world_state"]
        previous_summaries = progress["previous_summaries"]
        start_chapter = progress["last_completed_chapter"] + 1
    else:
        # ---------------------------------------------------------------
        # STAGE 1: Context Compilation
        # ---------------------------------------------------------------
        compiled = compile_context(retrieval_json, user_prompt)

        # ---------------------------------------------------------------
        # STAGE 2: Blueprint (single LLM call)
        # ---------------------------------------------------------------
        blueprint = generate_blueprint(compiled)
        blueprint["total_chapters"] = blueprint.get("total_chapters", TARGET_CHAPTERS)

        # Save blueprint
        with open(out_path / "blueprint.json", "w", encoding="utf-8") as f:
            json.dump(blueprint, f, indent=2, ensure_ascii=False)

        # Initialise world state from compiled context
        world_state = initialise_world_state(compiled)

    total_chapters = blueprint.get("total_chapters", TARGET_CHAPTERS)
    print(
        f"\n[Pipeline] Generating '{blueprint.get('title')}' — "
        f"{total_chapters} chapters\n"
    )

    # -----------------------------------------------------------------------
    # STAGES 3 + 4: Chapter by Chapter
    # -----------------------------------------------------------------------
    for chapter_num in range(start_chapter, total_chapters + 1):
        print(f"\n{'='*60}")
        print(f"[Chapter {chapter_num}/{total_chapters}]")
        print(f"{'='*60}")

        # --- Stage 3: Chapter outline (1 LLM call) ---
        print("[Stage 3] Generating chapter outline...")
        outline = generate_chapter_outline(
            blueprint=blueprint,
            compiled_context=compiled,
            world_state=world_state,
            previous_summaries=previous_summaries,
            chapter_number=chapter_num,
        )

        # --- Stage 4: Scene prose (1 LLM call per scene) ---
        print("[Stage 4] Generating scene prose...")
        scenes = outline.get("scenes", [])
        scenes_prose = []
        last_ending = compiled["story_ending"].get("last_scene_summary", "")

        for scene in scenes:
            scene_num = scene.get("scene_number", len(scenes_prose) + 1)
            print(
                f"  -> Scene {scene_num}/{len(scenes)}: {scene.get('summary', '')[:60]}..."
            )

            prose = generate_scene_prose(
                scene_outline=scene,
                chapter_outline=outline,
                world_state=world_state,
                previous_scene_ending=last_ending,
                book_title=compiled["book_title"],
            )

            # Consistency check (rule-based, free)
            warnings = check_consistency(prose, outline, world_state)
            for w in warnings:
                print(f"  [!] Consistency warning: {w}")

            scenes_prose.append(prose)
            # Last ~150 chars as the "ending" for the next scene's context
            last_ending = prose[-150:].strip()

            # Small delay between scene calls to be kind to rate limits
            time.sleep(0.5)

        # --- Write chapter to disk ---
        chapter_file = save_chapter(
            output_dir=out_path,
            chapter_number=chapter_num,
            chapter_title=outline.get("chapter_title", f"Chapter {chapter_num}"),
            scenes_prose=scenes_prose,
        )
        print(f"  -> Saved: {chapter_file}")

        # --- Update world state (rule-based, no LLM) ---
        world_state = update_world_state(world_state, outline)

        # --- Append rolling chapter summary (no LLM) ---
        previous_summaries.append(chapter_summary_from_outline(outline))

        # --- Save progress after every chapter ---
        save_progress(
            out_path,
            {
                "blueprint": blueprint,
                "compiled_context": compiled,
                "world_state": world_state,
                "previous_summaries": previous_summaries,
                "last_completed_chapter": chapter_num,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        # Courtesy delay between chapters
        time.sleep(1)

    print(f"\n[SAGA] Generation complete. Output: {out_path.resolve()}")
    return out_path


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAGA Sequel Generator")
    parser.add_argument(
        "--context",
        default="sequel_context.json",
        help="Path to the retrieval JSON from retrieval.py",
    )
    parser.add_argument(
        "--prompt",
        default="Focus on Nesta and Cassian's relationship arc",
        help="Your creative direction for the sequel",
    )
    parser.add_argument(
        "--output",
        default="output/sequel",
        help="Output directory for generated chapters",
    )
    args = parser.parse_args()

    generate_sequel(
        retrieval_json_path=args.context,
        user_prompt=args.prompt,
        output_dir=args.output,
    )
