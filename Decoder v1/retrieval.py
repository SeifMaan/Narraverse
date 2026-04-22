import json
import argparse
from datetime import datetime, timezone
from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_AUTH = ("neo4j", "12345678")
DB_NAME = "db5"

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)


def run(session, query, **params):
    return session.run(query, **params).data()


# ===========================================================================
# RETRIEVAL FUNCTIONS
# ===========================================================================


def get_book_meta(s, book_title):
    """Basic book info."""
    rows = run(s, "MATCH (b:Book {title: $title}) RETURN b", title=book_title)
    if not rows:
        raise ValueError(f"Book '{book_title}' not found in database.")
    return dict(rows[0]["b"])


def get_ending_events(s, book_title, top_n=10):
    """
    Critical path events ordered by critical_order descending.
    These are the story beats the decoder must treat as ground truth.
    """
    rows = run(
        s,
        """
        MATCH (b:Book {title: $title})-[:HAS_EVENT]->(e:Event)
        WHERE e.is_critical = true
        RETURN e.id             AS id,
               e.description    AS description,
               e.chapter_index  AS chapter,
               e.criticality_score AS score,
               e.why_critical   AS why_critical,
               e.critical_order AS order,
               e.story_impact   AS story_impact
        ORDER BY e.critical_order DESC
        LIMIT $n
        """,
        title=book_title,
        n=top_n,
    )
    # Re-sort chronologically for the decoder
    return sorted(rows, key=lambda r: r.get("order") or 0)


def get_character_states(s, book_title):
    """
    Latest known state per character:
    - canon_* properties (from canon snapshot)
    - all StateTransition nodes ordered chronologically
    - aliases
    """
    # Entities linked to the book that are characters
    entities = run(
        s,
        """
        MATCH (b:Book {title: $title})-[:HAS_ENTITY]->(e:Entity)
        WHERE e.entity_type = 'character'
        RETURN e.name           AS name,
               e.mention_count  AS mention_count,
               e.descriptions   AS descriptions,
               e.first_seen_ch  AS first_seen_chapter,
               properties(e)    AS all_props
        ORDER BY e.mention_count DESC
        """,
        title=book_title,
    )

    results = []
    for ent in entities:
        name = ent["name"]

        # Extract canon_* properties
        canon_state = {
            k.replace("canon_", ""): v
            for k, v in (ent["all_props"] or {}).items()
            if k.startswith("canon_")
        }

        # Get aliases
        aliases = run(
            s,
            """
            MATCH (e:Entity {name: $name})-[:HAS_ALIAS]->(a:Alias)
            RETURN a.text AS alias
            """,
            name=name,
        )

        # Get state transitions ordered chronologically
        transitions = run(
            s,
            """
            MATCH (e:Entity {name: $name})-[:HAD_STATE_CHANGE]->(st:StateTransition)
            RETURN st.attribute      AS attribute,
                   st.previous_state AS previous_state,
                   st.new_state      AS new_state,
                   st.change_type    AS change_type,
                   st.evidence       AS evidence,
                   st.chapter_index  AS chapter
            ORDER BY st.chapter_index ASC
            """,
            name=name,
        )

        results.append(
            {
                "name": name,
                "mention_count": ent["mention_count"],
                "first_seen_chapter": ent["first_seen_chapter"],
                "descriptions": ent.get("descriptions") or [],
                "aliases": [a["alias"] for a in aliases],
                "canon_state": canon_state,
                "state_transitions": [dict(t) for t in transitions],
            }
        )

    return results


def get_relationship_summary(s):
    """
    Current relationship state between character pairs.
    These are the HAS_RELATIONSHIP summary edges — one per pair,
    reflecting the latest known dynamic.
    """
    rows = run(
        s,
        """
        MATCH (a:Entity)-[r:HAS_RELATIONSHIP]->(b:Entity)
        RETURN a.name             AS entity_a,
               b.name             AS entity_b,
               r.type             AS relationship_type,
               r.latest_change    AS latest_change,
               r.latest_evidence  AS evidence,
               r.last_seen_ch     AS last_seen_chapter
        ORDER BY r.last_seen_ch DESC
        """,
    )
    return [dict(r) for r in rows]


def get_unresolved_threads(s, book_title, min_potential=7):
    """
    DivergencePoints with high potential = open story threads.
    These are the premises the decoder can build a sequel around.
    """
    rows = run(
        s,
        """
        MATCH (e:Event)-[:IS_DIVERGENCE_POINT]->(d:DivergencePoint)
        WHERE d.divergence_potential >= $min_potential
        RETURN e.id                    AS event_id,
               e.description           AS event_description,
               e.chapter_index         AS chapter,
               e.is_critical           AS is_critical,
               d.decision_made         AS decision_made,
               d.alternatives          AS alternatives,
               d.divergence_potential  AS divergence_potential,
               d.alternate_timeline    AS alternate_timeline
        ORDER BY d.divergence_potential DESC
        """,
        min_potential=min_potential,
    )
    return [dict(r) for r in rows]


def get_causal_chains(s):
    """
    Named story arcs with their event sequences.
    Gives the decoder the macro structure of the story.
    """
    chains = run(
        s,
        """
        MATCH (cc:CausalChain)
        RETURN cc.chain_id      AS chain_id,
               cc.description   AS description,
               cc.chain_type    AS chain_type,
               cc.story_function AS story_function
        """,
    )
    result = []
    for chain in chains:
        chain_id = chain["chain_id"]
        events = run(
            s,
            """
            MATCH (e:Event)-[:IN_CHAIN]->(cc:CausalChain {chain_id: $chain_id})
            RETURN e.id          AS event_id,
                   e.description AS description,
                   e.chapter_index AS chapter,
                   e.time_index  AS time_index
            ORDER BY e.time_index ASC
            """,
            chain_id=chain_id,
        )
        result.append(
            {
                **dict(chain),
                "events": [dict(e) for e in events],
            }
        )
    return result


def get_flexible_events(s, book_title):
    """
    Events flagged is_flexible=true — safe creative latitude zones
    the decoder can alter, skip, or remix without breaking the critical spine.
    """
    rows = run(
        s,
        """
        MATCH (b:Book {title: $title})-[:HAS_EVENT]->(e:Event)
        WHERE e.is_flexible = true
        RETURN e.id               AS event_id,
               e.description      AS description,
               e.chapter_index    AS chapter,
               e.flexibility_score AS flexibility_score,
               e.why_flexible     AS why_flexible
        ORDER BY e.flexibility_score DESC
        """,
        title=book_title,
    )
    return [dict(r) for r in rows]


def get_last_scene(s, book_title):
    """
    The final scene of the book — summary, location, who was present,
    what relationship and state changes occurred.
    """
    scene = run(
        s,
        """
        MATCH (b:Book {title: $title})-[:HAS_CHAPTER]->(ch:Chapter)-[:HAS_SCENE]->(sc:Scene)
        RETURN sc.summary      AS summary,
               sc.book_index   AS book_index,
               sc.chapter_index AS chapter_index,
               sc.scene_index  AS scene_index
        ORDER BY sc.chapter_index DESC, sc.scene_index DESC
        LIMIT 1
        """,
        title=book_title,
    )
    if not scene:
        return {}

    sc = dict(scene[0])
    bi, ci, si = sc["book_index"], sc["chapter_index"], sc["scene_index"]

    # Who was present
    present = run(
        s,
        """
        MATCH (sc:Scene {book_index: $bi, chapter_index: $ci, scene_index: $si})
              -[:FEATURES]->(e:Entity)
        RETURN e.name AS name, e.entity_type AS entity_type
        """,
        bi=bi,
        ci=ci,
        si=si,
    )

    # Location
    location = run(
        s,
        """
        MATCH (sc:Scene {book_index: $bi, chapter_index: $ci, scene_index: $si})
              -[:LOCATED_IN]->(l:Entity)
        RETURN l.name AS name, l.description AS description
        """,
        bi=bi,
        ci=ci,
        si=si,
    )

    # Relationship changes in this scene
    rel_changes = run(
        s,
        """
        MATCH (sc:Scene {book_index: $bi, chapter_index: $ci, scene_index: $si})
              -[:HAS_RELATIONSHIP_CHANGE]->(rc:RelationshipChange)
        RETURN rc.source_entity AS source,
               rc.target_entity AS target,
               rc.relationship  AS relationship,
               rc.change        AS change,
               rc.evidence      AS evidence
        """,
        bi=bi,
        ci=ci,
        si=si,
    )

    # State changes in this scene
    state_changes = run(
        s,
        """
        MATCH (sc:Scene {book_index: $bi, chapter_index: $ci, scene_index: $si})
              -[:HAS_SCENE]-(:Chapter)
        WITH sc
        MATCH (e:Entity)-[:HAD_STATE_CHANGE]->(st:StateTransition {
            chapter_index: $ci,
            scene_index: $si
        })
        RETURN e.name       AS entity,
               st.attribute AS attribute,
               st.new_state AS new_state,
               st.evidence  AS evidence
        """,
        bi=bi,
        ci=ci,
        si=si,
    )

    return {
        **sc,
        "location": location[0] if location else None,
        "entities_present": [dict(p) for p in present],
        "relationship_changes": [dict(r) for r in rel_changes],
        "state_changes": [dict(r) for r in state_changes],
    }


def get_character_timelines_summary(s, top_n_chars=6):
    """
    For the most mentioned characters, get their last 5 events
    so the decoder knows their recent trajectory.
    """
    top_chars = run(
        s,
        """
        MATCH (e:Entity)
        WHERE e.entity_type = 'character' AND e.mention_count IS NOT NULL
        RETURN e.name AS name
        ORDER BY e.mention_count DESC
        LIMIT $n
        """,
        n=top_n_chars,
    )

    result = []
    for char in top_chars:
        name = char["name"]
        events = run(
            s,
            """
            MATCH (c:Entity {name: $name})-[r:APPEARS_IN_EVENT]->(e:Event)
            RETURN e.id           AS event_id,
                   e.description  AS description,
                   e.chapter_index AS chapter,
                   r.time_index   AS time_index
            ORDER BY r.time_index DESC
            LIMIT 5
            """,
            name=name,
        )
        result.append(
            {
                "character": name,
                "last_events": [dict(e) for e in reversed(events)],
            }
        )
    return result


# ===========================================================================
# ASSEMBLE
# ===========================================================================


def retrieve_sequel_context(book_title, out_path):
    print(f"[SAGA] Retrieving sequel context for: {book_title}")

    with driver.session(database=DB_NAME) as s:

        print("  -> Book metadata...")
        book_meta = get_book_meta(s, book_title)

        print("  -> Ending events (critical path tail)...")
        ending_events = get_ending_events(s, book_title, top_n=10)

        print("  -> Last scene...")
        last_scene = get_last_scene(s, book_title)

        print("  -> Character states...")
        character_states = get_character_states(s, book_title)

        print("  -> Relationship summary...")
        relationships = get_relationship_summary(s)

        print("  -> Unresolved threads (divergence points)...")
        unresolved_threads = get_unresolved_threads(s, book_title, min_potential=7)

        print("  -> Causal chains...")
        causal_chains = get_causal_chains(s)

        print("  -> Flexible events...")
        flexible_events = get_flexible_events(s, book_title)

        print("  -> Character recent trajectories...")
        char_trajectories = get_character_timelines_summary(s, top_n_chars=6)

    # -----------------------------------------------------------------------
    # Build the context document
    # -----------------------------------------------------------------------
    context = {
        "meta": {
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "retrieval_type": "sequel_setup",
            "book_title": book_title,
        },
        # What the decoder must treat as fixed ground truth
        "story_ending": {
            "last_scene": last_scene,
            "critical_path_tail": ending_events,
        },
        # Who the characters are and where they ended up
        "character_states": character_states,
        # Current relationship dynamics between pairs
        "relationship_summary": relationships,
        # Open story threads — the decoder's sequel premises
        "unresolved_threads": unresolved_threads,
        # Macro story arcs — structural context for the decoder
        "causal_chains": causal_chains,
        # Where the decoder has creative freedom
        "flexible_events": flexible_events,
        # Recent trajectory per main character
        "character_trajectories": char_trajectories,
        # Summary stats for quick inspection
        "stats": {
            "critical_ending_events": len(ending_events),
            "characters_retrieved": len(character_states),
            "relationship_pairs": len(relationships),
            "unresolved_threads": len(unresolved_threads),
            "causal_chains": len(causal_chains),
            "flexible_events": len(flexible_events),
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[SAGA] Done. Context saved to: {out_path}")
    print(f"       Stats: {json.dumps(context['stats'], indent=8)}")
    return context


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAGA Sequel Retrieval Layer")
    parser.add_argument(
        "--book",
        default="A Court of Frost and Starlight.epub",
        help="Book title as stored in Neo4j",
    )
    parser.add_argument(
        "--out",
        default="sequel_context.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    retrieve_sequel_context(args.book, args.out)
