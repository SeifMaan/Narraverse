from fastapi import FastAPI, Body, HTTPException
from neo4j import GraphDatabase
import traceback

app = FastAPI()

driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "12345678"))

DB_NAME = "db5"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def run(session, query, **params):
    """Thin wrapper so every call goes through one place."""
    session.run(query, **params)


# ---------------------------------------------------------------------------
# Ingestion endpoint
# ---------------------------------------------------------------------------


@app.post("/ingest_saga_contract")
def ingest_saga_contract(payload: dict = Body(...)):
    """
    Ingests a full SAGA contract JSON into Neo4j.

    Graph nodes created:
        Book, Chapter, Scene, Entity (Character / Location / other),
        Alias, Event, StateTransition, CausalChain, DivergencePoint

    Key relationships:
        Book -[:HAS_CHAPTER]-> Chapter -[:HAS_SCENE]-> Scene
        Book  -[:HAS_ENTITY]-> Entity
        Entity -[:HAS_ALIAS]-> Alias
        Scene  -[:LOCATED_IN]-> Entity (location)
        Scene  -[:FEATURES]-> Entity (characters present)
        Scene  -[:HAS_EVENT]-> Event
        Scene  -[:HAS_RELATIONSHIP_CHANGE]-> RelationshipChange -> Entity x2
        Entity -[:HAD_STATE_CHANGE]-> StateTransition
        Event  -[:CAUSES / :CAUSED_BY / :PREVENTS / :REQUIRED_FOR]-> Event
        Event  -[:CRITICAL_NEXT]-> Event (critical path chain)
        Event  -[:IN_CHAIN]-> CausalChain
        Event  -[:IS_DIVERGENCE_POINT]-> DivergencePoint
        Entity -[:APPEARS_IN_EVENT]-> Event (character_timelines)
    """

    try:
        with driver.session(database=DB_NAME) as s:

            # ------------------------------------------------------------------
            # 0. TOP-LEVEL METADATA
            # ------------------------------------------------------------------
            inputs_meta = payload.get("inputs", {})
            books_meta = inputs_meta.get("books", [{}])
            book_title = (
                books_meta[0].get("title", "Unknown Book")
                if books_meta
                else "Unknown Book"
            )
            generated_at = payload.get("generated_at_utc", "")
            contract_version = payload.get("contract_version", "")

            run(
                s,
                """
                MERGE (b:Book {title: $title})
                SET b.generated_at   = $generated_at,
                    b.contract_version = $contract_version
                """,
                title=book_title,
                generated_at=generated_at,
                contract_version=contract_version,
            )

            outputs = payload.get("outputs", {})

            # ------------------------------------------------------------------
            # 1. CHAPTERS
            # ------------------------------------------------------------------
            chapters = outputs.get("chapters", [])
            for ch in chapters:
                run(
                    s,
                    """
                    MATCH  (b:Book {title: $book_title})
                    MERGE  (c:Chapter {book_index: $bi, chapter_index: $ci})
                    SET    c.title = $title
                    MERGE  (b)-[:HAS_CHAPTER]->(c)
                    """,
                    book_title=book_title,
                    bi=ch.get("book_index"),
                    ci=ch.get("chapter_index"),
                    title=ch.get("chapter_title", ""),
                )

            # ------------------------------------------------------------------
            # 2. ENTITY REGISTRY  (characters, locations, etc.)
            # ------------------------------------------------------------------
            entity_registry = outputs.get("entity_registry", [])
            for ent in entity_registry:
                name = ent.get("name")
                if not name:
                    continue

                fs = ent.get("first_seen", {})
                run(
                    s,
                    """
                    MATCH  (b:Book {title: $book_title})
                    MERGE  (e:Entity {name: $name})
                    SET    e.entity_type      = $entity_type,
                           e.mention_count    = $mention_count,
                           e.first_seen_book  = $fs_book,
                           e.first_seen_ch    = $fs_ch,
                           e.first_seen_scene = $fs_scene
                    MERGE  (b)-[:HAS_ENTITY]->(e)
                    """,
                    book_title=book_title,
                    name=name,
                    entity_type=ent.get("entity_type", "unknown"),
                    mention_count=ent.get("mention_count", 0),
                    fs_book=fs.get("book_index"),
                    fs_ch=fs.get("chapter_index"),
                    fs_scene=fs.get("scene_index"),
                )

                # Entity descriptions as properties on the node (stored as list)
                descriptions = [
                    d.get("description", "")
                    for d in ent.get("descriptions", [])
                    if d.get("description")
                ]
                if descriptions:
                    run(
                        s,
                        "MATCH (e:Entity {name: $name}) SET e.descriptions = $descs",
                        name=name,
                        descs=descriptions,
                    )

                # State changes as individual nodes linked to the entity
                for sc in ent.get("state_changes", []):
                    run(
                        s,
                        """
                        MATCH (e:Entity {name: $name})
                        MERGE (st:StateTransition {
                            entity_name: $name,
                            attribute:   $attribute,
                            book_index:  $bi,
                            chapter_index: $ci,
                            scene_index: $si
                        })
                        SET st.previous_state = $prev,
                            st.new_state      = $new,
                            st.change_type    = $change_type,
                            st.evidence       = $evidence
                        MERGE (e)-[:HAD_STATE_CHANGE]->(st)
                        """,
                        name=name,
                        attribute=sc.get("attribute", ""),
                        bi=sc.get("book_index"),
                        ci=sc.get("chapter_index"),
                        si=sc.get("scene_index"),
                        prev=sc.get("previous_state", ""),
                        new=sc.get("new_state", ""),
                        change_type=sc.get("change_type", ""),
                        evidence=sc.get("evidence", ""),
                    )

            # ------------------------------------------------------------------
            # 3. ALIAS MAP  (identity_result)
            # ------------------------------------------------------------------
            alias_map = (
                payload.get("outputs", {})
                .get("identity_result", {})
                .get("alias_map", {})
            )
            for canonical, aliases in alias_map.items():
                # Ensure the canonical entity node exists (may not be in registry)
                run(
                    s,
                    "MERGE (e:Entity {name: $name})",
                    name=canonical,
                )
                for alias in aliases:
                    if alias == canonical:
                        continue  # skip self-alias
                    run(
                        s,
                        """
                        MATCH (e:Entity {name: $canonical})
                        MERGE (a:Alias {text: $alias})
                        MERGE (e)-[:HAS_ALIAS]->(a)
                        """,
                        canonical=canonical,
                        alias=alias,
                    )

            # ------------------------------------------------------------------
            # 4. STATE TRANSITIONS  (state_result.transitions — global list)
            # ------------------------------------------------------------------
            transitions = outputs.get("state_result", {}).get("transitions", [])
            for tr in transitions:
                entity_name = tr.get("entity_name")
                if not entity_name:
                    continue
                run(
                    s,
                    """
                    MATCH (e:Entity {name: $name})
                    MERGE (st:StateTransition {
                        entity_name:   $name,
                        attribute:     $attribute,
                        book_index:    $bi,
                        chapter_index: $ci,
                        scene_index:   $si
                    })
                    SET st.previous_state = $prev,
                        st.new_state      = $new,
                        st.change_type    = $change_type,
                        st.evidence       = $evidence,
                        st.state_index    = $state_index
                    MERGE (e)-[:HAD_STATE_CHANGE]->(st)
                    """,
                    name=entity_name,
                    attribute=tr.get("attribute", ""),
                    bi=tr.get("book_index"),
                    ci=tr.get("chapter_index"),
                    si=tr.get("scene_index"),
                    prev=tr.get("previous_state", ""),
                    new=tr.get("new_state", ""),
                    change_type=tr.get("change_type", ""),
                    evidence=tr.get("evidence", ""),
                    state_index=tr.get("state_index"),
                )

            # ------------------------------------------------------------------
            # 5. CANON SNAPSHOT  (latest known state per entity)
            # ------------------------------------------------------------------
            canon_snapshot = outputs.get("canon_snapshot", [])
            for snap in canon_snapshot:
                entity_name = snap.get("entity_name")
                if not entity_name:
                    continue
                attrs = snap.get("attributes", {})

                # Sanitize keys: Neo4j property names cannot contain spaces
                def safe_key(k):
                    return (
                        k.strip().replace(" ", "_").replace("-", "_").replace("/", "_")
                    )

                # Flatten attributes dict into the Entity node as canon_* properties
                set_clauses = ", ".join(
                    [f"e.canon_{safe_key(k)} = $attr_{safe_key(k)}" for k in attrs]
                )
                if set_clauses:
                    params = {f"attr_{safe_key(k)}": v for k, v in attrs.items()}
                    params["name"] = entity_name
                    run(
                        s,
                        f"MATCH (e:Entity {{name: $name}}) SET {set_clauses}",
                        **params,
                    )

            # ------------------------------------------------------------------
            # 6. SCENES  (from scene_analyses — we use resolved_scene_analyses
            #    for cleaner canonical character data, falling back to
            #    scene_analyses if absent)
            # ------------------------------------------------------------------
            resolved = outputs.get("resolved_scene_analyses") or outputs.get(
                "scene_analyses", []
            )

            for scene in resolved:
                bi = scene.get("book_index")
                ci = scene.get("chapter_index")
                si = scene.get("scene_index")

                # --- Scene node ---
                run(
                    s,
                    """
                    MERGE (sc:Scene {book_index: $bi, chapter_index: $ci, scene_index: $si})
                    SET sc.summary             = $summary,
                        sc.length              = $length,
                        sc.analysis_duration_s = $dur
                    WITH sc
                    MATCH (ch:Chapter {book_index: $bi, chapter_index: $ci})
                    MERGE (ch)-[:HAS_SCENE]->(sc)
                    """,
                    bi=bi,
                    ci=ci,
                    si=si,
                    summary=scene.get("scene_summary", ""),
                    length=scene.get("length", 0),
                    dur=scene.get("analysis_duration_seconds", 0.0),
                )

                # --- Location ---
                loc = scene.get("location")
                if loc and loc.get("name"):
                    loc_name = loc["name"]
                    run(
                        s,
                        """
                        MERGE (l:Entity {name: $name})
                        SET l.entity_type  = $etype,
                            l.description  = $desc
                        WITH l
                        MATCH (sc:Scene {book_index: $bi, chapter_index: $ci, scene_index: $si})
                        MERGE (sc)-[:LOCATED_IN]->(l)
                        """,
                        name=loc_name,
                        etype=loc.get("entity_type", "location"),
                        desc=loc.get("description", ""),
                        bi=bi,
                        ci=ci,
                        si=si,
                    )

                # --- Characters / entities present ---
                for ep in scene.get("entities_present", []):
                    ep_name = ep.get("name")
                    if not ep_name:
                        continue
                    run(
                        s,
                        """
                        MERGE (e:Entity {name: $name})
                        SET e.entity_type = coalesce(e.entity_type, $etype)
                        WITH e
                        MATCH (sc:Scene {book_index: $bi, chapter_index: $ci, scene_index: $si})
                        MERGE (sc)-[:FEATURES]->(e)
                        """,
                        name=ep_name,
                        etype=ep.get("entity_type", "character"),
                        bi=bi,
                        ci=ci,
                        si=si,
                    )

                # --- Scene-level events ---
                for ev in scene.get("events", []):
                    ev_id = ev.get("event_id")
                    if not ev_id:
                        continue
                    run(
                        s,
                        """
                        MERGE (e:Event {id: $ev_id})
                        SET e.description = $desc,
                            e.event_type  = $etype
                        WITH e
                        MATCH (sc:Scene {book_index: $bi, chapter_index: $ci, scene_index: $si})
                        MERGE (sc)-[:HAS_EVENT]->(e)
                        """,
                        ev_id=ev_id,
                        desc=ev.get("description", ""),
                        etype=ev.get("type", ""),
                        bi=bi,
                        ci=ci,
                        si=si,
                    )
                    # Characters involved in this event
                    for char_name in ev.get("characters", []):
                        run(
                            s,
                            """
                            MERGE (c:Entity {name: $name})
                            WITH c
                            MATCH (e:Event {id: $ev_id})
                            MERGE (c)-[:INVOLVED_IN]->(e)
                            """,
                            name=char_name,
                            ev_id=ev_id,
                        )

                # --- Relationship changes ---
                for rc in scene.get("relationship_changes", []):
                    src = rc.get("source_entity")
                    tgt = rc.get("target_entity")
                    if not src or not tgt:
                        continue
                    run(
                        s,
                        """
                        MATCH  (sc:Scene {book_index: $bi, chapter_index: $ci, scene_index: $si})
                        MERGE  (src:Entity {name: $src})
                        MERGE  (tgt:Entity {name: $tgt})
                        MERGE  (rc:RelationshipChange {
                                    source_entity: $src,
                                    target_entity: $tgt,
                                    book_index:    $bi,
                                    chapter_index: $ci,
                                    scene_index:   $si
                               })
                        SET    rc.relationship = $rel,
                               rc.change       = $change,
                               rc.evidence     = $evidence
                        MERGE  (sc)-[:HAS_RELATIONSHIP_CHANGE]->(rc)
                        MERGE  (rc)-[:CHANGE_SOURCE]->(src)
                        MERGE  (rc)-[:CHANGE_TARGET]->(tgt)
                        """,
                        bi=bi,
                        ci=ci,
                        si=si,
                        src=src,
                        tgt=tgt,
                        rel=rc.get("relationship", ""),
                        change=rc.get("change", ""),
                        evidence=rc.get("evidence", ""),
                    )

            # ------------------------------------------------------------------
            # 7. CAUSAL GRAPH  (events, critical path, causal chains,
            #                   divergence points)
            # ------------------------------------------------------------------
            cg = outputs.get("causal_graph_result", {}).get("graph", {})

            # 7a. Event nodes with full metadata
            cg_events = cg.get("events", [])
            for ev in cg_events:
                ev_id = ev.get("id")
                if not ev_id:
                    continue
                run(
                    s,
                    """
                    MATCH (b:Book {title: $book_title})
                    MERGE (e:Event {id: $ev_id})
                    SET e.description    = $desc,
                        e.event_type     = $etype,
                        e.story_impact   = $impact,
                        e.reversibility  = $rev,
                        e.time_index     = $ti,
                        e.book_index     = $bi,
                        e.chapter_index  = $ci,
                        e.scene_index    = $si,
                        e.source_summary = $summary
                    MERGE (b)-[:HAS_EVENT]->(e)
                    """,
                    book_title=book_title,
                    ev_id=ev_id,
                    desc=ev.get("description", ""),
                    etype=ev.get("event_type", ""),
                    impact=ev.get("story_impact"),
                    rev=ev.get("reversibility"),
                    ti=ev.get("time_index"),
                    bi=ev.get("book_index"),
                    ci=ev.get("chapter_index"),
                    si=ev.get("scene_index"),
                    summary=ev.get("source_summary", ""),
                )

                # Characters involved in this causal event
                for char_name in ev.get("characters", []):
                    run(
                        s,
                        """
                        MERGE (c:Entity {name: $name})
                        WITH c
                        MATCH (e:Event {id: $ev_id})
                        MERGE (c)-[:INVOLVED_IN]->(e)
                        """,
                        name=char_name,
                        ev_id=ev_id,
                    )

                # Causal edges
                for rel_key, neo4j_rel in [
                    ("causes", "CAUSES"),
                    ("caused_by", "CAUSED_BY"),
                    ("prevents", "PREVENTS"),
                    ("required_for", "REQUIRED_FOR"),
                ]:
                    for linked in ev.get(rel_key, []):
                        target_id = linked.get("event_id")
                        if not target_id:
                            continue
                        run(
                            s,
                            f"""
                            MATCH (a:Event {{id: $from_id}})
                            MERGE (b:Event {{id: $to_id}})
                            MERGE (a)-[r:{neo4j_rel}]->(b)
                            SET r.explanation = $explanation
                            """,
                            from_id=ev_id,
                            to_id=target_id,
                            explanation=linked.get("explanation", ""),
                        )

            # 7b. Critical path — mark events + chain them
            critical_path = cg.get("critical_path", [])
            for idx, cp_ev in enumerate(critical_path):
                ev_id = cp_ev.get("event_id")
                if not ev_id:
                    continue
                run(
                    s,
                    """
                    MERGE (e:Event {id: $ev_id})
                    SET e.is_critical        = true,
                        e.why_critical       = $why,
                        e.criticality_score  = $score,
                        e.critical_order     = $order
                    """,
                    ev_id=ev_id,
                    why=cp_ev.get("why_critical", ""),
                    score=cp_ev.get("criticality_score"),
                    order=idx,
                )
                if idx < len(critical_path) - 1:
                    next_id = critical_path[idx + 1].get("event_id")
                    if next_id:
                        run(
                            s,
                            """
                            MATCH (a:Event {id: $curr})
                            MERGE (b:Event {id: $nxt})
                            MERGE (a)-[r:CRITICAL_NEXT]->(b)
                            SET r.sequence = $seq
                            """,
                            curr=ev_id,
                            nxt=next_id,
                            seq=idx,
                        )

            # 7c. Causal chains
            for chain in cg.get("causal_chains", []):
                chain_id = chain.get("chain_id")
                if not chain_id:
                    continue
                run(
                    s,
                    """
                    MERGE (cc:CausalChain {chain_id: $chain_id})
                    SET cc.description     = $desc,
                        cc.chain_type      = $ctype,
                        cc.story_function  = $func
                    """,
                    chain_id=chain_id,
                    desc=chain.get("description", ""),
                    ctype=chain.get("chain_type", ""),
                    func=chain.get("story_function", ""),
                )
                for ev_id in chain.get("event_sequence", []):
                    run(
                        s,
                        """
                        MERGE (e:Event {id: $ev_id})
                        WITH e
                        MATCH (cc:CausalChain {chain_id: $chain_id})
                        MERGE (e)-[:IN_CHAIN]->(cc)
                        """,
                        ev_id=ev_id,
                        chain_id=chain_id,
                    )

            # 7d. Divergence points
            for dp in cg.get("divergence_points", []):
                ev_id = dp.get("event_id")
                if not ev_id:
                    continue
                run(
                    s,
                    """
                    MERGE (d:DivergencePoint {event_id: $ev_id})
                    SET d.decision_made          = $decision,
                        d.divergence_potential   = $potential,
                        d.alternate_timeline     = $alt,
                        d.alternatives           = $alts
                    WITH d
                    MATCH (e:Event {id: $ev_id})
                    MERGE (e)-[:IS_DIVERGENCE_POINT]->(d)
                    """,
                    ev_id=ev_id,
                    decision=dp.get("decision_made", ""),
                    potential=dp.get("divergence_potential"),
                    alt=dp.get("alternate_timeline", ""),
                    alts=dp.get("alternatives", []),
                )

            # ------------------------------------------------------------------
            # 8. GLOBAL TIMELINE  (time_index ordered events)
            # ------------------------------------------------------------------
            timeline = outputs.get("timeline", [])
            for tl in timeline:
                ev_id = tl.get("event_id")
                if not ev_id:
                    continue
                run(
                    s,
                    """
                    MERGE (e:Event {id: $ev_id})
                    SET e.time_index    = coalesce(e.time_index, $ti),
                        e.timeline_summary = $summary
                    """,
                    ev_id=ev_id,
                    ti=tl.get("time_index"),
                    summary=tl.get("summary", ""),
                )
                # Characters referenced in the global timeline entry
                for char_name in tl.get("characters", []):
                    run(
                        s,
                        """
                        MERGE (c:Entity {name: $name})
                        WITH c
                        MATCH (e:Event {id: $ev_id})
                        MERGE (c)-[:INVOLVED_IN]->(e)
                        """,
                        name=char_name,
                        ev_id=ev_id,
                    )

            # ------------------------------------------------------------------
            # 9. CHARACTER TIMELINES  (per-character ordered event list)
            # ------------------------------------------------------------------
            char_timelines = outputs.get("character_timelines", [])
            for ct in char_timelines:
                char_name = ct.get("character")
                if not char_name:
                    continue
                run(s, "MERGE (c:Entity {name: $name})", name=char_name)
                for ct_ev in ct.get("events", []):
                    ev_id = ct_ev.get("event_id")
                    if not ev_id:
                        continue
                    run(
                        s,
                        """
                        MERGE (e:Event {id: $ev_id})
                        SET e.time_index = coalesce(e.time_index, $ti)
                        WITH e
                        MATCH (c:Entity {name: $name})
                        MERGE (c)-[r:APPEARS_IN_EVENT]->(e)
                        SET r.time_index = $ti
                        """,
                        ev_id=ev_id,
                        name=char_name,
                        ti=ct_ev.get("time_index"),
                    )

            # ------------------------------------------------------------------
            # 10. RELATIONSHIP SUMMARY EDGES  (Character <-> Character)
            #
            # Derived from relationship_changes across all scenes.
            # Strategy: walk scenes in chapter order; for each (src, tgt) pair
            # keep updating the HAS_RELATIONSHIP edge so the final write
            # reflects the LATEST known state — giving the decoder a single
            # "current relationship" edge to query without traversing history.
            #
            # The full history is still preserved via RelationshipChange nodes
            # (ingested in section 6), so nothing is lost.
            # ------------------------------------------------------------------
            rel_summary_count = 0

            # Sort scenes by chapter so we overwrite in chronological order
            scenes_sorted = sorted(
                resolved,
                key=lambda sc: (
                    sc.get("book_index", 0),
                    sc.get("chapter_index", 0),
                    sc.get("scene_index", 0),
                ),
            )

            for scene in scenes_sorted:
                bi = scene.get("book_index")
                ci = scene.get("chapter_index")
                si = scene.get("scene_index")

                for rc in scene.get("relationship_changes", []):
                    src = rc.get("source_entity")
                    tgt = rc.get("target_entity")
                    if not src or not tgt:
                        continue

                    # Normalise the relationship label into a Neo4j-safe rel type
                    rel_type = (
                        rc.get("relationship", "RELATED_TO")
                        .upper()
                        .replace(" ", "_")
                        .replace("-", "_")
                        .replace("/", "_")
                    )

                    # MERGE the summary edge, then SET all its properties.
                    # Because we process chronologically, the last SET wins —
                    # that's intentional: it represents the current state.
                    query = f"""
                        MERGE (a:Entity {{name: $src}})
                        MERGE (b:Entity {{name: $tgt}})
                        MERGE (a)-[r:HAS_RELATIONSHIP {{pair: $pair}}]->(b)
                        SET r.type            = $rel_type,
                            r.latest_change   = $change,
                            r.latest_evidence = $evidence,
                            r.last_seen_book  = $bi,
                            r.last_seen_ch    = $ci,
                            r.last_seen_scene = $si
                    """
                    run(
                        s,
                        query,
                        src=src,
                        tgt=tgt,
                        # canonical pair key so (A,B) and (B,A) don't duplicate
                        pair=f"{min(src, tgt)}|{max(src, tgt)}",
                        rel_type=rel_type,
                        change=rc.get("change", ""),
                        evidence=rc.get("evidence", ""),
                        bi=bi,
                        ci=ci,
                        si=si,
                    )
                    rel_summary_count += 1

            # ------------------------------------------------------------------
            # 11. FLEXIBLE EVENTS  (safe-to-modify events for decoder branching)
            # ------------------------------------------------------------------
            flex_events = cg.get("flexible_events", [])
            for flex_ev in flex_events:
                ev_id = flex_ev.get("event_id")
                if not ev_id:
                    continue
                run(
                    s,
                    """
                    MERGE (e:Event {id: $ev_id})
                    SET e.is_flexible       = true,
                        e.flexibility_score = $score,
                        e.why_flexible      = $why
                    """,
                    ev_id=ev_id,
                    score=flex_ev.get("flexibility_score"),
                    why=flex_ev.get("why_flexible", ""),
                )

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        return {
            "status": f"SAGA contract ingested into {DB_NAME}",
            "ingested": {
                "book": book_title,
                "chapters": len(chapters),
                "entities": len(entity_registry),
                "aliases": sum(len(v) for v in alias_map.values()),
                "state_transitions": len(transitions),
                "scenes": len(resolved),
                "causal_events": len(cg_events),
                "critical_path_events": len(critical_path),
                "causal_chains": len(cg.get("causal_chains", [])),
                "divergence_points": len(cg.get("divergence_points", [])),
                "timeline_entries": len(timeline),
                "character_timelines": len(char_timelines),
                "relationship_summary_edges": rel_summary_count,
                "flexible_events": len(flex_events),
            },
        }

    except Exception as e:
        detail = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }
        print("ERROR:", detail)
        raise HTTPException(status_code=500, detail=detail)
