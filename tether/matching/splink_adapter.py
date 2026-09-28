"""Every call into Splink lives here, so a Splink version change is a one-file change.

Responsibilities:

* build a ``SettingsCreator`` from a profile (comparisons with graded levels, blocking rules)
* register frames with a backend (DuckDB for the pilot; Spark via the same ``db_api`` seam)
* deterministic pre-pass with guards rendered through Splink's dialect layer
* u estimation by random sampling, m by EM; optional m starting values from the knowledge base
* prediction, clustering of an edge table, model export
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from splink import DuckDBAPI, Linker, SettingsCreator, block_on
from splink import blocking_rule_library as brl
from splink import comparison_level_library as cll
from splink.internals.comparison_creator import ComparisonCreator
from splink.internals.database_api import DatabaseAPI

from tether.config import MatchingConfig
from tether.fields.base import FieldType
from tether.profiles.base import BlockingRule, DeterministicRule, EntityProfile, FieldSpec, Guard

MProbabilities = Mapping[str, Mapping[str, float]]
"""``{comparison output column: {level label: m}}`` as stored in the knowledge base."""


# ------------------------------------------------------------------ backends
def make_db_api(backend: str = "duckdb", **kwargs: Any) -> DatabaseAPI:
    """Instantiate a Splink backend. ``spark`` requires a SparkSession in ``kwargs``."""
    if backend == "duckdb":
        return DuckDBAPI(**kwargs)
    if backend == "spark":  # pragma: no cover - not exercised in the pilot test suite
        from splink import SparkAPI

        return SparkAPI(**kwargs)
    raise ValueError(f"unknown backend '{backend}'")


# ------------------------------------------------------------------ settings
def blocking_rule_creator(rule: BlockingRule | Sequence[str] | str):
    """Translate a profile/config blocking rule into a Splink BlockingRuleCreator."""
    if isinstance(rule, BlockingRule):
        return block_on(*rule.columns) if rule.columns else brl.CustomRule(rule.sql)
    if isinstance(rule, str):
        return brl.CustomRule(rule)
    return block_on(*rule)


def build_comparisons(
    field_specs: Mapping[str, FieldSpec],
    field_types: Mapping[str, FieldType],
    geo_available: bool,
    m_probabilities: MProbabilities | None = None,
) -> list[ComparisonCreator]:
    """Comparisons for every model field, optionally seeded with m starting values."""
    comparisons: list[ComparisonCreator] = []
    for fname, spec in field_specs.items():
        if spec.role == "report_only":
            continue
        opts = spec.comparison_options(geo_available=geo_available)
        for comp in field_types[fname].comparisons(fname, opts):
            comparisons.append(comp)
    if m_probabilities:
        apply_m_probabilities(comparisons, m_probabilities)
    return comparisons


def apply_m_probabilities(comparisons: Sequence[ComparisonCreator], m_probabilities: MProbabilities) -> int:
    """Seed comparison levels with m starting values keyed by (output column, level label).

    Only levels present in both the model and ``m_probabilities`` are touched; u is never
    seeded (it is population-specific and cheap to estimate). Returns the count applied.
    """
    applied = 0
    for comp in comparisons:
        name = comp.create_output_column_name()
        levels_m = m_probabilities.get(name)
        if not levels_m:
            continue
        for level in comp.get_configured_comparison_levels():
            label = level.get_comparison_level("duckdb").label_for_charts
            if label in levels_m:
                level.configure(m_probability=float(levels_m[label]))
                applied += 1
    return applied


def build_settings(
    profile: EntityProfile,
    field_specs: Mapping[str, FieldSpec],
    field_types: Mapping[str, FieldType],
    matching: MatchingConfig,
    geo_available: bool,
    m_probabilities: MProbabilities | None = None,
    retain_columns: Sequence[str] = (),
) -> SettingsCreator:
    """Assemble Splink settings from the profile plus engagement overrides."""
    rules = matching.blocking_rules if matching.blocking_rules is not None else profile.blocking_rules
    settings = SettingsCreator(
        link_type=matching.link_type,
        comparisons=build_comparisons(field_specs, field_types, geo_available, m_probabilities),
        blocking_rules_to_generate_predictions=[blocking_rule_creator(r) for r in rules],
        probability_two_random_records_match=matching.probability_two_random_records_match or 1e-4,
        max_iterations=matching.em_max_iterations,
        retain_matching_columns=True,
        retain_intermediate_calculation_columns=True,
        additional_columns_to_retain=list(retain_columns),
        unique_id_column_name="unique_id",
        source_dataset_column_name="source_dataset",
    )
    return settings


# ------------------------------------------------------- deterministic rules
def _lr(sql: str, *columns: str) -> str:
    """Rewrite Splink's ``col_l`` / ``"col_l"`` aliases to ``l.col`` / ``r.col`` for join predicates."""
    for col in columns:
        for side in ("l", "r"):
            sql = sql.replace(f'"{col}_{side}"', f'{side}."{col}"')
            sql = re.sub(rf"\b{re.escape(col)}_{side}\b", f"{side}.{col}", sql)
    return sql


def render_guard(guard: Guard, dialect: str) -> str:
    """Render a Guard as join-predicate SQL, using Splink's own level generators for fuzzy parts."""
    col = guard.column
    parts = [f"l.{col} IS NULL", f"r.{col} IS NULL", f"l.{col} = r.{col}"]
    if guard.jw_threshold is not None:
        jw = cll.JaroWinklerLevel(col, guard.jw_threshold).get_comparison_level(dialect).sql_condition
        parts.append(_lr(jw, col))
    if guard.swap_with:
        parts.append(f"l.{col} = r.{guard.swap_with}")
        parts.append(f"l.{guard.swap_with} = r.{col}")
    return "(" + " OR ".join(parts) + ")"


def deterministic_rule_sql(rule: DeterministicRule, dialect: str) -> str:
    """Full pairwise SQL for a deterministic rule (equality on match columns + guards)."""
    eq = [f"l.{c} = r.{c}" for c in rule.match_columns]
    return " AND ".join(eq + [render_guard(g, dialect) for g in rule.guards])


# --------------------------------------------------------------- the linker
@dataclass
class TrainedModel:
    """What a matching run produces for reuse and audit."""

    settings_dict: dict[str, Any]
    m_probabilities: dict[str, dict[str, float]]
    u_probabilities: dict[str, dict[str, float]]
    training_log: list[dict[str, Any]] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.settings_dict, indent=2, default=str), encoding="utf-8")


def _extract_mu(settings_dict: dict[str, Any]) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    m: dict[str, dict[str, float]] = {}
    u: dict[str, dict[str, float]] = {}
    for comp in settings_dict.get("comparisons", []):
        name = comp["output_column_name"]
        m[name], u[name] = {}, {}
        for level in comp["comparison_levels"]:
            label = level.get("label_for_charts") or level["sql_condition"]
            if "m_probability" in level:
                m[name][label] = level["m_probability"]
            if "u_probability" in level:
                u[name][label] = level["u_probability"]
    return m, u


class SplinkMatcher:
    """Thin orchestration wrapper over a Splink ``Linker``."""

    def __init__(
        self,
        frames: Sequence[pd.DataFrame],
        source_names: Sequence[str],
        settings: SettingsCreator,
        db_api: DatabaseAPI | None = None,
    ):
        self.db_api = db_api or DuckDBAPI()
        self.dialect = self.db_api.sql_dialect.sql_dialect_str
        self.source_names = list(source_names)
        self.registered = [
            self.db_api.register(df, dataset_display_name=name) for df, name in zip(frames, self.source_names)
        ]
        self.settings = settings
        self.linker = Linker(self.registered if len(self.registered) > 1 else self.registered[0], settings)
        self.training_log: list[dict[str, Any]] = []

    # ---- deterministic pre-pass
    def deterministic_pairs(self, rules: Sequence[DeterministicRule]) -> pd.DataFrame:
        """Pairs linked by deterministic rules. Columns: unique_id_l, unique_id_r, rule."""
        if not rules:
            return pd.DataFrame(columns=["unique_id_l", "unique_id_r", "rule"])
        frames = []
        base = self.settings.create_settings_dict(self.dialect)
        for rule in rules:
            sql = deterministic_rule_sql(rule, self.dialect)
            det_settings = dict(base)
            det_settings["blocking_rules_to_generate_predictions"] = [sql]
            det_settings["comparisons"] = []
            det_linker = Linker(self.registered if len(self.registered) > 1 else self.registered[0],
                                det_settings, validate_settings=False)
            pairs = det_linker.inference.deterministic_link().as_pandas_dataframe()
            pairs = pairs[["unique_id_l", "unique_id_r"]].copy()
            pairs["rule"] = rule.description or ",".join(rule.match_columns)
            frames.append(pairs)
        out = pd.concat(frames, ignore_index=True)
        return out.drop_duplicates(subset=["unique_id_l", "unique_id_r"]).reset_index(drop=True)

    # ---- training
    def estimate_probability_two_random_records_match(self, rules: Sequence[DeterministicRule], recall: float) -> None:
        """Set lambda from the deterministic rules and an assumed recall for them."""
        sqls = [deterministic_rule_sql(r, self.dialect) for r in rules]
        self.linker.training.estimate_probability_two_random_records_match(sqls, recall=recall)
        self.training_log.append({"step": "lambda", "rules": sqls, "assumed_recall": recall})

    def estimate_u(self, max_pairs: float, seed: int | None = None) -> None:
        self.linker.training.estimate_u_using_random_sampling(max_pairs=max_pairs, seed=seed)
        self.training_log.append({"step": "u_random_sampling", "max_pairs": max_pairs})

    def estimate_m(self, training_rules: Sequence[BlockingRule | DeterministicRule | Sequence[str] | str]) -> None:
        """One EM round per training rule (each round fixes the comparisons it blocks on)."""
        for rule in training_rules:
            if isinstance(rule, DeterministicRule):
                creator = brl.CustomRule(deterministic_rule_sql(rule, self.dialect))
            else:
                creator = blocking_rule_creator(rule)
            session = self.linker.training.estimate_parameters_using_expectation_maximisation(creator)
            desc = getattr(rule, "description", None) or str(rule)
            history = getattr(session, "_iteration_history_records", None) or []
            self.training_log.append({"step": "em", "blocking": desc, "iterations": len(history)})

    def apply_probability_floor(self, floor: float) -> int:
        """Clamp trained m/u to ``[floor, 1 - floor]`` and rebuild the linker from the result.

        Guards against a level that never varied in training (m or u of exactly 0 or 1),
        which would otherwise contribute an unbounded match weight. Returns count changed.
        """
        if floor <= 0:
            return 0
        settings_dict = self.linker.misc.save_model_to_json()
        changed = 0
        for comp in settings_dict.get("comparisons", []):
            for level in comp["comparison_levels"]:
                for key in ("m_probability", "u_probability"):
                    if key in level and level[key] is not None:
                        clamped = min(max(float(level[key]), floor), 1 - floor)
                        if clamped != level[key]:
                            level[key], changed = clamped, changed + 1
        if changed:
            self.linker = Linker(self.registered if len(self.registered) > 1 else self.registered[0], settings_dict)
            self.training_log.append({"step": "probability_floor", "floor": floor, "values_clamped": changed})
        return changed

    # ---- inference
    def predict(self, threshold_match_probability: float) -> pd.DataFrame:
        """Scored candidate pairs at/above the threshold, as pandas."""
        df = self.linker.inference.predict(threshold_match_probability=threshold_match_probability)
        self._last_predictions = df
        return df.as_pandas_dataframe()

    def cluster_edges(self, edges: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """Connected components over an edge table via Splink's clustering (backend-scalable).

        ``edges`` needs ``unique_id_l``, ``unique_id_r``, ``match_probability``. Returns
        ``unique_id``, ``cluster_id`` for every input record (singletons included).
        """
        cols = ["unique_id_l", "unique_id_r", "match_probability"]
        e = edges[cols].copy() if len(edges) else pd.DataFrame(columns=cols)
        # Splink derives node ids from source_dataset + unique_id for multi-source runs.
        e["source_dataset_l"] = e["unique_id_l"].str.split(":", n=1).str[0]
        e["source_dataset_r"] = e["unique_id_r"].str.split(":", n=1).str[0]
        sdf = self.db_api.register(e)
        clustered = self.linker.clustering.cluster_pairwise_predictions_at_threshold(
            sdf, threshold_match_probability=threshold
        )
        out = clustered.as_pandas_dataframe()
        return out[["cluster_id", "unique_id"]].rename(columns={"cluster_id": "cluster_id"})

    # ---- export
    def trained_model(self) -> TrainedModel:
        settings_dict = self.linker.misc.save_model_to_json()
        m, u = _extract_mu(settings_dict)
        return TrainedModel(settings_dict=settings_dict, m_probabilities=m, u_probabilities=u,
                            training_log=list(self.training_log))

    def match_weights_chart(self):
        return self.linker.visualisations.match_weights_chart()

    def waterfall_chart(self, records: list[dict[str, Any]]):
        return self.linker.visualisations.waterfall_chart(records, filter_nulls=False)
