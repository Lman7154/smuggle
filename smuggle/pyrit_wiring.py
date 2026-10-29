"""Single import surface for the exact pip-installed PyRIT (v1.1.0) symbols this
artefact uses. Nothing here is re-implemented; it is all PyRIT."""
from __future__ import annotations

from pyrit.prompt_target import OpenAIChatTarget
from pyrit.executor.attack import (
    PromptSendingAttack,
    RedTeamingAttack,
    AttackAdversarialConfig,
    AttackConverterConfig,
    AttackScoringConfig,
)
from pyrit.score import (
    SelfAskTrueFalseScorer, SelfAskRefusalScorer, TrueFalseQuestion, LlamaGuardScorer,
    TrueFalseCompositeScorer, TrueFalseInverterScorer, TrueFalseScoreAggregator,
)
from pyrit.converter import (
    Converter,
    ConverterResult,
    SelectiveTextConverter,
    WordKeywordSelectionStrategy,
    UnicodeConfusableConverter,
    VariationSelectorSmugglerConverter,
    AsciiSmugglerConverter,
    SneakyBitsSmugglerConverter,
    UnicodeSubstitutionConverter,
    UnicodeReplacementConverter,
    ZeroWidthConverter,
    BidiConverter,
    DiacriticConverter,
    ZalgoConverter,
    SuperscriptConverter,
    BrailleConverter,
    Base2048Converter,
    EcojiConverter,
    EmojiConverter,
)
from pyrit.prompt_normalizer import ConverterConfiguration
from pyrit.memory import CentralMemory, SQLiteMemory
from pyrit.models import SeedPrompt

try:
    from pyrit.setup import IN_MEMORY, SQLITE, initialize_pyrit_async
except Exception:  # pragma: no cover
    IN_MEMORY = None
    SQLITE = None
    initialize_pyrit_async = None

__all__ = [
    "OpenAIChatTarget",
    "PromptSendingAttack", "RedTeamingAttack",
    "AttackAdversarialConfig", "AttackConverterConfig", "AttackScoringConfig",
    "SelfAskTrueFalseScorer", "SelfAskRefusalScorer", "TrueFalseQuestion", "LlamaGuardScorer",
    "TrueFalseCompositeScorer", "TrueFalseInverterScorer", "TrueFalseScoreAggregator",
    "Converter", "ConverterResult", "SelectiveTextConverter",
    "WordKeywordSelectionStrategy",
    "UnicodeConfusableConverter", "VariationSelectorSmugglerConverter",
    "AsciiSmugglerConverter", "SneakyBitsSmugglerConverter", "UnicodeSubstitutionConverter",
    "UnicodeReplacementConverter", "ZeroWidthConverter", "BidiConverter",
    "DiacriticConverter", "ZalgoConverter", "SuperscriptConverter", "BrailleConverter",
    "Base2048Converter", "EcojiConverter", "EmojiConverter",
    "ConverterConfiguration",
    "CentralMemory", "SQLiteMemory", "SeedPrompt", "IN_MEMORY", "SQLITE", "initialize_pyrit_async",
]
