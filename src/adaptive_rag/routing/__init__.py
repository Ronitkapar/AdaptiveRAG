"""
routing package initialization.
"""

from adaptive_rag.routing.analyzer import ANALYZER_VERSION, QueryFeatureAnalyzer
from adaptive_rag.routing.base import QueryAnalyzer, Router
from adaptive_rag.routing.escalation import POLICY_VERSION, EscalationPolicy
from adaptive_rag.routing.rule_based import QUESTION_TYPE_SUPPORT, ROUTER_VERSION, RuleBasedRouter
from adaptive_rag.routing.sufficiency import CHECKER_VERSION, SufficiencyChecker

__all__ = [
    "ANALYZER_VERSION",
    "CHECKER_VERSION",
    "POLICY_VERSION",
    "QUESTION_TYPE_SUPPORT",
    "ROUTER_VERSION",
    "EscalationPolicy",
    "QueryAnalyzer",
    "QueryFeatureAnalyzer",
    "Router",
    "RuleBasedRouter",
    "SufficiencyChecker",
]