"""
Multi-Agent Decision Making System
Handles the evaluation and selection of alpha factors using multiple agents with different risk preferences.
"""
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum

class RiskPreference(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

@dataclass
class AgentConfig:
    risk_preference: RiskPreference
    confidence_threshold: float
    max_exposure: float

class AlphaAgent:
    def __init__(self, config: AgentConfig):
        """
        Initialize an agent with specific risk preferences and evaluation criteria.
        
        Args:
            config (AgentConfig): Configuration for the agent's behavior
        """
        self.config = config
        self.risk_multiplier = self._get_risk_multiplier()
        
    def _get_risk_multiplier(self) -> float:
        """Get risk multiplier based on risk preference."""
        multipliers = {
            RiskPreference.CONSERVATIVE: 0.5,
            RiskPreference.MODERATE: 1.0,
            RiskPreference.AGGRESSIVE: 2.0
        }
        return multipliers[self.config.risk_preference]
    
    def evaluate_alpha(self, alpha_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single alpha factor based on agent's preferences.
        
        Args:
            alpha_data (Dict[str, Any]): Alpha factor data to evaluate
            
        Returns:
            Dict containing evaluation results and confidence score
        """
        confidence = self._calculate_confidence(alpha_data)
        exposure = self._calculate_exposure(alpha_data, confidence)
        
        return {
            "confidence": confidence,
            "exposure": exposure,
            "risk_preference": self.config.risk_preference.value,
            "selected": confidence >= self.config.confidence_threshold
        }
    
    def _calculate_confidence(self, alpha_data: Dict[str, Any]) -> float:
        """Calculate confidence score for the alpha factor."""
        base_confidence = alpha_data.get("confidence", 0.0)
        return min(1.0, base_confidence * self.risk_multiplier)
    
    def _calculate_exposure(self, alpha_data: Dict[str, Any], confidence: float) -> float:
        """Calculate exposure level based on confidence and risk preference."""
        return min(self.config.max_exposure, confidence * self.risk_multiplier)

class MultiAgentSystem:
    def __init__(self, agent_configs: List[AgentConfig]):
        """
        Initialize the multi-agent system with multiple agents.
        
        Args:
            agent_configs (List[AgentConfig]): List of agent configurations
        """
        self.agents = [AlphaAgent(config) for config in agent_configs]
    
    def evaluate_alphas(self, alphas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple alpha factors using all agents.
        
        Args:
            alphas (List[Dict[str, Any]]): List of alpha factors to evaluate
            
        Returns:
            List of evaluation results from all agents
        """
        results = []
        for alpha in alphas:
            agent_evaluations = []
            for agent in self.agents:
                evaluation = agent.evaluate_alpha(alpha)
                agent_evaluations.append(evaluation)
            
            # Aggregate results from all agents
            aggregated_result = self._aggregate_evaluations(agent_evaluations)
            results.append(aggregated_result)
        
        return results
    
    def _aggregate_evaluations(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate evaluations from multiple agents."""
        confidences = [eval["confidence"] for eval in evaluations]
        exposures = [eval["exposure"] for eval in evaluations]
        
        return {
            "mean_confidence": np.mean(confidences),
            "std_confidence": np.std(confidences),
            "mean_exposure": np.mean(exposures),
            "std_exposure": np.std(exposures),
            "agent_evaluations": evaluations
        }
