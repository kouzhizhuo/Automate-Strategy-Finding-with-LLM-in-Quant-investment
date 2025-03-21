"""
Main Module for Grail Investment Strategy Framework
Integrates the Seed Alphas Factory, Multi-Agent System, and Weight Optimization components.
"""
from typing import List, Dict, Any
import torch
from .seed_alphas_factory import SeedAlphasFactory
from .multi_agent_system import MultiAgentSystem, AgentConfig, RiskPreference
from .weight_optimization import WeightOptimizer, ModelConfig

class GrailFramework:
    def __init__(self, 
                 llm_model_name: str = "gpt2",
                 agent_configs: List[AgentConfig] = None,
                 model_config: ModelConfig = None):
        """
        Initialize the Grail Framework with all components.
        
        Args:
            llm_model_name (str): Name of the LLM model to use
            agent_configs (List[AgentConfig]): Configurations for the multi-agent system
            model_config (ModelConfig): Configuration for the weight optimization model
        """
        # Initialize components
        self.seed_alphas_factory = SeedAlphasFactory(model_name=llm_model_name)
        
        # Default agent configurations if none provided
        if agent_configs is None:
            agent_configs = [
                AgentConfig(RiskPreference.CONSERVATIVE, 0.7, 0.3),
                AgentConfig(RiskPreference.MODERATE, 0.6, 0.5),
                AgentConfig(RiskPreference.AGGRESSIVE, 0.5, 0.7)
            ]
        self.multi_agent_system = MultiAgentSystem(agent_configs)
        
        # Default model configuration if none provided
        if model_config is None:
            model_config = ModelConfig(
                input_size=100,
                hidden_size=64,
                output_size=10,
                learning_rate=0.001,
                batch_size=32,
                num_epochs=100
            )
        self.weight_optimizer = WeightOptimizer(model_config)
    
    def process_documents(self, documents: List[str]) -> List[Dict[str, Any]]:
        """
        Process documents through the Seed Alphas Factory.
        
        Args:
            documents (List[str]): List of documents to process
            
        Returns:
            List of processed alpha factors
        """
        return self.seed_alphas_factory.batch_process(documents)
    
    def evaluate_alphas(self, alphas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate alpha factors using the multi-agent system.
        
        Args:
            alphas (List[Dict[str, Any]]): List of alpha factors to evaluate
            
        Returns:
            List of evaluation results
        """
        return self.multi_agent_system.evaluate_alphas(alphas)
    
    def optimize_weights(self, 
                        alpha_data: torch.Tensor,
                        train_labels: torch.Tensor,
                        val_data: torch.Tensor = None,
                        val_labels: torch.Tensor = None) -> Dict[str, Any]:
        """
        Optimize weights for alpha factors using the deep learning model.
        
        Args:
            alpha_data (torch.Tensor): Input alpha factor data
            train_labels (torch.Tensor): Training target labels
            val_data (torch.Tensor, optional): Validation input data
            val_labels (torch.Tensor, optional): Validation target labels
            
        Returns:
            Training history and optimized weights
        """
        # Train the model
        history = self.weight_optimizer.train(
            alpha_data, train_labels, val_data, val_labels
        )
        
        # Get optimized weights
        weights = self.weight_optimizer.predict_weights(alpha_data)
        
        return {
            "history": history,
            "weights": weights
        }
    
    def run_pipeline(self, 
                    documents: List[str],
                    alpha_data: torch.Tensor,
                    train_labels: torch.Tensor,
                    val_data: torch.Tensor = None,
                    val_labels: torch.Tensor = None) -> Dict[str, Any]:
        """
        Run the complete Grail pipeline from document processing to weight optimization.
        
        Args:
            documents (List[str]): Input documents
            alpha_data (torch.Tensor): Alpha factor data for weight optimization
            train_labels (torch.Tensor): Training target labels
            val_data (torch.Tensor, optional): Validation input data
            val_labels (torch.Tensor, optional): Validation target labels
            
        Returns:
            Complete pipeline results
        """
        # Step 1: Process documents through Seed Alphas Factory
        processed_alphas = self.process_documents(documents)
        
        # Step 2: Evaluate alphas using multi-agent system
        evaluation_results = self.evaluate_alphas(processed_alphas)
        
        # Step 3: Optimize weights using deep learning
        optimization_results = self.optimize_weights(
            alpha_data, train_labels, val_data, val_labels
        )
        
        return {
            "processed_alphas": processed_alphas,
            "evaluation_results": evaluation_results,
            "optimization_results": optimization_results
        }
