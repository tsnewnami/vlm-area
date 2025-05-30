from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
import re
import math  # Add math import for logarithmic scaling

import torch

# Maximum realistic area difference based on shape generation parameters
# Largest shape is 80x80 square (6400), smallest is ~700, so maximum error around 6000
MAX_DIFF = 6400  # Largest possible shape area


class RewardEvaluator(ABC):
    """
    Base abstract class for calculating rewards during RL training.
    
    This class serves as an interface for reward calculators that evaluate
    model outputs during reinforcement learning. Subclass this to create
    custom reward functions for specific tasks.
    
    Key methods to implement:
    - compute_rewards: Evaluates and scores a batch of model outputs
    - get_reward_breakdown: Transforms raw scores into a structured dictionary
    """
    
    @abstractmethod
    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any
    ) -> Tuple[List[List[float]], Dict[str, float]]:
        """
        Evaluate and score a batch of model outputs.
        
        Args:
            prompts: List of input messages in conversation format
                    [{"role": "user", "content": "..."}, ...]
            completions: List of model-generated responses in conversation format
                        [{"role": "assistant", "content": "..."}, ...]
            answer: Expected or reference solution(s)
            
        Returns:
            rewards_per_func: List of lists with shape (num_completions, num_reward_functions)
                            containing scores from each reward function
            metrics: Dictionary with aggregated statistics including average scores
                    per function and total reward
        """
        pass

    @abstractmethod
    def get_reward_breakdown(self, reward_scores: List[List[float]]) -> Dict[str, float]:
        """
        Transform raw score lists into a named dictionary.
        
        Args:
            reward_scores: List of lists containing scores from compute_rewards
            
        Returns:
            Dictionary with reward function names mapped to their corresponding scores
        """
        pass


class SingleShapeEvaluator(RewardEvaluator):
    """
    Evaluator for the single shape area calculation task.
    
    Implements reward functions for:
        - Area correctness (based on absolute difference)
        - XX.XX format correctness (specifically 2 decimal points)
        - Strict XML formatting (<reasoning>/<answer> tags)
    """
    
    def __init__(self):
        """
        Initialize the SingleShapeEvaluator.
        """
        # Area, decimal point, and XML formatting
        self.num_reward_functions = 3
        
        # Regex pattern to extract area from the model's response (expects X.XX format)
        self.area_extract_pattern = re.compile(r'<answer>\s*(\d+\.\d{2})\s*</answer>', re.DOTALL)
        
        # Regex pattern to check strict XML formatting
        # Requires <reasoning>...</reasoning> followed by <answer>X.XX</answer>
        self.xml_format_pattern = re.compile(
            r'<reasoning>[\s\S]*?</reasoning>\s*<answer>\s*\d+(?:\.\d+)?\s*</answer>', 
            re.DOTALL
        )
        
        # Pattern to check specifically for 2 decimal point format (X.XX) within the answer tags
        self.two_decimal_pattern = re.compile(r'<answer>\s*\d+\.\d{2}\s*</answer>', re.DOTALL)
        
    def _extract_area_string(self, response_text: str) -> Optional[str]:
        """
        Extract area value from model response.
        
        Args:
            response_text: The model's response text to analyze
            
        Returns:
            Extracted area string or None if not found
        """
        # Extract area value if present
        area_match = self.area_extract_pattern.search(response_text)
        if area_match:
            area_str = area_match.group(1)
            
            # Additional validation that area is a valid number
            try:
                float(area_str)
                return area_str
            except ValueError:
                return None
        
        return None
        
    def _area_format_reward(self, completions: List[str]) -> List[float]:
        """
        Award points for correctly formatting area with 2 decimal places.
        
        Args:
            completions: List of model-generated response texts
            
        Returns:
            List with 0.5 for correctly formatted responses, 0.0 otherwise
        """
        return [0.5 if self.two_decimal_pattern.search(completion) else 0.0 for completion in completions]

    def _xml_format_reward(self, completions: List[str]) -> List[float]:
        """
        Award points for strictly following the required XML structure.
        
        Args:
            completions: List of model-generated response texts
            
        Returns:
            List with 0.5 for responses with correct XML formatting, 0.0 otherwise
        """
        return [0.5 if self.xml_format_pattern.search(completion) else 0.0 for completion in completions]

    def _area_correctness_reward(self, completions: List[float | None], answer: Any) -> Tuple[List[float], List[float]]:
        """
        Award points for correctly calculating the area.
        
        Args:
            completions: List of extracted area values as floats or None
            answer: Ground truth area values
        Returns:
            Tuple of (rewards list, absolute errors list)
        """
        rewards = []
        abs_errors = []
        max_reward = 2  # Changed
        min_reward = 0  # Changed

        for completion in completions:
            # Invalid prediction, assign min_reward
            if completion is None:
                rewards.append(min_reward) 
                abs_errors.append(MAX_DIFF)
            else:
                diff = abs(completion - answer)
                abs_errors.append(diff)

                # Linear scaling
                # Ensure diff doesn't exceed MAX_DIFF for calculation, though it shouldn't.
                # Reward is max_reward if diff is 0, min_reward if diff is MAX_DIFF.
                # reward = max_reward * (1 - (diff / MAX_DIFF))
                # Clamp reward to be within [min_reward, max_reward]
                # The formula naturally does this if 0 <= diff <= MAX_DIFF
                
                # Ensure diff is not greater than MAX_DIFF for the ratio calculation
                # to prevent negative rewards if diff somehow exceeded MAX_DIFF.
                normalized_diff = min(diff, MAX_DIFF) / MAX_DIFF
                reward = max_reward * (1 - normalized_diff)
                
                # Explicitly clamp to handle any edge cases, though theoretically covered.
                reward = max(min_reward, reward)

                rewards.append(reward)

        return (rewards, abs_errors)
        
    def compute_rewards(
        self,
        completions: List[str],
        answer: float
    ) -> Tuple[List[List[float]], Dict[str, float]]:
        """
        Calculate rewards for shape area predictions.
        
        Args:
            completions: List of parsed responses with "reasoning" and "answer" keys
            answer: Ground truth area value (single float)
            device: Optional torch device to use for tensor operations
            
        Returns:
            rewards_per_func: List of lists containing reward scores for each function
            metrics: Dictionary with aggregated metrics
        """
        num_completions = len(completions)
        rewards_per_func = [[0.0 for _ in range(self.num_reward_functions)] for _ in range(num_completions)]
        
        # Extract area values from the completions
        extracted_areas = []
        for completion_text in completions:
            area_str = self._extract_area_string(completion_text)
            if area_str is not None:
                try:
                    extracted_areas.append(float(area_str))
                except ValueError:
                    extracted_areas.append(None)
            else:
                extracted_areas.append(None)
        
        # Calculate rewards
        area_format_rewards = self._area_format_reward(completions)
        xml_format_rewards = self._xml_format_reward(completions)
        area_correctness_rewards, abs_errors = self._area_correctness_reward(extracted_areas, answer)
        
        # Combine rewards into the list structure
        for i in range(num_completions):
            rewards_per_func[i][0] = area_correctness_rewards[i]
            rewards_per_func[i][1] = area_format_rewards[i]
            rewards_per_func[i][2] = xml_format_rewards[i]
        
        # Calculate aggregate metrics
        total_rewards = [sum(rewards) for rewards in rewards_per_func]
        mean_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
        mean_area_correctness = sum(area_correctness_rewards) / len(area_correctness_rewards) if area_correctness_rewards else 0.0
        mean_area_format = sum(area_format_rewards) / len(area_format_rewards) if area_format_rewards else 0.0
        mean_xml_format = sum(xml_format_rewards) / len(xml_format_rewards) if xml_format_rewards else 0.0
        
        # Return rewards and metrics
        metrics = {
            "mean_reward": mean_reward,
            "mean_area_correctness": mean_area_correctness,
            "mean_area_format": mean_area_format,
            "mean_xml_format": mean_xml_format,
            "mean_abs_error": sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
        }
        
        return rewards_per_func, metrics
        
    def get_reward_breakdown(self, reward_scores: List[List[float]]) -> Dict[str, float]:
        """
        Convert reward scores into a labeled dictionary.
        
        Args:
            reward_scores: List of lists with calculated scores
            
        Returns:
            Dictionary with named reward components
        """
        if not reward_scores:
            return {
                "area_correctness": 0.0,
                "area_format": 0.0,
                "xml_format": 0.0,
                "total_reward": 0.0
            }
        
        # Calculate averages across all examples
        area_correctness_scores = [scores[0] for scores in reward_scores]
        area_format_scores = [scores[1] for scores in reward_scores]
        xml_format_scores = [scores[2] for scores in reward_scores]
        
        # Calculate total reward
        total_rewards = [sum(scores) for scores in reward_scores]
        
        return {
            "area_correctness": sum(area_correctness_scores) / len(area_correctness_scores),
            "area_format": sum(area_format_scores) / len(area_format_scores), 
            "xml_format": sum(xml_format_scores) / len(xml_format_scores),
            "total_reward": sum(total_rewards) / len(total_rewards)
        }

if __name__ == "__main__":
    
    # Create an instance of the evaluator
    evaluator = SingleShapeEvaluator()
    
    # Example outputs
    completions = ['<reasoning>\nThe shape appears to be a right triangle with its base lying on the horizontal axis and its height along the vertical axis. To find the area of this triangle, we can use the formula for the area of a triangle: \\( \\text{Area} = \\frac{1}{2} \\times \\text{base} \\times \\text{height} \\).\n\nFrom the graph, the base of the triangle is 20 units (from 150 to 170 on the x-axis), and the height is 40 units (from 0 to 40 on the y-axis). Plugging these values into the formula gives:\n\n\\[ \\text{Area} = \\frac{1}{2} \\times 20 \\times 40 = 400 \\]\n\n</reasoning>\n<answer>\n400.00\n</answer>'] 
    answer = 500   # Ground truth area
    
    # Compute rewards
    rewards, metrics = evaluator.compute_rewards(completions, answer)
    
    # Print results
    print("Rewards structure:", [len(reward) for reward in rewards])
    print("Rewards:", rewards)
    print("Metrics:", metrics)
    
    # Test reward breakdown
    reward_breakdown = evaluator.get_reward_breakdown(rewards)
    print("\nReward breakdown:")
    for key, value in reward_breakdown.items():
        print(f"  {key}: {value:.2f}")