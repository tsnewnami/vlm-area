from typing import Any, Tuple
from abc import ABC, abstractmethod
from shape_generator import generate_shape_plot
class DataLoader(ABC):
    """
    Abstract base class for data loading.

    
    Defines the core interface that must be implemented by all concrete data loaders.
    Child classes should extend this base class and provide implementations for
    all abstract methods.
    
    Attributes:
        current_index (int): Tracks position within dataset for sequential iteration
    """
    
    def __init__(self, random: bool = False) -> None:
        self.current_index = 0
        
    @abstractmethod
    def __len__(self) -> int:
        """Returns the total count of items contained in the dataset."""
        pass
        
    @abstractmethod
    def __iter__(self) -> 'DataLoader':
        """Makes this class iterable by returning itself."""
        return self
        
    @abstractmethod
    def __next__(self) -> Any:
        """Retrieves the next item(s) when iterating through the dataset."""
        pass

    @abstractmethod
    def reset(self):
        """Restarts iteration from the beginning of the dataset."""
        pass

    
SINGLE_AREA_PROMPT = """
You will be shown an image of a graph with a shape drawn on it. Your task is to determine the area of the shape.

You must answer in the following format:
<reasoning>
Briefly reason about the shape using axis labels or grid lines. Use a geometric formula to calculate the area in a few clear steps. If needed, estimate based on grid coverage.
Keep your reasoning concise. Do not explain more than necessary.
</reasoning>
<answer>
X.XX
</answer>

Replace X.XX with the area of the shape, rounded to 2 decimal places and only include numbers in your answer. Do not include any other text in your answer or any other text after </answer>.

What is the area of the shape drawn on the plot?
"""

class SingleShapeAreaDataLoader(DataLoader):
    def __init__(self, dataset_size: int = 50, is_train: bool = True) -> None:
        super().__init__() # Always generates random times
        self.dataset_size = dataset_size
        self.is_train = is_train
        self.prompt = SINGLE_AREA_PROMPT
        self.filename = "temp_shape.png" # Fixed path for the temporary image
        
    def __len__(self) -> int:
        # Return the specified size, mainly relevant for the test set iteration count
        return self.dataset_size
        
    def __iter__(self) -> 'SingleShapeAreaDataLoader':
        self.current_index = 0
        return self
        
    def __next__(self) -> Tuple[str, str]:
        # Stop iteration for the test set after reaching dataset_size
        if not self.is_train and self.current_index >= self.dataset_size:
            raise StopIteration
        
        self.current_index += 1

        (filepath, area) = generate_shape_plot()        

        return (filepath, area)

    def reset(self):
        self.current_index = 0 

if __name__ == "__main__":
    # Single Shape Area dataset test
    try:
        print("\n" + "="*50)
        print("== SINGLE SHAPE AREA DATASET TEST ==".center(50))
        print("="*50 + "\n")
        
        # Testing training dataset
        print("-"*50)
        print("TRAINING DATASET CONFIGURATION:")
        print("-"*50)
        train_loader = SingleShapeAreaDataLoader(dataset_size=10, is_train=True)
        print(f"• Dataset Size: {len(train_loader)} examples")
        
        # Generate and display sample from training dataset
        print("\nGENERATING SAMPLE FROM TRAINING DATASET:")
        filepath, area = next(train_loader)
        print(f"• Generated Image: {filepath}")
        print(f"• Shape Area: {area} square pixels")
        
        # Testing test dataset
        print("\n" + "-"*50)
        print("TEST DATASET CONFIGURATION:")
        print("-"*50)
        test_loader = SingleShapeAreaDataLoader(dataset_size=10, is_train=False)
        
        # Generate and display sample from test dataset
        print("\nGENERATING SAMPLE FROM TEST DATASET:")
        filepath, area = next(test_loader)
        print(f"• Generated Image: {filepath}")
        print(f"• Shape Area: {area} square pixels")
        
        # Display system prompt used for the task
        print("\n" + "-"*50)
        print("SYSTEM PROMPT:")
        print("-"*50)
        print(f"{train_loader.prompt.strip()}")
        print("\n" + "="*50)
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load single shape area dataset: {e}")