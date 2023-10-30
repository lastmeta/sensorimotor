# https://chat.openai.com/share/5a5e7acd-48c5-4c9a-82cb-cb682ffca54d

from typing import Callable, Tuple


class Model:
    def __init__(self, function: Callable[[float], float],):
        self.function = function

    def predict(self, x: float) -> float:
        return self.function(x)


class BoundaryParams:
    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper


class Observation:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


def update_model(existing_model: Model, boundary_params: BoundaryParams, new_observation: Observation) -> Tuple[Model, BoundaryParams]:
    """
    existing_model: Model, represents the current model for making predictions
    boundary_params: BoundaryParams, parameters describing the boundaries for the existing model
    new_observation: Observation, new data point (x, y)

    Returns:
    new_model: Model, updated model that includes the new observation
    new_boundary_params: BoundaryParams, updated boundary parameters
    """

    # 1. Analyze the new observation and existing model to compute adjustments
    #    - This is where the 'averaging' or other methodology would take place

    # 2. Update the boundary parameters based on the new observation
    #    - This would involve computing new boundaries that respect both the new and old data

    # 3. Create a new model function that respects these boundaries
    #    - This new function is constrained to stay within the established boundaries

    # 4. Return the new model and new boundary parameters

    return new_model, new_boundary_params

# Example usage:


# Define initial function, boundaries and model
def initial_function(x): return x * 2


initial_boundaries = BoundaryParams(lower=0, upper=10)
initial_model = Model(initial_function)

# New observation
new_data_point = Observation(x=3, y=7)

# Update model
updated_model, updated_boundaries = update_model(
    initial_model, initial_boundaries, new_data_point)
