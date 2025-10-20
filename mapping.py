from typing import List


class MappingOutput:

    def __init__(self, id, title: str, ingredients_vector: List, all_vector: List, image: str, ingredients: List, steps: List[dict], final_vector=None):
        self.id = id
        self.title = title
        self.image = image
        self.ingredients = ingredients
        self.steps = steps
        self.ingredients_vector = ingredients_vector
        self.all_vector = all_vector
        self.final_vector = final_vector

    def set_final_vector(self, vector):
        self.final_vector = vector

    def __str__(self, with_vector=False):
        if not with_vector:
            return f"MappingOutput(title={self.title}, image={self.image}, ingredients={self.ingredients}, steps={self.steps})"
        else:
            return f"MappingOutput(title={self.title}, image={self.image}, ingredients={self.ingredients}, steps={self.steps}, ingredients_vector={self.ingredients_vector}, all_vector={self.all_vector}, final_vector={self.final_vector})"
