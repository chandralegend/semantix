"""Nutrition, Ingridient Estimator using MultiModal LLMs."""

from dataclasses import dataclass
from typing import List

from semantix import Semantic
from semantix.llms import OpenAI
from semantix.types import Image

llm = OpenAI(verbose=True)


@dataclass
class NutritionInformation:
    calories: int
    protein: int
    carbohydrates: int
    fats: int
    fiber: int
    sodium: int


@dataclass
class FoodAnalysis:
    nutrition_info: NutritionInformation
    ingredients: List[str]
    health_rating: Semantic[str, "How Healthy is the Food"]  # type: ignore


NutritionInformation.__doc__ = ""
FoodAnalysis.__doc__ = ""


@llm.enhance("Analyze the given Food Image", method="CoT")
def analyze(img: Image) -> FoodAnalysis: ...


if __name__ == "__main__":
    analysis = analyze(img=Image("examples/ramen.jpg", "high"))
    print(analysis)
