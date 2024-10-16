"""Object Counter using MultiModal LLMs."""

from dataclasses import dataclass
from typing import List, Tuple

from semantix import Semantic
from semantix.llms import OpenAI
from semantix.types import Image
from PIL import Image as PILImage

llm = OpenAI(verbose=True)

@dataclass
class Object:
    label: str
    location: Semantic[Tuple[int, int], "XY Cordinates of the Object"] # type: ignore
    description: str

@dataclass
class Detections:
    objects: List[Object]
    summary: str

Object.__doc__ = ""
Detections.__doc__ = ""

@llm.enhance("Detect Objects in the given Image", method="CoT")
def detect(img: Image, img_dim: Tuple[int, int]) -> Detections: ...

if __name__ == "__main__":
    img_url = "examples/fruits.jpg"
    img = PILImage.open(img_url)
    detections = detect(img=Image(img_url, quality="high"), img_dim=img.size)
    print(detections)