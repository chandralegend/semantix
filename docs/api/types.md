# Types API Reference

The `semantix.types` module provides a set of classes and functions to work with the different types of data that are used in the library.

## Image

```python
class Image:
    file_path: str
    quality: str = "low"
```

The `Image` class is used to represent an image in the library. `Image` class converts the image into a base64 encoded string to be used in the Large Language Model.

!> `pillow` library is required to use the `Image` class. If you don't have it installed, you can install it using the `pip install pillow` or `pip install semantix[image]`.

### Parameters

- `file_path` (str): The path to the image file.
- `quality` (str): The quality of the image. Default is `"low"`. Options are `"low"`, `"medium"`, `"high"`.

### Example

```python
from semantix.types import Image

img = Image("path/to/image.jpg")

def get_person(img: Semantic[Image, "Image of the Person"]) -> Person:
    ...
```

## Video

```python
class Video:
    file_path: str
    seconds_per_frame: int = 2
    quality: str = "low"
```

The `Video` class is used to represent a video in the library. `Video` class converts the video into a base64 encoded list of frames to be used in the Large Language Model.

!> `opencv-python` library is required to use the `Video` class. If you don't have it installed, you can install it using the `pip install opencv-python-headless` or `pip install semantix[video]`.

### Parameters

- `file_path` (str): The path to the video file.
- `seconds_per_frame` (int): The number of seconds per frame to extract from the video. Default is 2.
- `quality` (str): The quality of the video. Default is `"low"`. Options are `"low"`, `"medium"`, `"high"`.

### Example

```python
from semantix.types import Video

video = Video("path/to/video.mp4", 2)

def get_person(video: Semantic[Video, "Video of the Person"]) -> Person:
    ...
```

!> Use a higher value for `seconds_per_frame` if you want to extract fewer frames from the video and also to reduced the context used in the Large Language Model.
