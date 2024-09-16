from typing import List

from semantix import Semantic, enhance
from semantix.llms import OpenAI
from semantix.utils import create_enum


llm = OpenAI(max_tokens=2048)

multilabel_classes = [
    "lists_createoradd",
    "calendar_query",
    "email_sendemail",
    "news_query",
    "play_music",
    "play_radio",
    "qa_maths",
    "email_query",
    "weather_query",
    "calendar_set",
    "iot_hue_lightdim",
    "takeaway_query",
    "social_post" "email_querycontact",
    "qa_factoid",
    "calendar_remove",
    "cooking_recipe",
    "lists_query",
    "general_quirky",
    "alarm_query",
    "takeaway_order",
    "iot_hue_lightup",
    "lists_remove",
    "qa_currency",
    "play_game",
    "play_audiobook",
    "qa_definition",
    "music_query",
    "datetime_query",
    "transport_query",
    "iot_hue_lightoff",
    "iot_hue_lightchange",
    "iot_hue_lighton",
    "alarm_set",
    "music_likeness",
    "recommendation_movies",
    "transport_ticket",
    "recommendation_locations",
    "audio_volume_mute",
    "iot_wemo_on",
    "play_podcasts",
    "datetime_convert",
    "audio_volume_other",
    "recommendation_events",
    "alarm_remove",
    "iot_coffee",
    "music_dislikeness",
    "general_joke",
    "social_query",
]

Label = create_enum(
    "Label",
    {label.upper(): label for label in multilabel_classes},
    "The labels for the multilabel classification task",
)


@enhance("Classify the given text into multiple labels", llm)
def classify(text: str) -> Semantic[List[Label], "Relevant Labels"]: ...  # type: ignore


texts = [
    "Hey Google, can you play some music for me?",
    "What is the weather like in London? Also, can you set an alarm for 6 AM tomorrow?",
    "Turn the lights on and play some music",
]

for text in texts:
    print(classify(text=text))
