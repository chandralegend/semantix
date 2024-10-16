# RELEASES

## `0.1.7` - 2024-10-16
- [FIX] Output Extraction Not Working
- [IMPROVEMENT] Now you can get the `enhance` decorator straight from the LLM
- [IMPROVEMENT] Prompts are more simplified to make it token efficient

## `0.1.6` - 2024-09-23
- [FIX] Fixed Anthropic, Cohere Integration
- [FIX] Fixed Self Healing Mechanism
- [FEATURE] Support for Pydantic Base Models (Can be used in advanced validation needs)
- [CHORE] Optimized Default Prompt Configurations

## `0.1.5` - 2024-09-17
- [FEATURE] Added TogetherAI, MistralAI, Groq Integration

## `0.1.4` - 2024-09-17
- [FEATURE] Added Helper Functions to create new classes and enums easily. Helpful when creating classes with large number of parameters or enums with large number of values.
- [FEATURE] Retries are now available in the `enhance` decorator. This will allow the function to retry the LLM inference if it fails, increasing the chances of getting a correct answer.
- [CHORE] Updated Documentation with the new features
- [CHORE] Added more examples in the `examples` folder

## `0.1.3` - 2024-09-16
- [FIX] Much better performance

## `0.1.2` - 2024-09-14
- [FEATURE] Added Anthropic, Cohere Integration
- [CHORE] Added Documentation

## `0.1.1` - 2024-09-08
- [CHORE] Updated README.md with badges and installation instructions
- [CHORE] Added try.ipynb to the repository

## `0.1.0` - 2024-09-07
- [FEATURE] Vision AI Capabilities added
- [FEATURE] OpenAI Integration added
- [FEATURE] Tool Integration is added. Not Yet Usable (ReACT Prompting is required)
- [FEATURE] CoT, Reasoning Techniques are added
