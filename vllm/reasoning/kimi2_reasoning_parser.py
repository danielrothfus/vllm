# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from collections.abc import Sequence
from typing import Optional, Union

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser, ReasoningParserManager

logger = init_logger(__name__)


@ReasoningParserManager.register_module("kimi2")
class Kimi2ReasoningParser(ReasoningParser):
    """
    Reasoning parser for Kimi model (regex-based implementation).

    The Kimi model uses ◁think▷...◁/think▷ tokens to denote reasoning
    text. This parser extracts the reasoning content from the model output
    using regex pattern matching.
    
    Kimi dev is loosey goosey with think tags and can send multiple,
    mismatched, or malformed tags. This parser is designed to be "best effort"
    and never fail, regardless of what the model outputs.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction.")

        # Define the start and end tokens for Kimi reasoning
        self.start_token = "◁think▷"
        self.end_token = "◁/think▷"
        
        # Create regex pattern to match reasoning content
        # Pattern: ◁think▷(.*?)◁/think▷(.*)
        # Group 1: reasoning content, Group 2: response content
        self.reasoning_regex = re.compile(
            rf"{re.escape(self.start_token)}(.*?){re.escape(self.end_token)}(.*)",
            re.DOTALL)

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        # Text-based parser doesn't use token IDs for reasoning detection
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract the content after the end tokens
        """
        # Text-based parser returns all input_ids as content
        return input_ids

    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content from the model output using regex.

        For text ◁think▷abc◁/think▷xyz:
        - 'abc' goes to reasoning_content
        - 'xyz' goes to content

        If the sequence doesn't match what we expect, i.e., the model generates
        something else, all content is considered non-reasoning content.

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """
        # Use regex to find reasoning pattern
        re_match = self.reasoning_regex.findall(model_output)
        if not re_match:
            # If no match found, treat everything as content, not reasoning
            return None, model_output
        
        reasoning_content, response_content = re_match[0]
        if not response_content:
            return reasoning_content, None
        return reasoning_content, response_content

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        """
        Extract reasoning content from a delta message using text-based parsing.
        
        This is a simplified version that focuses on robustness over complexity.
        For streaming, we use the same text-based approach as the main kimi parser.
        """
        # Check if ◁think▷ is present in previous or delta text
        if self.start_token in previous_text:
            if self.end_token in delta_text:
                # ◁think▷ in previous, ◁/think▷ in delta,
                # extract reasoning content
                end_index = delta_text.find(self.end_token)
                reasoning_content = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token):]
                return DeltaMessage(
                    reasoning_content=reasoning_content,
                    content=content if content else None,
                )
            elif self.end_token in previous_text:
                # ◁think▷ in previous, ◁/think▷ in previous,
                # reasoning content ends, this is regular content
                return DeltaMessage(content=delta_text)
            else:
                # ◁think▷ in previous, no ◁/think▷ in previous or delta,
                # reasoning content continues
                return DeltaMessage(reasoning_content=delta_text)
        elif self.start_token in delta_text:
            if self.end_token in delta_text:
                # ◁think▷ in delta, ◁/think▷ in delta, extract reasoning content
                start_index = delta_text.find(self.start_token)
                end_index = delta_text.find(self.end_token)
                reasoning_content = delta_text[start_index +
                                               len(self.start_token):end_index]
                content = delta_text[end_index + len(self.end_token):]
                return DeltaMessage(
                    reasoning_content=reasoning_content,
                    content=content if content else None,
                )
            else:
                # ◁think▷ in delta, no ◁/think▷ in delta,
                # reasoning content continues
                return DeltaMessage(reasoning_content=delta_text)
        else:
            # No ◁think▷ in previous or delta, also need to check for ◁/think▷.
            # Because the model may have generated ◁/think▷ without ◁think▷
            if self.end_token in delta_text:
                # ◁/think▷ in delta with more tokens,
                # extract reasoning content and content
                end_index = delta_text.find(self.end_token)
                reasoning_content = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token):]
                return DeltaMessage(
                    reasoning_content=reasoning_content,
                    content=content if content else None,
                )
            elif self.end_token in previous_text:
                # ◁/think▷ in previous, thinking content ends
                return DeltaMessage(content=delta_text)
            else:
                # no ◁/think▷ in previous or delta, reasoning content continues
                return DeltaMessage(reasoning_content=delta_text)