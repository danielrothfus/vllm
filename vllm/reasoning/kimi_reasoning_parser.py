# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Optional, Union

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser, ReasoningParserManager

logger = init_logger(__name__)


@ReasoningParserManager.register_module("kimi")
class KimiReasoningParser(ReasoningParser):
    """
    Reasoning parser for Kimi model.

    The Kimi model uses ◁think▷...◁/think▷ tokens to denote reasoning
    text. This parser extracts the reasoning content from the model output.
    """

    start_token: str = "◁think▷"
    end_token: str = "◁/think▷"

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction.")

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        # Text-based parser doesn't use token IDs for reasoning detection
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract the content after the end tokens
        """
        # Text-based parser returns all input_ids as content
        return input_ids

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
        Handles streaming output where previous + delta = current.
        For text ◁think▷abc◁/think▷xyz:
        - 'abc' goes to reasoning_content
        - 'xyz' goes to content
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
                # reasoning content continues
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

    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content from the model output.

        For text ◁think▷abc◁/think▷xyz:
        - 'abc' goes to reasoning_content
        - 'xyz' goes to content

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        # Kimi dev is loosey goosey with think tags and can send multiple,
        # mismatched, or malformed tags. This parser is designed to be "best effort"
        # and never fail, regardless of what the model outputs.
        #
        # Strategy:
        # 1. If text starts with ◁think▷: Extract everything up to first ◁/think▷ 
        #    as reasoning, rest as content
        # 2. If text doesn't start with ◁think▷: Everything goes to content, 
        #    even if think tags appear later
        # 3. Never fail regardless of malformed/multiple/mismatched tags
        #
        # This ensures vLLM won't crash with poorly behaved models while still
        # correctly handling the common case.
        
        # Check if the output starts with the start token
        if not model_output.startswith(self.start_token):
            # If no start token at beginning, treat everything as content, not reasoning
            return None, model_output
        
        # Remove the start token from the output
        model_output_without_start = model_output[len(self.start_token):]

        # Look for the end token
        if self.end_token not in model_output_without_start:
            # If we have start token but no end token, treat everything as reasoning
            return model_output_without_start, None
        else:
            reasoning_content, _, content = model_output_without_start.partition(
                self.end_token)
            # If generation stops right after end-of-think, return null content
            final_content = content or None
            return reasoning_content, final_content