# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser, ReasoningParserManager

from transformers import PreTrainedTokenizerBase

STATE_INITIAL = 0
STATE_REASONING = 1
STATE_OUTPUT = 2

START_TEXT = '◁think▷'
START_TEXT_LENGTH = len(START_TEXT)
START_PREFIXES = set(START_TEXT[:i] for i in range(len(START_TEXT) - 1))

END_TEXT = '◁/think▷'
END_TEXT_LENGTH = len(END_TEXT)
END_PREFIXES = [END_TEXT[:i] for i in range(1, len(END_TEXT) - 1)]

@ReasoningParserManager.register_module("kimi3")
class Kimi3ReasoningParser(ReasoningParser):

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        self._state = STATE_INITIAL
        self._delta_accumulator = ''

    # Intended behavior:
    #   "◁think▷a◁/think▷b" → ("a", "b")
    #   "◁think▷a" → ("a", None)
    #   "◁think▷◁/think▷b" → (None, "b")
    #   "◁think▷a◁/think▷" → ("a", None)
    #   "◁think▷a◁think▷b◁/think▷c" → ("a◁think▷b", "c")
    #   "◁think▷a◁/think▷b◁/think▷c" → ("a", "b◁/think▷c")
    #   "◁think▷a◁/think▷b◁think▷c" → ("a", "b◁think▷c")
    #   "x◁think▷a◁/think▷b" → (None, "x◁think▷a◁/think▷b")
    #   "◁/think▷a" → (None, "◁/think▷a")
    #   "x" → (None, "x")
    def extract_reasoning_content(self, model_output, request):
        if not model_output.startswith(START_TEXT):
            return None, model_output
        
        model_output = model_output[START_TEXT_LENGTH:]
        reasoning_done_index = model_output.find(END_TEXT)
        if reasoning_done_index == -1:
            return model_output, None
        
        content_start_index = reasoning_done_index + END_TEXT_LENGTH
        return model_output[:reasoning_done_index] or None, model_output[content_start_index:] or None

    def extract_reasoning_content_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
    ):
        delta = self._delta_accumulator + delta_text
        self._delta_accumulator = ''

        # TODO: Make this return everything if thi sis the last delta that will come through if that's possible to detect.

        if self._state == STATE_INITIAL:
            if delta.startswith(START_TEXT):
                self._state = STATE_REASONING
                delta = delta[START_TEXT_LENGTH:]
            elif delta in START_PREFIXES:
                # May still be the start tag at the beginning of the response. Save state and wait for more data.
                self._delta_accumulator = delta
                return None
            else:
                self._state = STATE_OUTPUT

        # If we get here, we're either in STATE_REASONING or STATE_OUTPUT
        reasoning_delta = None
        if self._state == STATE_REASONING:
            reasoning_done_index = delta.find(END_TEXT)
            if reasoning_done_index == -1:
                for p in END_PREFIXES:
                    if delta.endswith(p):
                        self._delta_accumulator = p
                        return DeltaMessage(reasoning_content=delta[:-len(p)])
                        
                # If we get here, the end of our current text can't possibly be part of the end tag. Return everything.
                return DeltaMessage(reasoning_content=delta)

            self._state = STATE_OUTPUT
            reasoning_delta = delta[:reasoning_done_index]
            delta = delta[reasoning_done_index + END_TEXT_LENGTH:]

        # If we get here, we're definitely in STATE_OUTPUT
        return DeltaMessage(reasoning_content=reasoning_delta, content=delta or None)
