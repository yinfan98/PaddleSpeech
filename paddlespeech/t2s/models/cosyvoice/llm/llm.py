# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Optional, Callable, List, Generator

import paddle
from paddlenlp.transformers import Qwen2ForCausalLM
from paddle.nn import Pad1D

class Qwen2Encoder(paddle.nn.Layer):
    def __init__(self, pretrain_path):
        super().__init__()
        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache


class Qwen2LM(paddle.nn.Layer):
    def __init__(
        self,
        llm_input_size: int,
        llm_output_size: int,
        speech_token_size: int,
        llm: paddle.nn.Layer,
        sampling: Callable,
        length_normalized_loss: bool = True,
        lsm_weight: float = 0.0,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2

        self.llm_embedding = paddle.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = paddle.nn.Linear(llm_output_size, speech_token_size + 3)

        # 3. [Optional] build speech token related modules
        self.speech_embedding = paddle.nn.Embedding(speech_token_size + 3, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def sampling_ids(
        self,
        weighted_scores: paddle.Tensor,
        decoded_tokens: List,
        sampling: int,
        ignore_eos: bool = True,
    ):
        num_trials, max_trials = 0, 100
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
            num_trials += 1
            if num_trials > max_trials:
                raise RuntimeError('sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!'.format(max_trials))
        return top_ids

    def inference(
        self,
        text: paddle.Tensor,
        text_len: paddle.Tensor,
        prompt_text: paddle.Tensor,
        prompt_text_len: paddle.Tensor,
        prompt_speech_token: paddle.Tensor,
        prompt_speech_token_len: paddle.Tensor,
        embedding: paddle.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
    ) -> Generator[paddle.Tensor, None, None]:
        device = text.device
        text = paddle.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.llm.model.model.embed_tokens(text)

        # 2. encode embedding
        embedding = paddle.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = paddle.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = paddle.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        cache = None
        for i in range(max_len):
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                      masks=paddle.tril(paddle.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(paddle.bool),
                                                      cache=cache)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            if top_ids > self.speech_token_size:
                continue
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)