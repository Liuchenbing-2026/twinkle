# Copyright (c) ModelScope Contributors. All rights reserved.
"""SGLang sampler with the non-blocking submission API the server data plane needs.

The sglang counterpart of :class:`VLLMSamplerTQ`. It adds no native TransferQueue rollout path -- only
the client/server data-plane path is served here -- so it is just :class:`SGLangSampler` plus the shared
:class:`GenerationSubmissionMixin` quartet (``submit_generation`` / ``get_generation_status`` /
``collect_generation`` / ``cancel_generation``) and the one engine-specific coroutine that mixin calls,
:meth:`_generate_inputs`. That makes ``sampler_type='sglang_async'`` able to back
``/twinkle/sample_to_data_plane``, which plain ``sglang`` cannot because it has no ``submit_generation``.
"""
from __future__ import annotations

import asyncio
from copy import copy
from typing import Any, Optional

from twinkle import remote_class
from twinkle.data_format import SampleResponse, SamplingParams
from twinkle.hub import HubOperation
from twinkle.sampler.sglang_sampler import SGLangSampler
from .generation_submissions import GenerationSubmissionMixin


@remote_class()
class SGLangSamplerTQ(GenerationSubmissionMixin, SGLangSampler):
    """:class:`SGLangSampler` that admits generations without blocking the Ray actor.

    The quartet lives in :class:`GenerationSubmissionMixin`; this class only supplies ``_async_loop``
    (inherited from ``SGLangSampler``), the ``_generation_submissions`` map, and the sglang-specific
    ``_generate_inputs``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The mixin's quartet tracks admitted-but-uncollected submissions here. SGLangSampler has no
        # such map, so initialise it after the engine (and its background loop) is up.
        self._generation_submissions: dict[str, Any] = {}

    async def _aregister_lora(self, adapter_path: Optional[str],
                              adapter_name: Optional[str] = None) -> Optional[str]:
        """Async counterpart of ``SGLangSampler._register_lora``.

        ``_generate_inputs`` runs *on* the sampler's background event loop, so it cannot call the
        blocking ``_register_lora`` (that would submit to the same loop and deadlock waiting for itself).
        Registration is cached by path for the same reason the sync one is: sglang's ``load_lora_adapter``
        on an already-registered name is an error, and a per-input adapter would otherwise re-register on
        every request.
        """
        if adapter_path is None:
            return None
        if adapter_path in self._registered_loras:
            return self._registered_loras[adapter_path]
        lora_name = adapter_name or f'lora_{len(self._registered_loras)}'
        local_path = await asyncio.to_thread(HubOperation.download_model, model_id_or_path=adapter_path)
        await self.engine.load_lora_adapter(lora_name=lora_name, lora_path=local_path)
        self._registered_loras[adapter_path] = lora_name
        return lora_name

    async def _generate_inputs(
        self,
        inputs: Any,
        sampling_params: SamplingParams | dict[str, Any] | None,
        *,
        adapter_name: str,
        adapter_path: str | None,
        use_base_model: bool,
    ) -> list[SampleResponse]:
        """Asynchronous counterpart of ``SGLangSampler.sample`` for the data-plane path."""
        if sampling_params is None:
            sampling_params = SamplingParams()
        elif isinstance(sampling_params, dict):
            sampling_params = SamplingParams.from_dict(sampling_params)

        inputs_list = self._normalize_inputs(inputs)
        if not inputs_list:
            return []

        # token-in-token-out correctness: sglang returns the generated token ids only as a side effect of
        # asking for logprobs -- without them SGLangEngine._output_token_ids re-tokenises the completion
        # text, and those ids can drift from what the model actually ran (special tokens, merges). The
        # data plane trains on new_input_feature's exact ids and pairs them with sampled_logprobs, so
        # force at least the sampled token's logprob. vLLM needs no such nudge because its engine always
        # returns token ids. copy() so the caller's params are untouched.
        sampling_params = copy(sampling_params)
        if not sampling_params.logprobs:
            sampling_params.logprobs = 1

        is_trajectory = 'input_ids' not in inputs_list[0]
        logprobs_only = False
        if sampling_params.max_tokens == 0:
            # sglang has no zero-token request, so ask for one token and drop it (mirrors SGLangSampler).
            sampling_params.max_tokens = 1
            logprobs_only = True

        image_data_list = [self._extract_image_data(feat) for feat in inputs_list]
        if is_trajectory:
            if self.template is None:
                raise ValueError('Use set_template to add a template when trying to input Trajectory')
            encoded_inputs = [
                self.encode_trajectory_for_sglang(trajectory, adapter_name, not logprobs_only)
                for trajectory in inputs_list
            ]
        else:
            encoded_inputs = inputs_list

        lora_name = None
        if not use_base_model:
            lora_name = await self._aregister_lora(adapter_path, adapter_name)

        return await asyncio.gather(*(
            self._sample_single(
                feat,
                sampling_params,
                image_data=image_data,
                lora_name=lora_name,
                logprobs_only=logprobs_only,
            ) for feat, image_data in zip(encoded_inputs, image_data_list)))
