# Copyright (c) ModelScope Contributors. All rights reserved.
"""Engine-agnostic non-blocking generation submission, shared by the TQ samplers.

The server's data-plane path (``/twinkle/sample_to_data_plane``) admits a rollout without blocking the
Ray actor: it calls :meth:`submit_generation` (which schedules the work on the sampler's background
event loop and returns immediately), polls :meth:`get_generation_status`, then :meth:`collect_generation`
-- with :meth:`cancel_generation` / :meth:`cancel_all_generations` to drop an in-flight or a retained
submission. That quartet is pure ``Future`` bookkeeping over the actor's event loop; only the actual
generation is engine-specific, so it lives here once and each backend supplies the single
``_generate_inputs`` coroutine. ``VLLMSamplerTQ`` and ``SGLangSamplerTQ`` both mix this in, which keeps a
concurrency fix from having to be mirrored across backends.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import Future
from typing import Any

from twinkle import remote_function
from twinkle.data_format import SampleResponse, SamplingParams


def _dispatch_generation(
    worker_count: int,
    worker_index: int,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    **_dispatch_kwargs,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Slice CS inputs while allowing a prompt count smaller than DP size."""
    sliced_args = list(args)
    sliced_kwargs = dict(kwargs)
    if len(sliced_args) > 1:
        inputs = sliced_args[1]
        target = ('args', 1)
    elif 'inputs' in sliced_kwargs:
        inputs = sliced_kwargs['inputs']
        target = ('kwargs', 'inputs')
    else:
        raise ValueError('submit_generation requires inputs')

    input_list = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
    size, remainder = divmod(len(input_list), worker_count)
    start = worker_index * size + min(worker_index, remainder)
    stop = (worker_index + 1) * size + min(worker_index + 1, remainder)
    shard = input_list[start:stop]
    if target[0] == 'args':
        sliced_args[target[1]] = shard
    else:
        sliced_kwargs[target[1]] = shard
    return tuple(sliced_args), sliced_kwargs


class GenerationSubmissionMixin:
    """The non-blocking submit/poll/collect/cancel generation API, engine-agnostic.

    A concrete subclass must be decorated with ``@remote_class()``, own a running background event loop
    at ``self._async_loop``, initialise ``self._generation_submissions`` to a dict, and implement
    ``async def _generate_inputs(self, inputs, sampling_params, *, adapter_name, adapter_path,
    use_base_model) -> list[SampleResponse]``. Everything else -- admission, status, collection and
    cancellation -- is the same ``Future`` bookkeeping regardless of the engine, so it is defined once
    here.
    """

    def _submit_in_loop(self, coro) -> Future:
        return asyncio.run_coroutine_threadsafe(coro, self._async_loop)

    async def _generate_inputs(
        self,
        inputs: Any,
        sampling_params: SamplingParams | dict[str, Any] | None,
        *,
        adapter_name: str,
        adapter_path: str | None,
        use_base_model: bool,
    ) -> list[SampleResponse]:
        """Run one admitted submission's generation on the engine. Supplied by each backend."""
        raise NotImplementedError(f'{type(self).__name__} must implement _generate_inputs')

    @remote_function(dispatch=_dispatch_generation, collect='none', lazy_collect=False)
    def submit_generation(
        self,
        submission_id: str,
        inputs: Any,
        sampling_params: SamplingParams | dict[str, Any] | None = None,
        adapter_name: str = '',
        adapter_path: str | None = None,
        *,
        use_base_model: bool = False,
    ) -> dict[str, Any]:
        """Submit a CS sampling shard without blocking the Ray actor.

        The generated responses stay local to this DP worker until
        :meth:`collect_generation` consumes them. This gives the HTTP
        service the same fast-admission property as the native TQ rollout path
        without exposing PromptGroup or BatchMeta to the client.
        """
        if submission_id in self._generation_submissions:
            raise KeyError(f'generation submission already exists: {submission_id}')
        future = self._submit_in_loop(
            self._generate_inputs(
                inputs,
                sampling_params,
                adapter_name=adapter_name,
                adapter_path=adapter_path,
                use_base_model=use_base_model,
            ))
        self._generation_submissions[submission_id] = future
        return {'submission_id': submission_id, 'status': 'running'}

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def get_generation_status(self, submission_id: str) -> dict[str, Any]:
        """Return this DP worker's submission state without waiting."""
        future = self._generation_submissions.get(submission_id)
        if future is None:
            return {
                'submission_id': submission_id,
                'status': 'missing',
                'error': f'unknown generation submission: {submission_id}',
            }
        if future.cancelled():
            return {'submission_id': submission_id, 'status': 'cancelled'}
        if not future.done():
            return {'submission_id': submission_id, 'status': 'running'}
        error = future.exception()
        if error is not None:
            return {
                'submission_id': submission_id,
                'status': 'failed',
                'error': f'{type(error).__name__}: {error}',
            }
        return {'submission_id': submission_id, 'status': 'completed'}

    @remote_function(dispatch='all', collect='flatten', lazy_collect=False)
    def collect_generation(self, submission_id: str) -> list[SampleResponse]:
        """Consume completed responses from every DP worker."""
        future = self._generation_submissions.get(submission_id)
        if future is None:
            raise KeyError(f'unknown generation submission: {submission_id}')
        if not future.done():
            raise RuntimeError(f'generation submission is still running: {submission_id}')
        try:
            return future.result()
        finally:
            self._generation_submissions.pop(submission_id, None)

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def cancel_generation(self, submission_id: str) -> dict[str, Any]:
        """Cancel and forget one generation submission on every DP worker."""
        future = self._generation_submissions.pop(submission_id, None)
        if future is None:
            return {'submission_id': submission_id, 'status': 'missing'}
        was_done = future.done()
        cancelled = future.cancel()
        if cancelled:
            status = 'cancelled'
        elif was_done:
            status = 'completed'
        else:
            status = 'cancellation_requested'
        return {
            'submission_id': submission_id,
            'status': status,
        }

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def cancel_all_generations(self) -> dict[str, int]:
        """Cancel all retained CS submissions during replica shutdown."""
        submissions = list(self._generation_submissions.values())
        self._generation_submissions.clear()
        cancelled = sum(future.cancel() for future in submissions if not future.done())
        return {'submissions': len(submissions), 'cancelled': cancelled}
