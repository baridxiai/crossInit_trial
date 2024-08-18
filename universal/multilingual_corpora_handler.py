from datasets import IterableDataset
import sys
import torch
from datasets.iterable_dataset import (
    # _apply_feature_types_on_example,
    # get_formatter,
    # TensorFormatter,
    # cast_to_python_objects,
    # _reset_fsspec_lock,
    _check_if_features_can_be_aligned,
    Features,
    _align_features,
    CyclingMultiSourcesExamplesIterable,
    RandomlyCyclingMultiSourcesExamplesIterable,
    DatasetInfo,
    _HasNextIterator
)
from copy import deepcopy
import numpy as np

class NewCyclingMultiSourcesExamplesIterable(CyclingMultiSourcesExamplesIterable):
    def __init__(
        self,
        ex_iterables,
        stopping_strategy = "first_exhausted",
        batch_size = None,
        gradient_accumulation_step = 1
    ):
        super().__init__(ex_iterables, stopping_strategy)
        self.batch_size =batch_size
        self.gradient_accumulation_step = gradient_accumulation_step
        self.current_gradient_tower_step = 0
        self.current_language = None
    def __iter__(self):
        iterators = [_HasNextIterator(ex_iterable) for ex_iterable in self.ex_iterables]
        indices_iterator = self._get_indices_iterator()

        is_exhausted = np.full(len(self.ex_iterables), False)
        self.current_gradient_tower_step += 1
        if self.current_gradient_tower_step <= self.gradient_accumulation_step:
            if self.current_language is None:
                for i in indices_iterator:
                    self.current_language = i
                    try:  # let's pick one example from the iterator at index i
                        if self.batch_size is None:
                            yield next(iterators[i])
                        else:
                            for _ in range(self.batch_size):
                                yield next(iterators[i])
                        # it will resume from the yield at the next call so that we can directly test if the iterable is exhausted and if we need to break out of the loop
                        if not iterators[i].hasnext():
                            is_exhausted[i] = True

                            if self.bool_strategy_func(is_exhausted):
                                # if the stopping criteria is met, break the main for loop
                                break
                            # otherwise reinitialise the iterator and yield the first example
                            iterators[i] = _HasNextIterator(self.ex_iterables[i])

                    except StopIteration:
                        # here it means that the i-th iterabledataset is empty, i.e we never have the occasion to yield an element of the i-th dataset.
                        # we still check if the stopping criteria is met and if we break out of the loop in case of an oversampling strategy
                        is_exhausted[i] = True

                        if self.bool_strategy_func(is_exhausted):
                            # if the stopping criteria is met, break the main for loop
                            break
            else:
                try:  # let's pick one example from the iterator at index i
                    if self.batch_size is None:
                        yield next(iterators[self.current_language])
                    else:
                        for _ in range(self.batch_size):
                            yield next(iterators[self.current_language])
                    # it will resume from the yield at the next call so that we can directly test if the iterable is exhausted and if we need to break out of the loop
                    if not iterators[self.current_language].hasnext():
                        is_exhausted[self.current_language] = True
                        iterators[self.current_language] = _HasNextIterator(self.ex_iterables[self.current_language])

                except StopIteration:
                    # here it means that the i-th iterabledataset is empty, i.e we never have the occasion to yield an element of the i-th dataset.
                    # we still check if the stopping criteria is met and if we break out of the loop in case of an oversampling strategy
                    is_exhausted[self.current_language] = True
        else:
            self.current_gradient_tower_step =0
            for i in indices_iterator:
                self.current_language = None
                try:  # let's pick one example from the iterator at index i
                    if self.batch_size is None:
                        yield next(iterators[i])
                    else:
                        for _ in range(self.batch_size):
                            yield next(iterators[i])
                    # it will resume from the yield at the next call so that we can directly test if the iterable is exhausted and if we need to break out of the loop
                    if not iterators[i].hasnext():
                        is_exhausted[i] = True

                        if self.bool_strategy_func(is_exhausted):
                            # if the stopping criteria is met, break the main for loop
                            break
                        # otherwise reinitialise the iterator and yield the first example
                        iterators[i] = _HasNextIterator(self.ex_iterables[i])

                except StopIteration:
                    # here it means that the i-th iterabledataset is empty, i.e we never have the occasion to yield an element of the i-th dataset.
                    # we still check if the stopping criteria is met and if we break out of the loop in case of an oversampling strategy
                    is_exhausted[i] = True

                    if self.bool_strategy_func(is_exhausted):
                        # if the stopping criteria is met, break the main for loop
                        break
class multilingualRandomlyCyclingMultiSourcesExamplesIterable(NewCyclingMultiSourcesExamplesIterable):
    def __init__(
        self,
        ex_iterables,
        generator,
        probabilities = None,
        stopping_strategy = "first_exhausted",
        batch_size = None,
        gradient_accumulation_step = 1
    ):
        super().__init__(ex_iterables, stopping_strategy,batch_size,gradient_accumulation_step)
        self.generator = deepcopy(generator)
        self.probabilities = probabilities
        # TODO(QL): implement iter_arrow

    @staticmethod
    def _iter_random_indices(
        rng: np.random.Generator,
        num_sources: int,
        random_batch_size=1000,
        p = None,
    ) :
        """Get an infinite iterator that randomly samples the index of the source to pick examples from."""
        if p is None:
            while True:
                yield from (int(i) for i in rng.integers(0, num_sources, size=random_batch_size))
        else:
            while True:
                yield from (int(i) for i in rng.choice(num_sources, size=random_batch_size, p=p))

    def _get_indices_iterator(self):
        rng = deepcopy(self.generator)
        # this is an infinite iterator that randomly samples the index of the source to pick examples from
        return self._iter_random_indices(rng, len(self.ex_iterables), p=self.probabilities)

    def shuffle_data_sources(self, generator: np.random.Generator) -> "multilingualRandomlyCyclingMultiSourcesExamplesIterable":
        """Shuffle the data sources of each wrapped examples iterable."""
        ex_iterables = [ex_iterable.shuffle_data_sources(generator) for ex_iterable in self.ex_iterables]
        return multilingualRandomlyCyclingMultiSourcesExamplesIterable(
            ex_iterables,
            generator=generator,
            probabilities=self.probabilities,
            stopping_strategy=self.stopping_strategy,
        )
    def shard_data_sources(self, worker_id: int, num_workers: int) -> "multilingualRandomlyCyclingMultiSourcesExamplesIterable":
        """Either keep only the requested shard, or propagate the request to the underlying iterable."""
        return multilingualRandomlyCyclingMultiSourcesExamplesIterable(
            [iterable.shard_data_sources(worker_id, num_workers) for iterable in self.ex_iterables],
            self.generator,
            self.probabilities,
            self.stopping_strategy,
        )
def multilingual_interleave_datasets(
    datasets,
    probabilities=None,
    seed=None,
    info=None,
    split=None,
    stopping_strategy="first_exhausted",
    batch_size = 6,
    gradient_accumulation_step = 2

):

    datasets = [d._resolve_features() for d in datasets]

    # Perform checks
    _check_if_features_can_be_aligned([dset.features for dset in datasets])

    # TODO: improve this to account for a mix of ClassLabel and Value for example
    # right now it would keep the type of the first dataset in the list
    features = Features(
        {
            k: v
            for features in _align_features([dset.features for dset in datasets])
            for k, v in features.items()
        }
    )

    ex_iterables = [d._ex_iterable for d in datasets]

    # Use cycling or random cycling of sources
    generator = np.random.default_rng(seed)
    ex_iterable = multilingualRandomlyCyclingMultiSourcesExamplesIterable(
        ex_iterables,
        generator=generator,
        probabilities=probabilities,
        stopping_strategy=stopping_strategy,
        batch_size = batch_size,
        gradient_accumulation_step = gradient_accumulation_step
    )
    # Set new info - we update the features
    # setting the features also ensures to fill missing columns with None
    if info is None:
        info = DatasetInfo.from_merge([d.info for d in datasets])
    else:
        info = info.copy()
    info.features = features
    # Get all the auth tokens per repository - in case the datasets come from different private repositories
    token_per_repo_id = {
        repo_id: token
        for dataset in datasets
        for repo_id, token in dataset._token_per_repo_id.items()
    }
    # Return new daset
    return IterableDataset(
        ex_iterable=ex_iterable,
        info=info,
        split=split,
        token_per_repo_id=token_per_repo_id,
    )
