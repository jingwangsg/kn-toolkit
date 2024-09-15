from torch.utils.data import DistributedSampler


class StatefulDistributedSampler(DistributedSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_pnt = -1

    def __iter__(self):
        indices = list(super().__iter__())

        st = self.gen_pnt + 1
        for i in range(st, len(indices)):
            self.gen_pnt = i
            yield indices[i]

    def state_dict(self):
        return {
            "gen_pnt": self.gen_pnt,
            "epoch": self.epoch,
            "seed": self.seed,
            "rank": self.rank,
            "num_replicas": self.num_replicas,
            "total_size": self.total_size,
            "shuffle": self.shuffle,
        }

    def load_state_dict(self, state_dict):
        def _safe_overwrite(variable_name, ignore_diff=False):
            if variable_name in state_dict:
                if not ignore_diff:
                    assert getattr(self, variable_name) == state_dict[variable_name]
                setattr(self, variable_name, state_dict[variable_name])

        _safe_overwrite("epoch", ignore_diff=True)
        _safe_overwrite("gen_pnt", ignore_diff=True)
        _safe_overwrite("seed", ignore_diff=True)
        for variable_name in state_dict.keys():
            _safe_overwrite(variable_name, ignore_diff=False)
