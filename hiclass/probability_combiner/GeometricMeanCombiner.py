import numpy as np
from networkx.exception import NetworkXError
from hiclass.probability_combiner.ProbabilityCombiner import ProbabilityCombiner
from collections import defaultdict

class GeometricMeanCombiner(ProbabilityCombiner):
    def combine(self, proba):
        res = [proba[0]]
        log_sum = [np.log(proba[0])]
        for level in range(1, self.classifier.max_levels_):
            level_probs = np.zeros_like(proba[level])
            level_log_sum = np.zeros_like(proba[level])
            # find all predecessors of a node
            predecessors = self._find_predecessors(level)

            for node in predecessors.keys():
                index = self.classifier.class_to_index_mapping_[level][node]
                # find indices of all predecessors
                predecessor_indices = [self.classifier.class_to_index_mapping_[level-1][predecessor] for predecessor in predecessors[node]]
                # combine probabilities of all predecessors
                predecessors_combined_log_prob = np.log(np.sum([np.exp(log_sum[level-1][:, pre_index]) for pre_index in predecessor_indices], axis=0))

                level_log_sum[:, index] += (np.log(proba[level][:, index]) + predecessors_combined_log_prob)
                level_probs[:, index] = np.exp(level_log_sum[:, index] / (level+1))
            
            log_sum.append(level_log_sum)
            res.append(level_probs)
        return self._normalize(res)
