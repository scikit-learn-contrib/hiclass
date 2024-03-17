import numpy as np
from networkx.exception import NetworkXError
from hiclass.probability_combiner.ProbabilityCombiner import ProbabilityCombiner

class GeometricMeanCombiner(ProbabilityCombiner):
    def combine(self, proba):
        res = [proba[0]]
        log_sum = [np.log(proba[0])]
        for level in range(1, self.classifier.max_levels_):
            level_probs = np.zeros_like(proba[level])
            level_log_sum = np.zeros_like(proba[level])
            for node in self.classifier.classes_[level]:
                try:
                    predecessor = list(self.classifier.hierarchy_.predecessors(node))[0]
                except NetworkXError:
                    # skip empty levels
                    continue

                predecessor_index = self.classifier.class_to_index_mapping_[level-1][predecessor]
                index = self.classifier.class_to_index_mapping_[level][node]
                level_log_sum[:, index] += (np.log(proba[level][:, index]) + log_sum[level-1][:, predecessor_index])
                level_probs[:, index] = np.exp(level_log_sum[:, index] / (level+1))

            log_sum.append(level_log_sum)
            res.append(level_probs)
        return res
