import numpy as np
from hiclass.probability_combiner.ProbabilityCombiner import ProbabilityCombiner

class MultiplyCombiner(ProbabilityCombiner):
    def combine(self, proba):
        res = [proba[0]]
        for level in range(1, self.classifier.max_levels_):
            
            level_probs = np.zeros_like(proba[level])
            for node in self.classifier.classes_[level]:
                predecessor = list(self.classifier.hierarchy_.predecessors(node))[0]
                predecessor_index = self.classifier.class_to_index_mapping_[level-1][predecessor]
                index = self.classifier.class_to_index_mapping_[level][node]

                level_probs[:, index] = res[level-1][:, predecessor_index] * proba[level][:, index]

            res.append(level_probs)
        return res
