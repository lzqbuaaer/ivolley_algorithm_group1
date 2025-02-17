from src.enums.action import Action
from src.G3.rules.dig import digRule
from src.G3.rules.serve import serveRule
from src.G3.rules.passb import passRule
from src.G3.rules.spike import spikeRule
from src.G3.rules.block import blockRule

class Rule:
    def __init__(self, type):
        self.type = type

    def __call__(self, actionPoints, images, candidates, persons, balls, hit=None, hand_pose=None, *args, **kwargs):
        # 按顺序调用具体规则
        if self.type is Action.Dig:
            return digRule.sum_rules(actionPoints, images, candidates, persons, balls, hit)
        elif self.type is Action.Pass:
            return passRule.sum_rules(actionPoints, images, candidates, persons, balls, hit, hand_pose)
        elif self.type is Action.Serve:
            return serveRule.sum_rules(actionPoints, images, candidates, persons, balls, hit)
        elif self.type is Action.Spike:
            return spikeRule.spike_analysis(actionPoints, images, candidates, persons, balls, hit)
        elif self.type is Action.Block:
            return blockRule.block_analysis(actionPoints, images, candidates, persons, balls, hit)
        else:
            return "暂不支持该动作"
