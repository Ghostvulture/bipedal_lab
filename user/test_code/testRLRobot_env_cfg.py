import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass



##
# Scene definition
##
@configclass
class testRLRobotSceneCfg(InteractiveSceneCfg):






@configclass
class ActionCfg:




@configclass
class ObservationCfg:




@configclass
class EventCfg:



@configclass
class RewardCfg:





@configclass
class TerminationCfg:




@configclass
class testRLRobotEnvCfg(ManagerBasedRLEnvCfg):