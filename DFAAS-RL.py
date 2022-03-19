
import argparse
import numpy as np
from gym.spaces import Discrete, Tuple, Box, Dict

import pandas as pd

import ray
from ray import tune
from ray.rllib.agents.maml.maml_torch_policy import KLCoeffMixin as \
    TorchKLCoeffMixin
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy, KLCoeffMixin, \
    ppo_surrogate_loss as tf_loss
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule
from ray.rllib.policy.torch_policy import LearningRateSchedule as TorchLR, \
    EntropyCoeffSchedule as TorchEntropyCoeffSchedule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.tf_ops import explained_variance, make_tf_callable
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces.multi_discrete import MultiDiscrete
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.modelv2 import ModelV2

import random

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

other_OBS = "other_obs"
other_ACTION = "other_action"

AGENT_OBS_SPACE = Dict({
    "action_mask": Box(0, 1.0, (100,)),
    "observation": Tuple([Discrete(100),Discrete(100)])
})

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf",
    help="The DL framework specifier.")
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.")
parser.add_argument(
    "--stop-iters",
    type=int,
    default=5000,
    help="Number of iterations to train.")
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=200000,
    help="Number of timesteps to train.")
parser.add_argument(
    "--stop-reward",
    type=float,
    default=10000,
    help="Reward at which we stop training.")

from typing import Dict as typingDict
from ray.rllib.utils.typing import TensorType as typingTensorType

from ray.rllib.utils.typing import TensorType, List, ModelConfigDict
from ray.rllib.models.modelv2 import restore_original_dimensions



class modFullyConnectedNetwork(FullyConnectedNetwork):
    def forward(self, input_dict: typingDict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):



      
        original_obs = restore_original_dimensions(input_dict["obs"], self.obs_space, "[tf|torch]")

        model_out, self._value_out = self.base_model(original_obs["observation"][0])
        inf_mask = tf.maximum(tf.math.log(original_obs["action_mask"]), tf.float32.min)
        masked_logits = model_out + inf_mask

        return masked_logits, state

class DFaaSCriticModel(TFModelV2):
    """Multi-agent model that implements a centralized value function."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(DFaaSCriticModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)


        # Base of the model
        self.model = modFullyConnectedNetwork(Box(-1.0, 1.0, (100,)), action_space, num_outputs, model_config, name)
        
        obs = tf.keras.layers.Input(shape=(300, ), name="obs")
        other_obs1 = tf.keras.layers.Input(shape=(300, ), name="other_obs1")
        other_act1 = tf.keras.layers.Input(shape=(300, ), name="other_act1")
        other_obs2 = tf.keras.layers.Input(shape=(300, ), name="other_obs2")
        other_act2 = tf.keras.layers.Input(shape=(300, ), name="other_act2")
        concat_obs = tf.keras.layers.Concatenate(axis=1)(
            [obs, other_obs1, other_act1, other_obs2, other_act2])
        central_vf_dense = tf.keras.layers.Dense(
            30, activation=tf.nn.tanh, name="c_vf_dense")(concat_obs)
        central_vf_out = tf.keras.layers.Dense(
            1, activation=None, name="c_vf_out")(central_vf_dense)
        self.central_vf = tf.keras.Model(
            inputs=[obs, other_obs1, other_act1, other_obs2, other_act2], outputs=central_vf_out)

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)


    def central_value_function(self, obs, other_obs1, other_actions1, other_obs2, other_actions2):
        return tf.reshape(
            self.central_vf([
                obs, other_obs1,
                tf.one_hot(tf.cast(other_actions1, tf.int32), 300),other_obs2,
                tf.one_hot(tf.cast(other_actions2, tf.int32), 300)
            ]), [-1])

    @override(ModelV2)
    def value_function(self):
        return self.model.value_function()  # not used




class CentralizedValueMixin:

    def __init__(self):
        if self.config["framework"] != "torch":
            self.compute_central_vf = make_tf_callable(self.get_session())(
                self.model.central_value_function)
        else:
            self.compute_central_vf = self.model.central_value_function


other_OBS1 = "other_obs1"
other_OBS2 = "other_obs1"
other_ACTION1 = "other_action1"
other_ACTION2 = "other_action2"


# Grabs the other obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(policy,
                                      sample_batch,
                                      other_agent_batches=None,
                                      episode=None):

    if policy.loss_initialized():
        assert other_agent_batches is not None

        other_batches = []

        for other_n_batch in other_agent_batches.values():
            other_batches.append(other_n_batch)


        #record the other obs and actions in the trajectory
        sample_batch[other_OBS1] = other_batches[0][1][SampleBatch.CUR_OBS]
        sample_batch[other_ACTION1] = other_batches[0][1][SampleBatch.ACTIONS]
        sample_batch[other_OBS2] = other_batches[1][1][SampleBatch.CUR_OBS]
        sample_batch[other_ACTION2] = other_batches[1][1][SampleBatch.ACTIONS]

        sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
            sample_batch[SampleBatch.CUR_OBS], sample_batch[other_OBS1],
            sample_batch[other_ACTION1], sample_batch[other_OBS2],
            sample_batch[other_ACTION2])

 
    else:
        # Policy hasn't been initialized yet, use zeros.
        sample_batch[other_OBS1] = np.zeros_like(
            sample_batch[SampleBatch.CUR_OBS])
        sample_batch[other_ACTION1] = np.zeros_like(
            sample_batch[SampleBatch.ACTIONS])
        sample_batch[other_OBS2] = np.zeros_like(
            sample_batch[SampleBatch.CUR_OBS])
        sample_batch[other_ACTION2] = np.zeros_like(
            sample_batch[SampleBatch.ACTIONS])    
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32)

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])



    return train_batch


def loss_with_central_critic(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)
    func = tf_loss

    vf_saved = model.value_function
    model.value_function = lambda: policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS], train_batch[other_OBS1],
        train_batch[other_ACTION1], train_batch[other_OBS2],
        train_batch[other_ACTION2])

    policy._central_value_out = model.value_function()
    loss = func(policy, model, dist_class, train_batch)

    model.value_function = vf_saved

    return loss



def setup_tf_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTFPolicy (w/o ValueNetworkMixin).
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def setup_torch_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTorchPolicy  (w/o ValueNetworkMixin).
    TorchKLCoeffMixin.__init__(policy, config)
    TorchEntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                       config["entropy_coeff_schedule"])
    TorchLR.__init__(policy, config["lr"], config["lr_schedule"])


def central_vf_stats(policy, train_batch, grads):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy._central_value_out)
    }


CCPPOTFPolicy = PPOTFPolicy.with_updates(
    name="CCPPOTFPolicy",
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=loss_with_central_critic,
    before_loss_init=setup_tf_mixins,
    grad_stats_fn=central_vf_stats,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        CentralizedValueMixin
    ])



def get_policy_class(config):
    if config["framework"] == "tf":
        return CCPPOTFPolicy

CCTrainer = PPOTrainer.with_updates(
    name="CCPPOTrainer_finale",
    default_policy=CCPPOTFPolicy,
    get_policy_class=get_policy_class,
)



from gym.spaces import MultiDiscrete, Dict, Discrete
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE




def mask(x):
    y = np.zeros(100)
    y[:x] = 1
    return y


def     quantinepuogestire(inv,max):
    if inv<=max:
        return inv
    else:
        return max        


dataset = pd.read_csv("csvwithinvocandmaxhere")
dataset1 = pd.read_csv("csvwithinvocandmaxhere")
dataset2 = pd.read_csv("csvwithinvocandmaxhere")




class DFaaSEnv(MultiAgentEnv):
    action_space = Discrete(100)
    observation_space =AGENT_OBS_SPACE

    def __init__(self, env_config):

        self.state = None
        self.agent_1 = 0
        self.agent_2 = 1
        self.agent_3 = 2

        # count manages iteration for CSVs
        self.count = 0

        self.invoc_1 = dataset["invoc_rate_norm_round"].tolist()
        self.invoc_2 = dataset1["invoc_rate_norm_round"].tolist()
        self.invoc_3 = dataset2["invoc_rate_norm_round"].tolist()

        self.invoc = [self.invoc_1,self.invoc_2,self.invoc_3]

        self.max_1 = dataset["max_rate_norm_round"].tolist()
        self.max_2 = dataset1["max_rate_norm_round"].tolist()
        self.max_3 = dataset2["max_rate_norm_round"].tolist()
        
        self.max = [self.max_1, self.max_2, self.max_3]
        
        # MADDPG emits action logits instead of actual discrete actions
        self.actions_are_logits = env_config.get("actions_are_logits", False)
        self.one_hot_state_encoding = env_config.get("one_hot_state_encoding",
                                                     False)
        self.with_state = env_config.get("separate_state_space", False)
        self.observation_space =AGENT_OBS_SPACE

        self.with_state = False



    def reset(self):
        self.state = np.array([1, 0, 0])
        return self._obs()




    def step(self, action_dict):

        
        indexes = [0,1,2]
        invoc = [self.invoc_1[self.count],self.invoc_2[self.count],self.invoc_3[self.count]]
        maxs = [self.max_1[self.count],self.max_2[self.count],self.max_3[self.count]]

        k=0.5

        gestiti = [(quantinepuogestire(invoc[0]-action_dict[0],maxs[0])), (quantinepuogestire(invoc[1]-action_dict[1],maxs[1])), (quantinepuogestire(invoc[2]-action_dict[2],maxs[2]))]

        capacita = [maxs[0]-gestiti[0], maxs[1]-gestiti[1],maxs[2]-gestiti[2]]
        persi = [invoc[0]-action_dict[0]-gestiti[0],invoc[1]-action_dict[1]-gestiti[1],invoc[2]-action_dict[2]-gestiti[2]]

        d = action_dict
        l = list(d.items())
        random.shuffle(l)
        d = dict(l)
        
        index_count = 0
        index = list(d.keys())[index_count]


        
        

        for action in d.values():

            split_action = int(action/2)
            indexes_copy = indexes.copy()
            restanti = 0
            indexes_copy.remove(index)
            primo_giro = True
            random.shuffle(indexes_copy)


            for indice in indexes_copy:

                split_action+=restanti
                restanti=0

                if capacita[indice]<split_action:
                    gestiti[index] += capacita[indice]
                    restanti+= split_action - capacita[indice]
                    capacita[indice] = 0

                else:
                    gestiti[index] += split_action
                    capacita[indice] -= split_action

                if primo_giro == False:
                    persi[index]+= int(restanti*k)

                primo_giro = not primo_giro    
                
            
            index = list(d.keys())[index_count+1]




        self.count = self.count +1

        if self.count>=2600:
            self.count = 0



        rewards = {
            self.agent_1: -persi[0],
            self.agent_2: -persi[1],
            self.agent_3: -persi[2]

        }

        obs = {
                self.agent_1: {
                "action_mask": mask(self.invoc_1[self.count]-1),
                "observation": [int(self.invoc_1[self.count]),int(self.max_1[self.count])]
                },
                self.agent_2: {
                "action_mask": mask(self.invoc_2[self.count]-1),
                "observation":[int(self.invoc_2[self.count]),int(self.max_2[self.count])]
                }, 
                self.agent_3: {
                "action_mask": mask(self.invoc_3[self.count]-1),
                "observation": [int(self.invoc_3[self.count]),int(self.max_3[self.count])]
                } 
            }
        dones = {"__all__": done}
        infos = {}


        return obs, rewards, dones, infos




    def _obs(self):
        return {
                self.agent_1: {
                "action_mask": mask(self.invoc_1[self.count]-1),
                "observation": [int(self.invoc_1[self.count]),int(self.max_1[self.count])]
                },
                self.agent_2: {
                "action_mask": mask(self.invoc_2[self.count]-1),
                "observation":[int(self.invoc_2[self.count]),int(self.max_2[self.count])]
                }, 
                self.agent_3: {
                "action_mask": mask(self.invoc_3[self.count]-1),
                "observation": [int(self.invoc_3[self.count]),int(self.max_3[self.count])]
                } 
        }


    





if __name__ == "__main__":
    ray.init(num_cpus=12, num_gpus=0)
    args = parser.parse_args()

    ModelCatalog.register_custom_model(
        "cc_model", DFaaSCriticModel)

    config = {
        "env": DFaaSEnv,
        "batch_mode": "truncate_episodes",
        "num_gpus": 0,
        "num_workers": 4,
        "rollout_fragment_length": 200,
        "train_batch_size": 4000,
        "multiagent": {
            "policies": {
                "shared_policy": (None, DFaaSEnv.observation_space, Discrete(100), {}),
            },
            "policy_mapping_fn": lambda aid, **kwargs: "shared_policy",
        },
        "model": {
            "custom_model": "cc_model",
        },
        "framework": args.framework,
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    results = tune.run(CCTrainer, config=config, stop=stop, verbose=1,checkpoint_at_end=True)


    if args.as_test:
        check_learning_achieved(results, args.stop_reward)