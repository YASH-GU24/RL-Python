# To implement a gRPC server for a reinforcement learning (RL) system

import grpc
from concurrent import futures
import format_pb2_grpc
import format_pb2
import logging
from neural_network import model
import numpy as np
import torch


# This RLServer class likely serves as the server-side component for a reinforcement learning (RL) system. It initializes an agent, which is responsible for interacting with and learning from an environment.

class RLServer(format_pb2_grpc.UnaryServicer):
    def __init__(self) -> None:
        super().__init__()
        self.old_observation = None   # Holds the previous observation
        self.action = None            # Holds the Action chosen by the Agent
        self.agent = model.Agent(     # Creation of instance of Agent class from the model module
            gamma = 0.99,     # Discount factor for future Rewards
            batch_size = 128, # Size of the batch used for training the agent
            n_actions = 4,    # Number of possible actions
            epsilon_greedy = True,
            epsilon = 1.0,   # Exploration probabilty at the Start
            eps_end = 0.1,   # Minimum Exploration probability
           # eps_decay = 0.0005 # Exponential decay rate for exploration prob
            eps_decay = 0.09,
            input_dims = [6], # Dimension of Input Space
            lr = 3e-3         # Learning Rate of Agent Optimizer
        )
        self.scores, self.eps_history = [], []
        
    def GetActionRL(self, request, context):
         # Extracting raw input values from the request
        taskLength = request.taskLength
        taskMaxLatency = request.taskMaxLatency
        localCPU = request.localCPU
        localMIPSTerm = request.localMIPSTerm
        edgeCPUTerm = request.edgeCPUTerm
        cloudCPUTerm = request.cloudCPUTerm
        
        print("Task Length:", taskLength)
        print("Task Max Latency:", taskMaxLatency)
        print("Local CPU:", localCPU)
        print("Local MIPS Term:", localMIPSTerm)
        print("Edge CPU Term:", edgeCPUTerm)
        print("Cloud CPU Term:", cloudCPUTerm)
    
        #exploration_probability = request.exploration_probability
        #iteration = request.iteration
        
        #numberOfPes = request.numberOfPes
        #fileSize = request.fileSize
        #outputSize = request.outputSize
        #containerSize = request.containerSize
        #maxLatency = request.maxLatency
        
        #print("Before Normalization State values",self.taskLength,self.taskMaxLatency, self.localCPU, self.localMIPSTerm, self.edgeCPUTerm, self.cloudCPUTerm)

        
        taskLength_min = 15000
        taskLength_max = 300000
        taskMaxLatency_min = 4
        taskMaxLatency_max = 30
        localCPU_min = 0
        localCPU_max = 100
        localMIPSTerm_min = 16000
        localMIPSTerm_max = 130000
        edgeCPUTerm_min = 0
        edgeCPUTerm_max = 100
        cloudCPUTerm_min = 0
        cloudCPUTerm_max = 100

        # Normalize the input values
        normalized_taskLength = (taskLength - taskLength_min) / (taskLength_max - taskLength_min)
        normalized_taskMaxLatency = (taskMaxLatency - taskMaxLatency_min) / (taskMaxLatency_max - taskMaxLatency_min)
        normalized_localCPU = (localCPU - localCPU_min) / (localCPU_max - localCPU_min)
        normalized_localMIPSTerm = (localMIPSTerm - localMIPSTerm_min) / (localMIPSTerm_max - localMIPSTerm_min)
        normalized_edgeCPUTerm = (edgeCPUTerm - edgeCPUTerm_min) / (edgeCPUTerm_max - edgeCPUTerm_min)
        normalized_cloudCPUTerm = (cloudCPUTerm - cloudCPUTerm_min) / (cloudCPUTerm_max - cloudCPUTerm_min)
    
        #print("Normalized State values:", normalized_taskLength, normalized_taskMaxLatency, normalized_localCPU,
        #normalized_localMIPSTerm, normalized_edgeCPUTerm, normalized_cloudCPUTerm)

        self.old_observation = torch.tensor(
            [
                normalized_taskLength,
                normalized_taskMaxLatency,
                normalized_localCPU,
                normalized_localMIPSTerm,
                normalized_edgeCPUTerm,
                normalized_cloudCPUTerm,
                #numberOfPes,
                #fileSize,
                #outputSize,
                #containerSize,
                #maxLatency,
            ],
            dtype=torch.float32,
        )
        
        #decay_step = 0
        #for decay_step in range(0,10):
        
        self.agent.Q_eval
        self.action = self.agent.choose_action(observation=self.old_observation)
        self.agent.Q_eval.train()
        return format_pb2.Action(action=self.action)
        
        
                #self.agent.Q_eval.eval()
        #self.action = self.agent.choose_action(observation = self.old_observation, decay_step = decay_step)
           # decay_step += 1
          #  print("Value of decay_step=", decay_step)
        

    def TrainModelRL(self, request, context):
        new_observation = request.new_state
        reward = request.reward
        done = request.is_done

        if reward != 0.0:
            self.new_observation = np.array(
                [
                    new_observation.taskLength,
                    new_observation.taskMaxLatency,
                    new_observation.localCPU,
                    new_observation.localMIPSTerm,
                    new_observation.edgeCPUTerm,
                    new_observation.cloudCPUTerm,
                    #new_observation.numberOfPes,
                    #new_observation.fileSize,
                    #new_observation.outputSize,
                    #new_observation.containerSize,
                    #new_observation.maxLatency,
                ]
            )

            self.scores += [reward]
            self.agent.store_transition(
                self.old_observation,
                action=self.action,
                reward=reward,
                state_=self.new_observation,
                done=done,
            )

            # if random.random() > 0.4:
            self.agent.learn()
            #print("Value of Random=", random.random()))
            print(f"Counter: {self.agent.counter}\tReward: {reward}")
            return format_pb2.Response(message="the model trained!")

        else:
            return format_pb2.Response(message="the model didn't trained!")


def serve():
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    format_pb2_grpc.add_UnaryServicer_to_server(RLServer(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()
