import retro
import numpy as np
import cv2
import neat
import pickle
import time
import threading


# create game environment
env = retro.make('KingOfFightersEX2HowlingBlood-GbAdvance', obs_type=retro.Observations.IMAGE)
print("Acciones posibles: ", env.action_space.n)
print("Accion aleatoria: ", env.action_space.sample())

# create a window to show the gameplay
cv2.namedWindow('KOF GBA | NEAT-Python | Evair', cv2.WINDOW_NORMAL)
cv2.moveWindow("KOF GBA | NEAT-Python | Evair", 950, 120)
cv2.resizeWindow('KOF GBA | NEAT-Python | Evair', 800,600)

def show(observation):
    shwimg = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
    cv2.imshow('KOF GBA | NEAT-Python | Evair', shwimg)
    cv2.waitKey(1)
    
# generation
generation = -1

# funtion to evaluate genomes during training process
def eval_genomes(genomes, config):
    # generation
    global generation
    generation += 1

    for genome_id, genome in genomes:
        #######################################################################################################
        ###########################################control variables###########################################
        #######################################################################################################

        log = True
        log_size = 300

        #######################################################################################################
        ###########################################control variables###########################################
        #######################################################################################################

        # get environment print screen
        observation = env.reset()
        
        # set shape size to input in neural network
        inx, iny, _ = env.observation_space.shape
        inx = int(inx/5) #28
        iny = int(iny/5) #40
        
        #input_size = inx * iny * 2
        
        # create the neural network
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        # initialize variables
        done = False
        fitness_current = 0
        fitness_current_max = 0
        counter = 0
        frame = 0

        imgarray = []
       
        # main loop
        while not done:

            # frame count
            frame += 1
            
            env.render()
            
            # show gameplay
            #show(observation)
        
            # prepare the print screen to use as neural network input
            observation = cv2.resize(observation, (inx, iny))
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
       
 
            
            input_data = np.ndarray.flatten(observation)    

            # process the print screen through the neural network to obtain the output (actions)
            nnOutput = net.activate(input_data)
        
            #print(nnOutput)
            _, _, _, info = env.step(nnOutput)
            health_before_action = info['health']
            enemy_before = info["enemy_health"]
            
            # apply actions to game environment to get new observation (print screen), reward gain by applying the action, and current values of info parameters (data.json)
            observation, reward, done, info = env.step(nnOutput)
            #print(nnOutput)
            enemy_after = info["enemy_health"]
            health_after_action = info['health']

            fitness_current += reward

            # set counter to stop
            if fitness_current > fitness_current_max:
                fitness_current_max = fitness_current
                counter = 0
            else:
                counter += 1
                
            rounds = info['round_p2']
            rounds2 = info['round_p1']

            diference = health_before_action - health_after_action
            if health_before_action > health_after_action:
                # A saúde do jogador 1 diminuiu após a ação do jogador 1
                if diference <= 80:
                    reward -= diference
                    print("perdeu ponto >>", f"{diference}")
           
                #print("perdeu essa vida mas nao creditou ponto >>", f"{diference}")
            diference2 = enemy_before - enemy_after
            if enemy_before > enemy_after:
                #A saúde do jogador 1 diminuiu após a ação do jogador 1
                if diference2 <= 80:
                    reward += diference2
                    print("ganhou ponto >>", f"{diference2}")

            # set genome fitness to train the neural network
            genome.fitness = fitness_current

            # counter to stop
            reason_stoped = ''
            if counter > 750:
                reason_stoped = 'Maximum frames without reward'
                reward -= 10.0
                done = True

                
            # if die stop
            if rounds == 1:
                reason_stoped = 'died'
                reward -= 20.0
                done = True
                
            # if die stop
            if rounds2 == 1:
                reward += 10.0
                reason_stoped = 'killed the enemy'
                #done = True
                     
            #print(reward)
            # logs
            if log and frame % log_size == 0:
                print('generation: ', generation, 'genome_id: ', genome_id, 'frame: ', frame, 'counter', counter, 'fitness_current: ', fitness_current)
            if done and log:
                print('------------------------------------------------------------------------------------------------------------------')
                print('generation: ', generation, 'genome_id: ', genome_id, 'frame: ', frame, 'counter', counter, 'fitness_current: ', fitness_current, 'Reason: ', reason_stoped)
                print('------------------------------------------------------------------------------------------------------------------')

# load configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-neat_kof')

# set configuration
p = neat.Population(config)
#p = neat.Checkpointer.restore_checkpoint('./neat-checkpoint-15')

# report trainning
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10)) #every x generations save a checkpoint

# run trainning
winner = p.run(eval_genomes)

# save the winner
with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

# close environment
env.close()

