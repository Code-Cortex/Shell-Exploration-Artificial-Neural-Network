from keras.models import Sequential, load_model, save_model
from keras.layers import GRU, Dense
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT, TimeoutExpired
import numpy as np
import tensorflow as tf
import random
from keras.backend import clear_session
from gc import collect
from datetime import datetime

tf.get_logger().setLevel('ERROR')

# env adjustments
cmd = 'echo Hello World'
length_penalty = .25
learning_reward = 10
array_len = 10000
max_cmd = 1000

# model adjustments
hidden_layers = 8
layer_neurons = 128
nb_actions = 96
model_num = -1

# training adjustments
save_current_pool = True
total_models = 50
starting_fitness = 1

# variable assignment
current_pool = []
new_weights = []
best_weights = []
fitness = []
init = True
cmd_in = True
highest_fitness = -100
term_out = ''
error_count = 0
global e


def term_interact(cmd, cmd_in, model_num, init):
    global fitness
    global term_out
    if cmd_in:
        term_out = ''
        proc = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
        try:
            stdout = proc.communicate(timeout=1)[0].decode()
            exitcode = proc.returncode
        except TimeoutExpired:
            proc.kill()
            stdout = proc.communicate()[0].decode()
            exitcode = proc.returncode
        term_out = ''.join(char for char in stdout if char.isprintable())
        input_data = term_out + ' ' + str(Path.cwd()) + '> '
        filename = Path('mem.txt')
        filename.touch(exist_ok=True)
        if not term_out:
            term_out = 'Done!'
            input_data = term_out + ' ' + str(Path.cwd()) + '> '
            stdout = input_data
        if not init:
            if exitcode == 0:
                with open('mem.txt', 'r+') as mem:
                    for line in stdout.splitlines():
                        if line + '\n' not in mem:
                            mem.write(line + '\n')
                            fitness[model_num] += learning_reward
        print('\n')
        print(stdout)
        print(str(Path.cwd()) + '> ', end='', flush=True)
    else:
        input_data = term_out + ' ' + str(Path.cwd()) + '> ' + cmd
        print(input_data[-1], end='', flush=True)
        if not init:
            fitness[model_num] -= length_penalty
    idxs = np.swapaxes((np.atleast_3d((np.frombuffer(input_data.encode(), dtype=np.uint8) - 31) / 100)), 1, 2)
    if idxs.shape[2] < array_len:
        neural_input = np.append(idxs, np.zeros((1, 1, (array_len - idxs.shape[2]))), axis=2)
    else:
        neural_input = np.resize(idxs, (1, 1, array_len))
    return neural_input


def create_model(nb_actions):
    model = Sequential()
    for layer in range(hidden_layers):
        model.add(GRU(layer_neurons, name='GRU' + str(layer), return_sequences=True))
    model.add(GRU(layer_neurons, name='GRU' + str(hidden_layers)))
    model.add(Dense(nb_actions, name='output', activation='softmax'))
    model.compile(loss='mse', optimizer='adam')
    return model


def model_mutate(weights):
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            if random.uniform(0, 1) > .85:
                change = random.uniform(-.5, .5)
                weights[i][j] += change
    return weights


def model_crossover(parent1, parent2):
    # obtain parent weights
    # get random gene
    # swap genes
    global current_pool

    weight1 = current_pool[parent1].get_weights()
    weight2 = current_pool[parent2].get_weights()

    new_weight1 = weight1
    new_weight2 = weight2

    gene = random.randint(0, len(new_weight1) - 1)

    new_weight1[gene] = weight2[gene]
    new_weight2[gene] = weight1[gene]
    return np.asarray([new_weight1, new_weight2])


def save_pool():
    Path("SavedModels/").mkdir(parents=True, exist_ok=True)
    for xi in range(total_models):
        save_model(current_pool[xi], "SavedModels/model_new" + str(xi) + ".keras")
    print("Saved current pool!")


while True:
    try:
        if Path("SavedModels/").is_dir():
            for i in range(total_models):
                current_pool.append(load_model("SavedModels/model_new" + str(i) + ".keras"))
                fitness.append(starting_fitness)
        else:
            for i in range(total_models):
                model = create_model(nb_actions)
                fitness.append(starting_fitness)
                current_pool.append(model)
        while True:
            while model_num < total_models:
                prediction = current_pool[model_num].predict(term_interact(cmd, cmd_in, model_num, init))
                init = False
                if cmd_in:
                    cmd = ''
                    model_num += 1
                action = np.argmax(prediction)
                enc_ascii = action + 32
                if len(cmd) < max_cmd:
                    if enc_ascii != 127:
                        cmd += chr(enc_ascii)
                        cmd_in = False
                        continue
                    else:
                        cmd_in = True
                        continue
                else:
                    fitness[model_num] -= 100
                    cmd_in = True
                    continue
            model_num = 0

            parent1 = random.randint(0, total_models - 1)
            parent2 = random.randint(0, total_models - 1)

            for i in range(total_models):
                if fitness[i] >= fitness[parent1]:
                    parent1 = i

            for j in range(total_models):
                if j != parent1:
                    if fitness[j] >= fitness[parent2]:
                        parent2 = j
            for select in range(total_models):
                if fitness[select] >= highest_fitness:
                    updated = True
                    highest_fitness = fitness[select]
                    best_weights = current_pool[select].get_weights()
            for select in range(total_models // 2):
                cross_over_weights = model_crossover(parent1, parent2)
                updated = False
                if not updated:
                    cross_over_weights[1] = best_weights
                mutated1 = model_mutate(cross_over_weights[0])
                mutated2 = model_mutate(cross_over_weights[0])

                new_weights.append(mutated1)
                new_weights.append(mutated2)
            for select in range(total_models):
                fitness[select] = starting_fitness
                current_pool[select].set_weights(new_weights[select])
            if save_current_pool:
                save_pool()
    except Exception as e:
        logfile = Path('error_log.txt')
        logfile.touch(exist_ok=True)
        with open("error_log.txt", "a") as log:
            log.write(str(datetime.now()) + ' ' + str(e))
            log.write('\n')
        clear_session()
        collect()
        error_count += 1
        if error_count <= 10:
            continue
        else:
            print(e)
            break
