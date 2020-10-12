from keras.models import Model, Sequential
from keras.layers import Input, Concatenate, GRU, Dense, Reshape
from keras.optimizers import Adam
from keras.backend import clear_session
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT, getstatusoutput
import signal
import numpy as np
import tensorflow as tf
from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy
import os

cmd = 'echo Hello World!'
env_reward = 0
error_penalty = 0
len_penalty = .1
repeat_penalty = 0
learning_reward = 10

hidden_layers = 2
IO_units = 256
hidden_units = 512
learning_rate = 0.005
fc1_units = 64
fc2_units = 128
fc3_units = 256
nb_actions = 97

tf.get_logger().setLevel('ERROR')
done = True
obs_last = None

while True:
    if not cmd:
        done = False
    if done:
        proc = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
        stdout = proc.stdout.read().decode('utf-8')
        exitcode = int(str(getstatusoutput(cmd))[1])
        nnin = ''.join(char for char in stdout if char.isprintable())
        filename = Path('mem.txt')
        filename.touch(exist_ok=True)
        if exitcode > 0:
            env_reward -= error_penalty
        else:
            with open('mem.txt', 'r+') as mem:
                for line in stdout.splitlines():
                    if line + '\n' not in mem:
                        mem.write(line + '\n')
                        env_reward += learning_reward
                    else:
                        env_reward -= repeat_penalty
        idxs = np.frombuffer(nnin.encode(), dtype=np.uint8) - 97
        env = tf.one_hot(idxs, 256)
        print('\n')
        print(stdout)
        print('# ', end='')
        cmd = ''
        shape = env.shape
    env_reward -= len_penalty


    def build_actor_model(shape, nb_actions):
        model = Sequential()
        model.add(Reshape(shape[1::], input_shape=shape))
        model.add(GRU(IO_units, input_shape=shape, name='GRU1', return_sequences=True))
        for layer in range(2, hidden_layers):
            model.add(GRU(hidden_units, name='GRU' + str(layer), return_sequences=True))
        model.add(GRU(IO_units, name='GRU' + str(hidden_layers)))
        model.add(Dense(fc1_units, name='fc1', activation='relu'))
        model.add(Dense(fc2_units, name='fc2', activation='relu'))
        model.add(Dense(nb_actions, name='output', activation='softmax'))
        return model


    def build_main(shape, name_prefix='main.'):
        inputs = Input(shape=shape)
        x = inputs
        x = GRU(IO_units, name=name_prefix + 'GRU1', return_sequences=True)(x)
        for layer in range(2, hidden_layers):
            x = GRU(hidden_units, name=name_prefix + ('GRU' + str(layer)), return_sequences=True)(x)
        x = GRU(IO_units, name=name_prefix + ('GRU' + str(hidden_layers)))(x)
        model = Model(inputs, x, name=name_prefix + 'main')
        return model


    def build_inverse_model(main1, main2, nb_actions):
        obs1 = main1
        obs2 = main2
        x = Concatenate()([obs1.output, obs2.output])
        x = Dense(fc1_units, name='icm_i.fc1', activation='relu')(x)
        x = Dense(fc2_units, name='icm_i.fc2', activation='relu')(x)
        x = Dense(nb_actions, name='icm_i.output', activation='sigmoid')(x)
        i_model = Model([obs1.input, obs2.input], x, name='icm_inverse_model')
        return i_model


    def build_forward_model(main, nb_actions):
        obs1 = main
        act1 = Input(shape=nb_actions)
        x = Concatenate()([obs1.output, act1])
        x = Dense(fc1_units, name='icm_f.fc1', activation='relu')(x)
        x = Dense(fc2_units, name='icm_f.fc2', activation='relu')(x)
        x = Dense(fc3_units, name='icm_f.fc3', activation='relu')(x)
        output_shape = obs1.output_shape[1]
        x = Dense(output_shape, name='icm_f.output', activation='linear')(x)
        f_model = Model([obs1.input, act1], x, name='icm_forward_model')
        return f_model


    inv_weights_fname = '{}_inv_weights.h5f'.format("SMB")
    fwd_weights_fname = '{}_fwd_weights.h5f'.format("SMB")
    agent_weights_fname = '{}_agent_weights.h5f'.format("SMB")

    main = build_main(shape)
    main2 = build_main(shape, name_prefix='main2.')
    inverse_model = build_inverse_model(main, main2, nb_actions)
    inverse_model.compile(Adam(learning_rate), loss='mse', metrics=['mse'])
    forward_model = build_forward_model(main, nb_actions)
    forward_model.compile(Adam(learning_rate), loss='mse', metrics=['mse'])
    model = build_actor_model((1,) + shape, nb_actions)
    policy = BoltzmannQPolicy()
    agent = SARSAAgent(model=model, nb_actions=nb_actions, policy=policy)
    agent.compile(Adam(learning_rate), metrics=['mae'])
    agent.reset_states()

    if os.path.isfile(inv_weights_fname):
        inverse_model.load_weights(inv_weights_fname)
    if os.path.isfile(fwd_weights_fname):
        forward_model.load_weights(fwd_weights_fname)
    if os.path.isfile(agent_weights_fname):
        agent.load_weights(agent_weights_fname)
        agent.training = True

    obs_now = env
    if obs_last is None:
        obs_last = obs_now
    action = agent.forward(obs_now)
    icm_action = np.zeros(nb_actions)
    icm_action[action] = 1
    inv_loss = inverse_model.train_on_batch([np.expand_dims(obs_last, 0), np.expand_dims(obs_now, 0)],[np.expand_dims(icm_action, 0)])
    features_now = main.predict(np.expand_dims(obs_now, 0))
    fwd_loss = forward_model.train_on_batch([np.expand_dims(obs_last, 0), np.expand_dims(icm_action, 0)],[features_now])
    obs_last = obs_now
    r_intr = (fwd_loss[0] ** 0.5) / 100
    reward = r_intr + env_reward
    agent.backward(reward, done)
    clear_session()

    enc_ascii = action + 32
    if enc_ascii != 128:
        if enc_ascii < 127:
            cmd += chr(enc_ascii)
        if enc_ascii == 127:
            proc.send_signal(signal.CTRL_C_EVENT if os.name == 'nt' else signal.SIGINT)
        print(cmd[-1], end='')
        done = False
        continue
    inverse_model.save_weights(inv_weights_fname, overwrite=True)
    forward_model.save_weights(fwd_weights_fname, overwrite=True)
    agent.save_weights(agent_weights_fname, overwrite=True)
    done = True
