from keras.models import Model, Sequential
from keras.layers import Input, Concatenate, GRU, Dense, Reshape
from keras.optimizers import Adam
from keras.backend import clear_session
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT, TimeoutExpired
import numpy as np
import tensorflow as tf
from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy
import os

cmd = 'echo Hello World!'
env_reward = 0
length_penalty = .25
learning_reward = 10

hidden_layers = 4
layer_neurons = 128
learning_rate = 0.001
nb_actions = 96

tf.get_logger().setLevel('ERROR')

done = False
cmd_in = True
obs_last = None

while True:
    if cmd_in:
        proc = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
        try:
            stdout = proc.communicate(timeout=1)[0].decode()
            exitcode = proc.returncode
        except TimeoutExpired:
            proc.kill()
            stdout = proc.communicate()[0].decode()
            exitcode = proc.returncode
        nnin = ''.join(char for char in stdout if char.isprintable())
        filename = Path('mem.txt')
        filename.touch(exist_ok=True)
        if not nnin:
            nnin = 'Done!'
            stdout = nnin
        if exitcode == 0:
            done = True
            with open('mem.txt', 'r+') as mem:
                for line in stdout.splitlines():
                    if line + '\n' not in mem:
                        mem.write(line + '\n')
                        env_reward += learning_reward
        cmd = ''
        print('\n')
        print(stdout)
        print('# ', end='', flush=True)
    else:
        nnin = cmd
        print(nnin[-1], end='', flush=True)
        env_reward -= length_penalty
    idxs = (np.frombuffer(nnin.encode(), dtype=np.uint8) - 32) / 100
    env = tf.reshape(idxs, idxs.shape + (1,))
    shape = env.shape


    def build_actor_model(shape, nb_actions):
        model = Sequential()
        model.add(Reshape(shape[1::], input_shape=shape))
        for layer in range(hidden_layers):
            model.add(GRU(layer_neurons, name='GRU' + str(layer), return_sequences=True))
        model.add(GRU(layer_neurons, name='GRU' + str(hidden_layers)))
        model.add(Dense(nb_actions, name='output', activation='softmax'))
        return model


    def build_embed(shape, name_prefix='embed.'):
        inputs = Input(shape=shape)
        x = inputs
        for layer in range(hidden_layers):
            x = GRU(layer_neurons, name=name_prefix + ('GRU' + str(layer)), return_sequences=True)(x)
        x = GRU(layer_neurons, name=name_prefix + ('GRU' + str(hidden_layers)))(x)
        model = Model(inputs, x, name=name_prefix + 'embed')
        return model


    def build_inverse_model(obs1, obs2, nb_actions):
        x = Concatenate()([obs1.output, obs2.output])
        x = Dense(nb_actions, name='icm_i.output', activation='sigmoid')(x)
        i_model = Model([obs1.input, obs2.input], x, name='icm_inverse_model')
        return i_model


    def build_forward_model(obs1, nb_actions):
        act1 = Input(shape=nb_actions)
        x = Concatenate()([obs1.output, act1])
        output_shape = obs1.output_shape[1]
        x = Dense(output_shape, name='icm_f.output', activation='linear')(x)
        f_model = Model([obs1.input, act1], x, name='icm_forward_model')
        return f_model


    inv_weights_fname = '{}_inv_weights.h5f'.format("SMB")
    fwd_weights_fname = '{}_fwd_weights.h5f'.format("SMB")
    agent_weights_fname = '{}_agent_weights.h5f'.format("SMB")

    embed = build_embed(shape)
    embed2 = build_embed(shape, name_prefix='embed2.')
    inverse_model = build_inverse_model(embed, embed2, nb_actions)
    inverse_model.compile(Adam(learning_rate), loss='mse', metrics=['mse'])
    inverse_model.input_spec = None
    forward_model = build_forward_model(embed, nb_actions)
    forward_model.compile(Adam(learning_rate), loss='mse', metrics=['mse'])
    forward_model.input_spec = None
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
    inv_loss = inverse_model.train_on_batch([np.expand_dims(obs_last, 0), np.expand_dims(obs_now, 0)],
                                            [np.expand_dims(icm_action, 0)])
    features_now = embed.predict(np.expand_dims(obs_now, 0))
    fwd_loss = forward_model.train_on_batch([np.expand_dims(obs_last, 0), np.expand_dims(icm_action, 0)],
                                            [features_now])
    obs_last = obs_now
    r_intr = (fwd_loss[0] ** 0.5) / 100
    reward = r_intr + env_reward
    agent.backward(reward, done)
    clear_session()
    done = False

    enc_ascii = action + 32
    if enc_ascii != 127:
        cmd += chr(enc_ascii)
        cmd_in = False
        continue
    cmd_in = True
    inverse_model.save_weights(inv_weights_fname, overwrite=True)
    forward_model.save_weights(fwd_weights_fname, overwrite=True)
    agent.save_weights(agent_weights_fname, overwrite=True)
