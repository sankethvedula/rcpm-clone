#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import sys

import flax.optim
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Neutral', call_pdb=1)

import numpy as np

import jax
import jax.numpy as jnp
from jax.config import config; config.update("jax_enable_x64", True)


import pickle as pkl

# from flax import linen as nn
# from flax import optim

import time

# import hydra

import csv
import os

import functools

# import flows
# import utils
import densities
from manifolds import Sphere
from flows import SequentialFlow, InfAffine, ExpMapFlow

from setproctitle import setproctitle
setproctitle('iccnn')


def kl_ess(log_model_prob, log_target_prob):
    weights = jnp.exp(log_target_prob) / jnp.exp(log_model_prob)
    Z = jnp.mean(weights)
    KL = jnp.mean(log_model_prob - log_target_prob) + jnp.log(Z)
    ESS = jnp.sum(weights) ** 2 / jnp.sum(weights ** 2)
    return Z, KL, ESS


class Workspace:
    def __init__(
        self,
        loss_type: str = "likelihood",
        eval_samples: int = 20000,
        batch_size: int = 256,
        iterations: int = 1e6,
    ):
        self.loss_type = loss_type
        self.iterations = iterations
        self.batch_size = batch_size
        self.work_dir = os.getcwd()
        self.eval_samples = eval_samples
        print(f'workspace: {self.work_dir}')

        # self.manifold = hydra.utils.instantiate(self.cfg.manifold)
        self.manifold = Sphere(D=3)

        self.base = densities.SphereUniform(self.manifold)
        self.target = densities.LouSphereFourModes(self.manifold)

        self.key = jax.random.PRNGKey(42)

        potential = InfAffine(n_components=68,
                              init_alpha_mode="uniform",
                              init_alpha_linear_scale=1.0,
                              init_alpha_minval=0.4,
                              init_alpha_range=0.01,
                              cost_gamma=0.1,
                              min_zero_gamma=0.0,
                              manifold=self.manifold)
        single_transform = ExpMapFlow(potential_=potential, manifold=self.manifold)
        self.flow = SequentialFlow(
            n_transforms=1,
            manifold=self.manifold,
            single_transform=single_transform
        )

        self.key, k1, k2, k3, k4, k5 = jax.random.split(self.key, 6)

        batch = self.base.sample(k1, self.batch_size)
        init_params = self.flow.init(k2, batch)

        self.base_samples = self.base.sample(k3, self.eval_samples)
        self.base_log_probs = self.base.log_prob(self.base_samples)
        if loss_type == 'likelihood':
            self.eval_target_samples = self.target.sample(
                k5, self.eval_samples)

        optimizer_def = flax.optim.Adam(learning_rate=1e-3, beta1=0.9, beta2=0.999)
        self.optimizer = optimizer_def.create(init_params)

        self.iter = 0

    def run(self):
        if self.loss_type == 'kl':
            self.train_kl()
        elif self.loss_type == 'likelihood':
            self.train_likelihood()
        else:
            assert False

    def train_kl(self):
        @jax.jit
        def loss(params, base_samples, base_log_probs):
            z, ldjs = self.flow.apply(params, base_samples)
            loss =  (base_log_probs - ldjs -
                     self.target.log_prob(z)).mean()
            return loss

        @jax.jit
        def update(optimizer, base_samples, base_log_probs):
            l, grads = jax.value_and_grad(loss)(
                optimizer.target, base_samples, base_log_probs)
            optimizer = optimizer.apply_gradient(grads)
            return l, optimizer

        logf, writer = self._init_logging()

        times = []
        if self.iter == 0:
            model_samples, ldjs = self.flow.apply(
                self.optimizer.target, self.base_samples)
            self.manifold.plot_samples(
                model_samples, save=f'{self.iter:06d}.png')

            self.manifold.plot_density(self.target.log_prob, 'target.png')

        while self.iter < self.iterations:
            start = time.time()
            self.key, subkey = jax.random.split(self.key)
            base_samples = self.base.sample(subkey, self.batch_size)
            base_log_probs = self.base.log_prob(base_samples)
            l, self.optimizer = update(
                self.optimizer, base_samples, base_log_probs)

            times.append(time.time() - start)
            self.iter += 1
            if self.iter % 1000 == 0:
                l = loss(self.optimizer.target,
                         self.base_samples, self.base_log_probs)

                model_samples, ldjs = self.flow.apply(
                    self.optimizer.target, self.base_samples)
                self.manifold.plot_samples(
                    model_samples, save=f'{self.iter:06d}.png')
                if False:
                    for i, t in enumerate(jnp.linspace(0.1,1,11)):
                        model_samples, ldjs = self.flow.apply(
                        self.optimizer.target, self.base_samples, t = t)
                        self.manifold.plot_samples(
                            model_samples,
                            save=f'{self.iter:06d}_{i}.png')


                log_prob = self.base_log_probs - ldjs
                _,  kl, ess = kl_ess(
                    log_prob, self.target.log_prob(model_samples))
                ess = ess / self.eval_samples * 100
                msg = "Iter {} | Loss {:.3f} | KL {:.3f} | ESS {:.2f}% | {:.2e}s/it"
                print(msg.format(
                    self.iter, l, kl, ess, jnp.mean(jnp.array(times))))
                writer.writerow({
                    'iter': self.iter, 'loss': l, 'kl': kl, 'ess': ess
                })
                logf.flush()
                self.save('latest')

                times = []


    def train_likelihood(self):
        @jax.jit
        def logprob(params, target_samples, t = 1):
            zs, ldjs = self.flow.apply(params, target_samples, t = t)
            log_prob = ldjs + self.base.log_prob(zs)
            return log_prob

        @jax.jit
        def loss(params, target_samples):
            return -logprob(params, target_samples).mean()

        @jax.jit
        def update(optimizer, target_samples):
            l, grads = jax.value_and_grad(loss)(
                optimizer.target, target_samples)
            optimizer = optimizer.apply_gradient(grads)
            return l, optimizer

        target_sample_jit = jax.jit(self.target.sample, static_argnums=(1,))
        base_sample_jit = jax.jit(self.base.sample, static_argnums=(1,))

        logf, writer = self._init_logging()

        times = []

        if self.iter == 0 and False:
            model_samples, ldjs = self.flow.apply(
                self.optimizer.target, self.eval_target_samples)
            try:
                self.manifold.plot_density(
                 self.target.log_prob, save=f'target_density.png')
            except:
                pass
            self.manifold.plot_samples(
                self.eval_target_samples, save=f'target_samples.png')
            self.manifold.plot_samples(
                base_sample_jit(self.key, self.eval_samples),
                save=f'base_samples.png')
            self.manifold.plot_density(
                self.base.log_prob, save=f'base_density.png')
            self.manifold.plot_samples(
                model_samples, save=f'samples_{self.iter:06d}.png')
            self.manifold.plot_density(
                functools.partial(logprob, self.optimizer.target),
                save=f'density_{self.iter:06d}.png')
            # if False:
            #     for i, t in enumerate(jnp.linspace(0.1,1,11)):
            #         self.manifold.plot_density(
            #             functools.partial(logprob, self.optimizer.target, t = t),
            #             save=f'density_{self.iter:06d}_{i}.png')


        while self.iter < self.iterations:
            start = time.time()
            self.key, subkey = jax.random.split(self.key)
            target_samples = target_sample_jit(subkey, self.batch_size)
            l, self.optimizer = update(self.optimizer, target_samples)

            times.append(time.time() - start)
            self.iter += 1
            if self.iter % 1000 == 0:
                l = loss(self.optimizer.target, self.eval_target_samples)
                model_samples, ldjs = self.flow.apply(
                    self.optimizer.target, self.eval_target_samples)
                self.manifold.plot_samples(
                    model_samples, save=f'samples_{self.iter:06d}.png')
                self.manifold.plot_density(
                    functools.partial(logprob, self.optimizer.target),
                    save=f'density_{self.iter:06d}.png')
                if False:
                    for i, t in enumerate(jnp.linspace(0.1,1,10)):
                        self.manifold.plot_density(
                            functools.partial(logprob, self.optimizer.target, t = t),
                            save=f'density_{self.iter:06d}_{i}.png')



                msg = "Iter {} | Loss {:.3f} | {:.2e}s/it"
                print(msg.format(
                    self.iter, l, jnp.mean(jnp.array(times))))
                writer.writerow({
                    'iter': self.iter, 'loss': l,
                })
                logf.flush()
                self.save('latest')
                times = []



    def save(self, tag='latest'):
        path = os.path.join(self.work_dir, f'{tag}.pkl')
        with open(path, 'wb') as f:
            pkl.dump(self, f)


    def _init_logging(self):
        logf = open('log.csv', 'a')
        fieldnames = ['iter', 'loss', 'kl', 'ess']
        writer = csv.DictWriter(logf, fieldnames=fieldnames)
        if os.stat('log.csv').st_size == 0:
            writer.writeheader()
            logf.flush()
        return logf, writer


# Import like this for pickling
from main import Workspace as W

def main():
    fname = os.getcwd() + '/latest.pt'
    if os.path.exists(fname):
        print(f'Resuming fom {fname}')
        with open(fname, 'rb') as f:
            workspace = pkl.load(f)
    else:
        workspace = W()

    workspace.run()

if __name__ == '__main__':
    main()
